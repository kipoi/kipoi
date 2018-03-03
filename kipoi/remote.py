from __future__ import absolute_import
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty
import sys
import os
import six
import subprocess
import logging
from collections import OrderedDict
from .utils import lfs_installed, get_file_path, cd, list_files_recursively
from .components import ModelDescription, DataLoaderDescription
import pandas as pd
import kipoi
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# TODO - optionally don't pull the recent files?

def get_component_file(component_dir, which="model"):
    # TODO - if component_dir has an extension, then just return that file path
    return get_file_path(component_dir, which, extensions=[".yml", ".yaml"])


def list_yamls_recursively(root_dir, basename):
    return [os.path.dirname(x) for x in list_files_recursively(root_dir, basename, suffix='y?ml')]


def list_softlink_realpaths(root_dir):
    for root, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            fname = os.path.join(root, name)
            if os.path.islink(fname):
                yield os.path.realpath(fname)


def load_component_descr(component_path, which="model"):
    """Return the parsed yaml file
    """
    with cd(os.path.dirname(component_path)):
        if which == "model":
            return ModelDescription.load(os.path.basename(component_path))
        elif which == "dataloader":
            return DataLoaderDescription.load(os.path.basename(component_path))
        else:
            raise ValueError("which needs to be from {'model', 'dataloader'}")


def list_softlink_dependencies(component_dir, source_path):
    """List dependencies of a directory

    Returns a set
    """
    return {relative_path(f, source_path) if os.path.isdir(f)
            else relative_path(os.path.dirname(f), source_path)
            for f in list_softlink_realpaths(component_dir)
            if is_subdir(f, source_path)}


def is_subdir(path, directory):
    """Check if the path is in a particular directory
    """
    path = os.path.realpath(path)
    directory = os.path.realpath(directory)
    relative = os.path.relpath(path, directory)
    return not (relative == os.pardir or relative.startswith(os.pardir + os.sep))


def relative_path(path, directory):
    path = os.path.realpath(path)
    assert directory != ""
    directory = os.path.realpath(directory)
    relative = os.path.relpath(path, directory)
    return relative


def get_model_descr(model, source="kipoi"):
    """Get model description

    # Arguments
      model: model's relative path/name in the source. 2nd column in the `kipoi.list_models() `pd.DataFrame`.
      source: Model source. 1st column in the `kipoi.list_models()` `pd.DataFrame`.
    """
    return kipoi.config.get_source(source).get_model_descr(model)


def get_dataloader_descr(dataloader, source="kipoi"):
    """Get dataloder description

    # Arguments
      datalaoder: dataloader's relative path/name in the source. 2nd column in the `kipoi.list_dataloader() `pd.DataFrame`.
      source: Model source. 1st column in the `kipoi.list_models()` `pd.DataFrame`.
    """
    return kipoi.config.get_source(source).get_dataloader_descr(dataloader)


def to_namelist(nested_dict):
    """no-recursion
    """
    if isinstance(nested_dict, list):
        return [x.name for x in nested_dict]
    elif isinstance(nested_dict, dict):
        return list(nested_dict)
    else:
        return nested_dict.name


class Source(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def TYPE(self):
        pass

    @abstractmethod
    def _list_components(self, which="model"):
        """List available models a strings
        """
        pass

    @abstractmethod
    def _pull_component(self, component, which="model"):
        """Pull/update the model locally and
        returns a local path to it
        """
        return

    def pull_model(self, model):
        return self._pull_component(model, "model")

    def pull_dataloader(self, dataloader):
        return self._pull_component(dataloader, "dataloader")

    def list_models(self):
        """List all the models as a data.frame
        """
        def dict2df_dict(d, model):
            return OrderedDict([
                ("model", model),
                # ("name", d.info.name),
                ("version", d.info.version),
                ("authors", d.info.authors),
                ("contributors", d.info.contributors),
                ("doc", d.info.doc),
                ("type", d.type),
                ("inputs", to_namelist(d.schema.inputs)),
                ("targets", to_namelist(d.schema.targets)),
                ("postproc_score_variants", d.postprocessing.variant_effects is not None),
                ("license", d.info.license),
                ("cite_as", d.info.cite_as),
                ("trained_on", d.info.trained_on),
                ("training_procedure", d.info.training_procedure),
                ("tags", d.info.tags),
            ])

        df = pd.DataFrame([dict2df_dict(self.get_model_descr(model), model)
                           for model in self._list_components("model")])
        if len(df):
            # filter all template models
            return df[~df.model.str.contains("/template$")]
        else:
            return df

    def list_dataloaders(self):
        """List all the models as a data.frame
        """
        def dict2df_dict(d, dataloader):
            return OrderedDict([
                ("dataloader", dataloader),
                # ("name", d.info.name),
                ("version", d.info.version),
                ("authors", d.info.authors),
                ("doc", d.info.doc),
                ("type", d.type),
                ("inputs", to_namelist(d.output_schema.inputs)),
                ("targets", to_namelist(d.output_schema.targets)),
                ("license", d.info.license),
                ("tags", d.info.tags),
            ])

        df = pd.DataFrame([dict2df_dict(self.get_dataloader_descr(dataloader), dataloader)
                           for dataloader in self._list_components("dataloader")])
        # filter all template models
        return df[~df.dataloader.str.contains("/template$")]

    def list_models_by_group(self, group_filter=""):
        """Get a list of models by a group

        Args:
          group_filter, str: A relative path to the model group used to subset
            model list.

        Returns:
          a pd.DataFrame with columns (or None in case no groups are found):
            - group - name of the sub-group
            - N_models
            - is_group
            - authors
            - contributors
            - doc - ?
            - type (list of types)
            - tags - sum
        """
        df = self.list_models()

        # add slashes
        if group_filter == "":
            group = "/"
        else:
            group = "/" + group_filter + "/"
        df = df[df.model.str.contains("^" + group[1:])].copy()
        # df['parent_group'] = group[1:]
        df['model'] = df.model.str.replace("^" + group[1:], "")
        df['is_group'] = df.model.str.contains("/")
        if not df.is_group.any():
            return None

        df = df.join(df.model.str.split("/", n=1, expand=True).rename(columns={0: "group", 1: "child"}))

        def n_subgroups(ch):
            if ch.str.contains("/").sum() > 0:
                return len(ch[ch.str.contains("/")].str.split("/", n=1, expand=True).iloc[:, 0].unique())
            else:
                return 0

        def fn(x):
            # remove the templates
            return pd.Series(OrderedDict([
                ("N_models", x.shape[0]),
                ("N_subgroups", n_subgroups(x.child.fillna(""))),
                ("is_group", x.is_group.any()),
                ("authors", {author for authors in x.authors
                             for author in authors}),
                ("contributors", {contributor for contributors in x.contributors
                                  for contributor in contributors}),
                ("postproc_score_variants", x.postproc_score_variants.any()),
                ("type", {t for t in x.type}),
                ("license", {l for l in x.license}),
                ("cite_as", {c for c in x.cite_as if c is not None}),
                ("tags", {tag for tags in x.tags
                          for tag in tags}),
            ]))

        return df.groupby("group").apply(fn).reset_index()

    @abstractmethod
    def _get_component_descr(self, component, which="model"):
        pass

    def get_model_descr(self, model):
        return self._get_component_descr(model, which="model")

    def get_dataloader_descr(self, dataloader):
        return self._get_component_descr(dataloader, which="dataloader")

    @abstractmethod
    def get_config(self):
        pass

    @classmethod
    def from_config(cls, config):
        assert config.pop("type") == cls.TYPE

        return cls(**config)

    def __repr__(self):
        conf = self.get_config()
        cls_name = self.__class__.__name__
        conf.pop("type")

        kwargs = ', '.join('{0}={1}'.format(k, repr(v))
                           for k, v in six.iteritems(conf))
        return "{0}({1})".format(cls_name, kwargs)


class GitLFSSource(Source):

    TYPE = "git-lfs"

    def __init__(self, remote_url, local_path):
        """GitLFS Source
        """
        lfs_installed(raise_exception=False)
        self.remote_url = remote_url
        self.local_path = os.path.join(local_path, '')  # add trailing slash
        self._pulled = False

    def _list_components(self, which="model"):
        if not self._pulled:
            self.pull_source()
        return list_yamls_recursively(self.local_path, which)

    def clone(self):
        """Clone the self.remote_url into self.local_path
        """
        lfs_installed(raise_exception=True)
        if os.path.exists(self.local_path) and os.listdir(self.local_path):
            raise IOError("Directory {0} already exists and is non-empty".
                          format(self.local_path))

        logger.info("Cloning {remote} into {local}".
                    format(remote=self.remote_url,
                           local=self.local_path))
        subprocess.call(["git",
                         "clone",
                         self.remote_url,
                         self.local_path],
                        env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))
        self._pulled = True

    def pull_source(self):
        """Pull/update the source
        """
        lfs_installed(raise_exception=True)
        if not os.path.exists(self.local_path) or not os.listdir(self.local_path):
            return self.clone()

        logger.info("Update {0}".
                    format(self.local_path))
        subprocess.call(["git",
                         "pull"],
                        cwd=self.local_path,
                        env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))
        self._pulled = True

    def _pull_component(self, component, which="model"):
        lfs_installed(raise_exception=True)
        if not self._pulled:
            self.pull_source()

        component_dir = os.path.join(self.local_path, component)

        # get a list of directories to source (relative to the local_path)
        softlink_dirs = list(list_softlink_dependencies(component_dir, self.local_path))

        cpath = get_component_file(component_dir, which)
        if not os.path.exists(cpath):
            raise ValueError("{0}: {1} doesn't exist in {2}".
                             format(component, self.remote_url))

        for pull_dir in [component] + softlink_dirs:
            cmd = ["git-lfs",
                   "pull",
                   "-I {0}/**".format(pull_dir)]
            logger.info(" ".join(cmd))
            subprocess.call(cmd,
                            cwd=self.local_path,
                            env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))
        logger.info("{0} {1} loaded".format(which, component))
        return cpath

    def _get_component_descr(self, component, which="model"):
        if not self._pulled:
            self.pull_source()

        cpath = get_component_file(os.path.join(self.local_path, component), which)
        if not os.path.exists(cpath):
            raise ValueError("{0}: {1} doesn't exist in {2}".
                             format(which, component, self.remote_url))

        return load_component_descr(cpath, which)

    def get_config(self):
        return OrderedDict([("type", self.TYPE),
                            ("remote_url", self.remote_url),
                            ("local_path", self.local_path)])


class GitSource(Source):
    TYPE = "git"

    def __init__(self, remote_url, local_path):
        """Git Source
        """
        self.remote_url = remote_url
        self.local_path = os.path.join(local_path, '')  # add trailing slash
        self._pulled = False

    def _list_components(self, which="model"):
        if not self._pulled:
            self.pull_source()
        return list_yamls_recursively(self.local_path, which)

    def clone(self):
        """Clone the self.remote_url into self.local_path
        """
        if os.path.exists(self.local_path) and os.listdir(self.local_path):
            raise IOError("Directory {0} already exists and is non-empty".
                          format(self.local_path))

        logger.info("Cloning {remote} into {local}".
                    format(remote=self.remote_url,
                           local=self.local_path))

        subprocess.call(["git",
                         "clone",
                         self.remote_url,
                         self.local_path])
        self._pulled = True

    def pull_source(self):
        """Pull/update the source
        """
        if not os.path.exists(self.local_path) or not os.listdir(self.local_path):
            return self.clone()

        logger.info("Update {0}".
                    format(self.local_path))
        subprocess.call(["git",
                         "pull"],
                        cwd=self.local_path)
        self._pulled = True

    def _pull_component(self, component, which="model"):
        if not self._pulled:
            self.pull_source()

        cpath = get_component_file(os.path.join(self.local_path, component), which)
        if not os.path.exists(cpath):
            raise ValueError("{0} {1} doesn't exist in {2}".
                             format(which, component, self.remote_url))
        logger.info("{0} {1} loaded".format(which, component))
        return cpath

    def _get_component_descr(self, component, which="model"):
        return load_component_descr(self._pull_component(component, which), which)

    def get_config(self):
        return OrderedDict([("type", self.TYPE),
                            ("remote_url", self.remote_url),
                            ("local_path", self.local_path)])


class LocalSource(Source):

    TYPE = "local"

    def __init__(self, local_path):
        """Local files
        """
        self.local_path = os.path.join(local_path, '')  # add trailing slash

    def _list_components(self, which="model"):
        return list_yamls_recursively(self.local_path, which)

    def _pull_component(self, component, which="model"):
        cpath = get_component_file(os.path.join(self.local_path, component), which)
        if not os.path.exists(cpath):
            raise ValueError("{0} {1} doesn't exist".
                             format(which, component))
        return cpath

    def _get_component_descr(self, component, which="model"):
        return load_component_descr(self._pull_component(component, which), which)

    def get_config(self):
        return OrderedDict([("type", self.TYPE),
                            ("local_path", self.local_path)])


# --------------------------------------------
# all available models
source_classes = [GitLFSSource, GitSource, LocalSource]


def load_source(config):
    """Load the source from config
    """
    type_cls = {cls.TYPE: cls for cls in source_classes}
    if config["type"] not in type_cls:
        raise ValueError("config['type'] needs to be one of: {0}".
                         format(list(type_cls.keys())))

    cls = type_cls[config["type"]]
    return cls.from_config(config)
