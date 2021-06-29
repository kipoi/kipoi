from __future__ import absolute_import
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty
import sys
import os
import six
import subprocess
import logging
from collections import OrderedDict
from kipoi_utils.utils import unique_list, lfs_installed, get_file_path, cd, list_files_recursively, is_subdir, relative_path
import pandas as pd
import kipoi
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --------------------------------------------

# main functions

def list_subcomponents(component, source, which="model"):
    """List all the available submodels

    Args:
      model: model name or a subname: e.g. instaead of
        Model1/CTCF we can give Model1 and then all the sub-models would be included
      source: model source
    """
    src = kipoi.get_source(source)
    if src._is_component(component, which):
        return [component]
    else:
        return [x for x in src._list_components(which)
                if x.startswith(component) and "/template" not in x]


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


# --------------------------------------------

# helper

def get_component_file(component_dir, which="model", raise_err=True):
    # TODO - if component_dir has an extension, then just return that file path
    return get_file_path(component_dir, which, extensions=[".yml", ".yaml"], raise_err=raise_err)


def list_yamls_recursively(root_dir, basename):
    return [os.path.dirname(x) for x in list_files_recursively(root_dir, basename, suffix='y?ml')]


def list_softlink_realpaths(root_dir):
    for root, dirnames, filenames in os.walk(root_dir):
        for name in filenames:
            fname = os.path.join(root, name)
            if os.path.islink(fname):
                yield os.path.realpath(fname)


def load_component_descr(component_dir, which="model"):
    """Return the parsed yaml file
    """
    from kipoi.specs import ModelDescription, DataLoaderDescription

    fname = get_component_file(os.path.abspath(component_dir), which, raise_err=True)

    with cd(os.path.dirname(fname)):
        if which == "model":
            return ModelDescription.load(fname)
        elif which == "dataloader":
            return DataLoaderDescription.load(fname)
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


def to_namelist(nested_dict):
    """no-recursion
    """
    if isinstance(nested_dict, list):
        return [x.name for x in nested_dict]
    elif isinstance(nested_dict, dict):
        return list(nested_dict)
    else:
        return nested_dict.name


def list_models_by_group(df, group_filter=""):
    """Get a list of models by a group

    Args:
      df: Dataframe returned by list_models()
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
    # df = self.list_models()
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
            ("authors", unique_list([author for authors in x.authors
                                     for author in authors])),
            ("contributors", unique_list([contributor for contributors in x.contributors
                                          for contributor in contributors])),
            ("veff_score_variants", x.veff_score_variants.any()),
            ("type", unique_list([t for t in x.type])),
            ("license", unique_list([l for l in x.license])),
            ("cite_as", unique_list([c for c in x.cite_as if c is not None])),
            ("tags", unique_list([tag for tags in x.tags
                                  for tag in tags])),
        ]))

    return df.groupby("group").apply(fn).reset_index()


class LocalComponentGroup(object):
    """Abstraction for a folder containing the following files:

    - model-template.yaml
    - models.tsv
    - (optional) dataloader.yaml
    - ...
    """

    def __init__(self, component_template_yaml, models_tsv, which='model'):
        from jinja2 import Template
        # read the yaml file as a string
        self.which = which
        assert self.which in ['model', 'dataloader']

        self.component_template_yaml = component_template_yaml
        with open(self.component_template_yaml, "r") as f:
            template_str = f.read()
            if sys.version_info[0] == 2:
                template_str = template_str.decode("utf-8")
            self.template = Template(template_str)

        self.models_tsv = models_tsv
        self.df = pd.read_csv(models_tsv, sep='\t', comment='#')

        if 'model' not in self.df:
            raise ValueError("Column 'model' has to exist in {}. Make "
                             "also sure the tsv file is correctly formatted".format(models_tsv))

        # assert each model occurs once
        assert len(self.df.model.duplicated()) == len(self.df.model)

    def get_model_row(self, model):
        if model not in list(self.df.model):
            raise ValueError("model {} not found in {}".format(model, list(self.df.model)))
        return self.df[self.df.model == model].iloc[0].to_dict()

    def _is_component(self, component):
        return component in self._list_components()

    def _list_components(self):
        return list(self.df.model)

    def _get_component_descr(self, component):
        from kipoi.specs import ModelDescription, DataLoaderDescription

        # render the template
        rendered_yaml = self.template.render(**self.get_model_row(component))

        if self.which == 'model':
            return ModelDescription.from_string(rendered_yaml)
        elif self.which == 'dataloader':
            return DataLoaderDescription.from_string(rendered_yaml)
        else:
            raise ValueError("Unknown component {}".format(self.which))

    # --------------------------------------------
    # class methods

    @classmethod
    def is_group(cls, path, which='model'):
        models_tsv = os.path.join(path, 'models.tsv')
        yaml_template = get_component_file(path, which + '-template', raise_err=False)
        if yaml_template is None:
            return False
        return os.path.exists(models_tsv) and os.path.exists(yaml_template)

    @classmethod
    def group_path(cls, component_path, which='model'):
        """Get the path of the group given the component

        if None is returned, then the component was not found
        """
        tmp_path = component_path
        while tmp_path not in ['', '/']:
            if cls.is_group(tmp_path, which):
                return tmp_path
            else:
                tmp_path = os.path.dirname(tmp_path)
        return None

    @classmethod
    def load(cls, path, which='model'):
        """Load the component group from the directory
        """
        models_tsv = os.path.join(path, 'models.tsv')
        yaml_template = get_component_file(path, which + '-template')

        if not os.path.exists(models_tsv):
            raise ValueError("models.tsv doesn't exist in model group {}".format(path))
        if not os.path.exists(yaml_template):
            raise ValueError("{}-template.yaml doesn't exist in model group {}".format(which, path))
        return cls(yaml_template, models_tsv, which)

# --------------------------------------------

# Source abstract class


class Source(object):

    __metaclass__ = ABCMeta

    # --------------------------------------------
    # implemented by childs
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
        """Pull/update the model locally
        """
        return

    @abstractmethod
    def _is_component(self, component, which='model'):
        """Returns True if the component exists
        """
        return

    @abstractmethod
    def _get_component_dir(self, component, which='model'):
        """Get component directory
        """
        return

    @abstractmethod
    def _get_component_download_dir(self, component, which='model'):
        """Get component dedicated download directory
        """
        return

    @abstractmethod
    def _get_component_descr(self, component, which="model"):
        """Given the component name, return the description
        """
        pass

    @abstractmethod
    def get_config(self):
        pass

    @abstractmethod
    def get_group_name(self, component, which='model'):
        pass
    # --------------------------------------------

    def assert_is_component(self, component, which='model'):
        if not self._is_component(component, which):
            raise ValueError("{} {} doesn't exist".format(which, component))

    def pull_model(self, model):
        self._pull_component(model, "model")
        return self.get_model_dir(model)

    def pull_dataloader(self, dataloader):
        self._pull_component(dataloader, "dataloader")
        return self.get_dataloader_dir(dataloader)

    def get_model_dir(self, model):
        return self._get_component_dir(model, 'model')

    def get_dataloader_dir(self, model):
        return self._get_component_dir(model, 'dataloader')

    def get_model_download_dir(self, model):
        return self._get_component_download_dir(model, 'model')

    def get_dataloader_download_dir(self, dataloader):
        return self._get_component_download_dir(dataloader, 'dataloader')

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
                ("veff_score_variants", "variant_effects" in d.postprocessing),
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
                ("inputs", to_namelist(d.get_output_schema().inputs)),
                ("targets", to_namelist(d.get_output_schema().targets)),
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
        return list_models_by_group(self.list_models(), group_filter)

    def get_model_descr(self, model):
        return self._get_component_descr(model, which="model")

    def get_dataloader_descr(self, dataloader):
        return self._get_component_descr(dataloader, which="dataloader")

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


# --------------------------------------------

# individual source implementations


class LocalSource(Source):

    TYPE = "local"

    def __init__(self, local_path=None, name=None):
        """Local files
        """
        self.name = name
        if local_path is not None:
            self._local_path = os.path.join(os.path.realpath(local_path), '')  # add trailing slash

            # load the config file
            config_file = os.path.join(self._local_path, 'config.yaml')
            if os.path.exists(config_file):
                from kipoi.specs import SourceConfig
                self.config = SourceConfig.load(config_file)

                if not self.config.dependencies.all_installed(verbose=False):
                    import colorlog
                    print(colorlog.escape_codes['red'])
                    print("WARNING: Dependencies for model source '{}' stored at local_path {} not satisfied.: \n---".
                          format(self.name, self._local_path))
                    self.config.dependencies.all_installed(verbose=True)
                    print("---\ninstall or update the missing packages")
                    print(colorlog.escape_codes['reset'])
                    print("Note: If you don't want to auto_update the model source, \n"
                          "add `auto_update: False` to ~/.kipoi/config.yaml\n")
            else:
                self.config = None
        else:
            # undetermined local path
            self._local_path = None
            self.config = None
        self.component_yaml_list = None
        self.component_group_list = None

    @property
    def local_path(self):
        if self._local_path is None:
            return os.getcwd()
        else:
            return self._local_path

    def _list_component_yamls(self, which='model'):
        return list_yamls_recursively(self.local_path, which)  # , skip='downloaded|.*_files')

    def _list_component_groups(self, which='model'):
        return {tdir: LocalComponentGroup.load(os.path.join(self.local_path, tdir), which)
                for tdir in list_yamls_recursively(self.local_path, which + '-template')}  # , skip='downloaded|.*_files')}

    def cache_component_list(self, force=False):
        if force or self.component_yaml_list is None or self.component_group_list is None:
            self.component_yaml_list = dict(model=self._list_component_yamls(which="model"),
                                            dataloader=self._list_component_yamls(which="dataloader"))
            self.component_group_list = dict(model=self._list_component_groups(which="model"),
                                             dataloader=self._list_component_groups(which="dataloader"))

    def _list_components(self, which="model"):
        self.cache_component_list(force=self._local_path is None)
        return self.component_yaml_list[which] + [os.path.join(k, c)
                                                  for k, grp in six.iteritems(self.component_group_list[which])
                                                  for c in grp._list_components()]

    def get_group_name(self, component, which='model'):
        component = os.path.normpath(component)

        if self.component_group_list is not None:
            # already cached
            for k in self.component_group_list[which]:
                if component.startswith(os.path.join(k, "")):
                    return k
            return None
        else:
            group_path = LocalComponentGroup.group_path(os.path.join(self.local_path, component), which)
            if group_path is None:
                return None
            else:
                return relative_path(group_path, self.local_path)

    def _is_nongroup_component(self, component, which):
        path = os.path.join(self.local_path, os.path.normpath(component))
        if get_component_file(path, which=which, raise_err=False) is not None:
            return True
        else:
            return False

    def _get_component_dir(self, component, which='model'):
        component = os.path.normpath(component)

        self.assert_is_component(component, which)
        # special case: component can be outside of the root directory
        if self._is_nongroup_component(component, which):
            return os.path.join(self.local_path, os.path.normpath(component))
        else:
            k = self.get_group_name(component, which)
            assert k is not None
            return os.path.join(self.local_path, k)

    def _get_component_download_dir(self, component, which='model', name=None):
        component = os.path.normpath(component)

        if name is None:
            name = which
        insert_path = os.path.join("downloaded", '{}_files'.format(name))

        # special case: component can be outside of the root directory
        if self._is_nongroup_component(component, which):
            return os.path.join(self.local_path, os.path.normpath(component), insert_path)
        else:
            k = self.get_group_name(component, which)
            if k is None and which == 'dataloader':
                # fallback: get model's download directory
                return self._get_component_download_dir(component, which='model', name='dataloader')
            if k is None:
                raise ValueError("Couldn't get {} download_dir. Model"
                                 " or group doesn't exist for {}".format(which, component))
            return os.path.join(self.local_path, k, insert_path, relative_path(component, k))

    def _is_component(self, component, which="model"):
        component = os.path.normpath(component)
        if self._is_nongroup_component(component, which):
            # it contains a {which}.y?ml
            return True
        else:
            # it's present in one of the groups

            k = self.get_group_name(component, which)
            if k is not None:
                # check that it's indeed found in one of the components
                if self.component_group_list is not None:
                    # already cached
                    return component in self._list_components(which)
                else:
                    grp = LocalComponentGroup.load(os.path.join(self.local_path, k), which)
                    return grp._is_component(relative_path(component, k))
            else:
                return False

    def _get_component_descr(self, component, which="model"):
        component = os.path.normpath(component)
        self.assert_is_component(component, which)

        if self._is_nongroup_component(component, which):
            # component has an explicit yaml file

            # TODO - move into the component directory when loading
            return load_component_descr(os.path.join(self.local_path, component), which)
        else:
            k = self.get_group_name(component, which)
            if k is None:
                raise ValueError("{} {} doesn't exist".format(which, component))
            else:
                if self.component_group_list is not None:
                    # already cached
                    return self.component_group_list[which][k]._get_component_descr(relative_path(component, k))
                else:
                    grp = LocalComponentGroup.load(os.path.join(self.local_path, k), which)
                    return grp._get_component_descr(relative_path(component, k))

    def _pull_component(self, component, which="model"):
        self.assert_is_component(component, which)

    def get_config(self):
        return OrderedDict([("type", self.TYPE),
                            ("local_path", self.local_path)])


class GitSource(Source):

    TYPE = "git"

    def __init__(self, remote_url, local_path, auto_update=True, use_lfs=False, name=None):
        """Git Source
        """
        self.name = name
        self.remote_url = remote_url
        self.local_path = os.path.join(os.path.realpath(local_path), '')  # add trailing slash
        self.local_source = LocalSource(self.local_path, name=name)

        self.auto_update = auto_update
        self._pulled = False
        self.use_lfs = use_lfs
        if self.use_lfs:
            lfs_installed(raise_exception=False)

    def clone(self, depth=1):
        """Clone the self.remote_url into self.local_path

        Args:
          depth: --depth argument to git clone. If None, clone the whole history.
        """
        if os.path.exists(self.local_path) and os.listdir(self.local_path):
            raise IOError("Directory {0} already exists and is non-empty".
                          format(self.local_path))

        logger.info("Cloning {remote} into {local}".
                    format(remote=self.remote_url,
                           local=self.local_path))
        cmd = ["git", "clone"]
        if depth is not None:
            cmd.append("--depth={0}".format(depth))
        cmd.append(self.remote_url)
        cmd.append(self.local_path)
        subprocess.call(cmd,
                        env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))
        self._pulled = True

        if os.path.exists(os.path.join(self.local_path, ".gitattributes")):
            if not self.use_lfs:
                logger.info(".gitattributes detected in {}. Using git-lfs".format(self.local_path))
            self.use_lfs = True
            lfs_installed(raise_exception=True)

    def pull_source(self):
        """Pull/update the source
        """
        if not os.path.exists(self.local_path) or not os.listdir(self.local_path):
            return self.clone()

        if not self.auto_update:
            logger.warning("Pulling source even though auto_update=False")

        logger.info("Update {0}".
                    format(self.local_path))
        subprocess.call(["git",
                         "pull"],
                        cwd=self.local_path,
                        env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))

        if os.path.exists(os.path.join(self.local_path, ".gitattributes")):
            if not self.use_lfs:
                logger.info(".gitattributes detected in {}. Using git-lfs".format(self.local_path))
            self.use_lfs = True
            lfs_installed(raise_exception=True)

        self._pulled = True

    def _commit_checkout(self, commit):
        """Checkout a particular commit
        """
        logger.info("Update {0}".
                    format(self.local_path))
        subprocess.call(["git",
                         "reset",
                         "--hard",
                         commit],
                        cwd=self.local_path,
                        env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))

    def _list_components(self, which="model"):
        if not self._pulled and self.auto_update:
            self.pull_source()
        return self.local_source._list_components(which)

    def _get_component_dir(self, component, which='model'):
        if not self._pulled and self.auto_update:
            self.pull_source()
        return self.local_source._get_component_dir(component, which)

    def _get_component_download_dir(self, component, which='model'):
        if not self._pulled and self.auto_update:
            self.pull_source()
        return self.local_source._get_component_download_dir(component, which)

    def _pull_component(self, component, which="model"):
        if not self._pulled and self.auto_update:
            self.pull_source()

        component_dir = self.local_source._get_component_dir(component, which)

        if self.use_lfs:
            # the only call to git-lfs -> pulling specific sub-files

            lfs_installed(raise_exception=True)
            # get a list of directories to source (relative to the local_path)
            softlink_dirs = list(list_softlink_dependencies(component_dir, self.local_path))
            # pull these softlinks
            for pull_dir in [component, relative_path(component_dir, self.local_path)] + softlink_dirs:
                cmd = ["git-lfs",
                       "pull",
                       "-I {0}/**".format(pull_dir)]
                logger.info(" ".join(cmd))
                subprocess.call(cmd,
                                cwd=self.local_path,
                                env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"))

        return self.local_source._pull_component(component, which)

    def _is_component(self, component, which="model"):
        if not self._pulled and self.auto_update:
            self.pull_source()
        return self.local_source._is_component(component, which)

    def get_group_name(self, component, which='model'):
        return self.local_source.get_group_name(component, which)

    def _get_component_descr(self, component, which="model"):
        if not self._pulled and self.auto_update:
            self.pull_source()
        return self.local_source._get_component_descr(component, which)

    def get_config(self):
        return OrderedDict([("type", self.TYPE),
                            ("remote_url", self.remote_url),
                            ("local_path", self.local_path),
                            ("auto_update", self.auto_update),
                            ("use_lfs", self.use_lfs)])


class GitLFSSource(GitSource):
    TYPE = 'git-lfs'

    def __init__(self, remote_url, local_path, auto_update=True, name=None):
        """Git-LFS Source
        """
        super(GitLFSSource, self).__init__(remote_url=remote_url,
                                           local_path=local_path,
                                           auto_update=auto_update,
                                           use_lfs=True,
                                           name=name)


class GithubPermalinkSource(Source):

    TYPE = "github-permalink"

    def __init__(self, local_path, name=None):
        """Local files
        """
        self.name = name
        self.local_path = os.path.join(os.path.realpath(local_path), '')  # add trailing slash

    @classmethod
    def _parse_url(cls, url):
        """Map github url to local directory
        """
        github_url = "https://github.com/"
        if not url.startswith(github_url):
            raise ValueError("url of the permalink: {0} doesn't start with {1}".format(url, github_url))
        url_dir = url[len(github_url):]
        if "/tree/" not in url_dir:
            raise ValueError("'/tree/' missing in the url {0}. Typical github format " +
                             "is github.com/<user>/<repo>/tree/<commit>/<directory>")
        url_dir = url_dir.replace("/tree", "")

        user, repo, commit, model = url_dir.split("/", 3)
        model = model.rstrip("/")
        return user, repo, commit, model

    def _list_components(self, which="model"):
        # Same as for local source
        return []  # list_yamls_recursively(self.local_path, which)

    def get_lfs_source(self, component):
        user, repo, commit, model = self._parse_url(component)
        remote_url = "https://github.com/{user}/{repo}.git".format(user=user, repo=repo)
        lfs_source = GitSource(remote_url, os.path.join(self.local_path, user, repo, commit),
                               auto_update=False,  # Don't git-pull
                               use_lfs=True)
        self._pulled = True  # actually not required due to auto_update=False

        if not os.path.exists(lfs_source.local_path) or not os.listdir(lfs_source.local_path):
            # clone the repository
            lfs_source.clone(depth=None)
            lfs_source._commit_checkout(commit)
        return lfs_source

    def _get_component_dir(self, component, which='model'):
        user, repo, commit, model = self._parse_url(component)
        return self.get_lfs_source(component)._get_component_dir(model, which='model')

    def _get_component_download_dir(self, component, which='model'):
        user, repo, commit, model = self._parse_url(component)
        return self.get_lfs_source(component)._get_component_download_dir(model, which='model')

    def _pull_component(self, component, which="model"):
        user, repo, commit, model = self._parse_url(component)
        self.get_lfs_source(component)._pull_component(model, which=which)

    def _is_component(self, component, which="model"):
        user, repo, commit, model = self._parse_url(component)
        return self.get_lfs_source(component)._is_component(model, which)

    def get_group_name(self, component, which='model'):
        user, repo, commit, model = self._parse_url(component)
        return self.get_lfs_source(component).get_group_name(component, which)

    def _get_component_descr(self, component, which="model"):
        user, repo, commit, model = self._parse_url(component)
        return self.get_lfs_source(component)._get_component_descr(model, which)

    def get_config(self):
        return OrderedDict([("type", self.TYPE),
                            ("local_path", self.local_path)])


# --------------------------------------------
# all available models
source_classes = [GitLFSSource, GitSource, LocalSource, GithubPermalinkSource]


def load_source(config, name):
    """Load the source from config
    """
    type_cls = {cls.TYPE: cls for cls in source_classes}
    if config["type"] not in type_cls:
        raise ValueError("config['type'] needs to be one of: {0}".
                         format(list(type_cls.keys())))

    if config['type'] == 'git-lfs':
        # local path is already checked out and .gitattributes doesn't exist
        # -> the repo is not an actuall git-lfs
        if os.path.exists(os.path.join(config['local_path'], '.git')) \
           and not os.path.exists(os.path.join(config['local_path'], '.gitattributes')):
            # if the .gitattributes doesn't exist for the local path

            if not config['remote_url'].endswith("kipoi/models.git"):
                # only display this message for remote url's other than kipoi/models
                logger.info(".gitattributes not found for git-lfs source: {} stored at {}".
                            format(config['remote_url'], config['local_path']))
                logger.info("Using 'type: git'")
            config['type'] = 'git'

    # add name
    config['name'] = name
    cls = type_cls[config["type"]]
    return cls.from_config(config)
