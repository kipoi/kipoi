from __future__ import absolute_import
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty
import sys
import os
import six
import subprocess
import logging
import glob
from collections import OrderedDict
from .utils import lfs_installed, get_file_path, cd
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
    if sys.version_info >= (3, 5):
        return [os.path.dirname(filename)[len(root_dir):] for filename in
                glob.iglob(root_dir + '**/{0}.y?ml'.format(basename), recursive=True)]
    else:
        import fnmatch
        return [os.path.dirname(os.path.join(root, filename))[len(root_dir):]
                for root, dirnames, filenames in os.walk(root_dir)
                for filename in fnmatch.filter(filenames, '{0}.y?ml'.format(basename))]


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
                ("authors", str(d.info.authors)),
                ("doc", d.info.doc),
                ("type", d.type),
                ("inputs", to_namelist(d.schema.inputs)),
                ("targets", to_namelist(d.schema.targets)),
                ("tags", d.info.tags),
            ])

        return pd.DataFrame([dict2df_dict(self.get_model_descr(model), model)
                             for model in self._list_components("model")])

    def list_dataloaders(self):
        """List all the models as a data.frame
        """
        def dict2df_dict(d, dataloader):
            return OrderedDict([
                ("dataloader", dataloader),
                # ("name", d.info.name),
                ("version", d.info.version),
                ("authors", str(d.info.authors)),
                ("doc", d.info.doc),
                ("type", d.type),
                ("inputs", to_namelist(d.output_schema.inputs)),
                ("targets", to_namelist(d.output_schema.targets)),
                ("tags", d.info.tags),
            ])

        return pd.DataFrame([dict2df_dict(self.get_dataloader_descr(dataloader), dataloader)
                             for dataloader in self._list_components("dataloader")])

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
        lfs_installed(raise_exception=True)
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

        subprocess.call(["git-lfs",
                         "clone",
                         "-I /",
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
        subprocess.call(["git-lfs",
                         "pull",
                         "-I /"],
                        cwd=self.local_path)
        self._pulled = True

    def _pull_component(self, component, which="model"):
        if not self._pulled:
            self.pull_source()

        cpath = get_component_file(os.path.join(self.local_path, component), which)
        if not os.path.exists(cpath):
            raise ValueError("{0}: {1} doesn't exist in {2}".
                             format(component, self.remote_url))

        cmd = ["git-lfs",
               "pull",
               "-I {component}/**".format(component=component)]
        logger.info(" ".join(cmd))
        subprocess.call(cmd,
                        cwd=self.local_path)
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
