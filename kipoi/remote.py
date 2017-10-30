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
from .utils import lfs_installed, get_file_path
from .components import ModelDescription, DataLoaderDescription
import pandas as pd
import kipoi

_logger = logging.getLogger('kipoi')


# TODO - replace with baked-in requirements
def get_requirements_file(model, source="kipoi"):
    """Get the requirements file path
    """
    if source == "dir":
        source = kipoi.remote.LocalSource(".")
    else:
        source = kipoi.config.get_source(source)
    return os.path.join(os.path.dirname(source.pull_model(model)), 'requirements.txt')


def get_component_file(component_dir, which="model"):
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


def load_component_info(component_path, which="model"):
    """Return the parsed yaml file
    """
    if which == "model":
        return ModelDescription.load(component_path)
    elif which == "dataloader":
        return DataLoaderDescription.load(component_path)
    else:
        raise ValueError("which needs to be from {'model', 'dataloader'}")


def get_model_info(model, source="kipoi"):
    """Get information about the model

    # Arguments
      model: model's relative path/name in the source. 2nd column in the `kipoi.list_models() `pd.DataFrame`.
      source: Model source. 1st column in the `kipoi.list_models()` `pd.DataFrame`.
    """
    return kipoi.config.get_source(source).get_model_info(model)


def get_dataloader_info(dataloader, source="kipoi"):
    """Get information about the dataloder

    # Arguments
      datalaoder: dataloader's relative path/name in the source. 2nd column in the `kipoi.list_dataloader() `pd.DataFrame`.
      source: Model source. 1st column in the `kipoi.list_models()` `pd.DataFrame`.
    """
    return kipoi.config.get_source(source).get_dataloader_info(dataloader)


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
                ("name", d.info.name),
                ("version", d.info.version),
                ("author", d.info.author),
                ("descr", d.info.descr),
                ("type", d.type),
                ("inputs", list(d.schema.inputs)),
                ("targets", list(d.schema.targets)),
                ("tags", d.info.tags),
            ])

        return pd.DataFrame([dict2df_dict(self.get_model_info(model), model)
                             for model in self._list_components("model")])

    def list_dataloaders(self):
        """List all the models as a data.frame
        """
        def dict2df_dict(d, dataloader):
            return OrderedDict([
                ("dataloader", dataloader),
                ("name", d.info.name),
                ("version", d.info.version),
                ("author", d.info.author),
                ("descr", d.info.descr),
                ("type", d.type),
                ("inputs", list(d.output_schema.inputs)),
                ("targets", list(d.output_schema.targets)),
                ("tags", d.info.tags),
            ])

        return pd.DataFrame([dict2df_dict(self.get_dataloader_info(dataloader), dataloader)
                             for dataloader in self._list_components("dataloader")])

    @abstractmethod
    def _get_component_info(self, component, which="model"):
        pass

    def get_model_info(self, model):
        return self._get_component_info(model, which="model")

    def get_dataloader_info(self, dataloader):
        return self._get_component_info(dataloader, which="dataloader")

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

        _logger.info("Cloning {remote} into {local}".
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

        _logger.info("Update {0}".
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
        _logger.info(" ".join(cmd))
        subprocess.call(cmd,
                        cwd=self.local_path)
        _logger.info("{0} {1} loaded".format(which, component))
        return cpath

    def _get_component_info(self, component, which="model"):
        if not self._pulled:
            self.pull_source()

        cpath = get_component_file(os.path.join(self.local_path, component), which)
        if not os.path.exists(cpath):
            raise ValueError("{0}: {1} doesn't exist in {2}".
                             format(which, component, self.remote_url))

        return load_component_info(cpath, which)

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

        _logger.info("Cloning {remote} into {local}".
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

        _logger.info("Update {0}".
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
        _logger.info("{0} {1} loaded".format(which, component))
        return cpath

    def _get_component_info(self, component, which="model"):
        return load_component_info(self._pull_component(component, which), which)

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

    def _get_component_info(self, component, which="model"):
        return load_component_info(self._pull_component(component, which), which)

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
