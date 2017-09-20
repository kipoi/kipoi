from __future__ import absolute_import
from __future__ import print_function

from abc import ABCMeta, abstractmethod, abstractproperty
import os
import six
import subprocess
import logging
import glob
from .data import dir_load_extractor
from .model import dir_load_model
from collections import OrderedDict

_logger = logging.getLogger('model-zoo')


def cmd_exists(cmd):
    """Check if a certain command exists
    """
    return subprocess.call("type " + cmd, shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0


def lfs_installed(raise_exception=False):
    """Check if git lfs is installed localls
    """
    ce = cmd_exists("git-lfs")
    if raise_exception:
        if not ce:
            raise OSError("git-lfs not installed")
    return ce


def list_models_recursively(root_dir):
    return [os.path.dirname(filename)[len(root_dir):] for filename in
            glob.iglob(root_dir + '**/model.yaml', recursive=True)]


class ModelSource(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def TYPE(self):
        pass

    @abstractmethod
    def list_models(self):
        """List available models
        """
        pass

    @abstractmethod
    def pull_model(self, model):
        """Pull/update the model locally and
        returns a local path to it
        """
        return

    def load_model(self, model):
        m_dir = self.pull_model(model)
        return dir_load_model(m_dir)

    def load_extractor(self, model):
        m_dir = self.pull_model(model)
        return dir_load_extractor(m_dir)

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


class GitLFSModelSource(ModelSource):

    TYPE = "git-lfs"

    def __init__(self, remote_url, local_path):
        """GitLFS ModelSource
        """
        lfs_installed(raise_exception=True)
        self.remote_url = remote_url
        self.local_path = local_path
        self._pulled = False

    def list_models(self):
        if not self._pulled:
            self.pull_source()
        return list_models_recursively(self.local_path)

    def clone(self):
        """Clone the self.remote_url into self.local_path
        """
        if os.path.exists(self.local_path):
            raise IOError("Directory {0} already exists".
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
        if not os.path.exists(self.local_path):
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

    def pull_model(self, model):
        if not self._pulled:
            self.pull_source()

        mpath = os.path.join(self.local_path, model)
        if not os.path.exists(mpath):
            raise ValueError("Model: {0} doesn't exist in {1}".
                             format(model, self.remote_url))

        cmd = ["git-lfs",
               "pull",
               "-I {model}/**".format(model=model)]
        _logger.info(" ".join(cmd))
        subprocess.call(cmd,
                        cwd=self.local_path)
        _logger.info("model {0} loaded".format(model))
        return mpath

    def get_config(self):
        return OrderedDict([("type", self.TYPE),
                            ("remote_url", self.remote_url),
                            ("local_path", self.local_path)])


class GitModelSource(ModelSource):
    TYPE = "git"

    def __init__(self, remote_url, local_path):
        """Git ModelSource
        """
        self.remote_url = remote_url
        self.local_path = local_path
        self._pulled = False

    def list_models(self):
        if not self._pulled:
            self.pull_model()
        return list_models_recursively(self.local_path)

    def clone(self):
        """Clone the self.remote_url into self.local_path
        """
        if os.path.exists(self.local_path):
            raise IOError("Directory {0} already exists".
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
        if not os.path.exists(self.local_path):
            return self.clone()

        _logger.info("Update {0}".
                     format(self.local_path))
        subprocess.call(["git",
                         "pull"],
                        cwd=self.local_path)
        self._pulled = True

    def pull_model(self, model):
        if not self._pulled:
            self.pull_source()

        mpath = os.path.join(self.local_path, model)
        if not os.path.exists(mpath):
            raise ValueError("Model {0} doesn't exist in {1}".
                             format(model, self.remote_url))
        _logger.info("model {0} loaded".format(model))
        return mpath

    def get_config(self):
        return OrderedDict([("type", self.TYPE),
                            ("remote_url", self.remote_url),
                            ("local_path", self.local_path)])


class LocalModelSource(ModelSource):

    TYPE = "local"

    def __init__(self, local_path):
        """Local files
        """
        self.local_path = local_path

    def list_models(self):
        return list_models_recursively(self.local_path)

    def pull_model(self, model):
        mpath = os.path.join(self.local_path, model)
        if not os.path.exists(mpath):
            raise ValueError("Model {0} doesn't exist".
                             format(model))
        return mpath

    def get_config(self):
        return OrderedDict([("type", self.TYPE),
                            ("local_path", self.local_path)])


# all available models
source_classes = [GitLFSModelSource, GitModelSource, LocalModelSource]


def load_source(config):
    """Load the source from config
    """
    type_cls = {cls.TYPE: cls for cls in source_classes}
    if config["type"] not in type_cls:
        raise ValueError("config['type'] needs to be one of: {0}".
                         format(list(type_cls.keys())))

    cls = type_cls[config["type"]]
    return cls.from_config(config)
