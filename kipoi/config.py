"""Configuration management for Kipoi

Following the Keras configuration management:
https://github.com/fchollet/keras/blob/6f3e6bb6fc97e706f37dc078ae821f490b78035b/keras/backend/__init__.py#L14-L43
"""
from __future__ import absolute_import
from __future__ import print_function

import os
from collections import OrderedDict
import pandas as pd
import six
from .remote import load_source, GitLFSModelSource
from .utils import yaml_ordered_dump, yaml_ordered_load
# --------------------------------------------


_kipoi_base_dir = os.path.expanduser('~')
if not os.access(_kipoi_base_dir, os.W_OK):
    _kipoi_base_dir = '/tmp'

_kipoi_dir = os.path.join(_kipoi_base_dir, '.kipoi')

# default model_sources
_MODEL_SOURCES = {
    "kipoi": GitLFSModelSource(remote_url="git@github.com:kipoi/models.git",
                               local_path=os.path.join(_kipoi_dir, "models/"))
}

# TODO - use related to model the config file as a class?


def model_sources():
    return _MODEL_SOURCES


def model_sources_dict():
    return OrderedDict([(k, v.get_config())
                        for k, v in six.iteritems(model_sources())])


def set_model_sources(_model_sources):
    global _MODEL_SOURCES

    _MODEL_SOURCES = _model_sources


def get_source(source):
    if source not in model_sources():
        raise ValueError("source={0} needs to be in model_sources()" +
                         "available sources: {1}".
                         format(source, list(model_sources().keys())))
    return model_sources()[source]


def add_source(name, obj):
    """Append a custom source to global model_sources

    # Arguments
      name: source name
      obj: source object. Can be a dictionary or a ModelSource instance (say `kipoi.remote.LocalModelSource("mydir/")`).

    """
    if isinstance(obj, dict):
        # parse the object
        obj = load_source(obj)
    c_dict = model_sources()
    c_dict.update({name: obj})
    set_model_sources(c_dict)


def list_models():
    """List models as a `pandas.DataFrame`
    """
    def get_df(source_name, source):
        df = source.list_models_df()
        df.insert(0, "source", source_name)
        return df

    return pd.concat([get_df(name, source) for name, source in six.iteritems(model_sources())])


# Attempt to read Kipoi config file.
_config_path = os.path.expanduser(os.path.join(_kipoi_dir, 'config.yaml'))
if os.path.exists(_config_path):
    try:
        _config = yaml_ordered_load(open(_config_path))
    except ValueError:
        _config = {}
    _model_sources = _config.get('model_sources', None)
    if _model_sources is None:
        _model_sources = model_sources()
    else:
        # dict  -> ModelSource class
        if "dir" in _model_sources:
            raise ValueError("'dir' is a protected key name in model_sources" +
                             " and hence can't be used")

        _model_sources = OrderedDict([(k, load_source(v))
                                      for k, v in six.iteritems(_model_sources)])
    assert isinstance(_model_sources, OrderedDict)
    set_model_sources(_model_sources)


# Save config file, if possible.
if not os.path.exists(_kipoi_dir):
    try:
        os.makedirs(_kipoi_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

# Writing the file
if not os.path.exists(_config_path):
    _config = {
        'model_sources': model_sources_dict(),
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(yaml_ordered_dump(_config, indent=4, default_flow_style=False))
    except IOError:
        # Except permission denied.
        pass
