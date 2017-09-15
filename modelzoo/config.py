"""Configuration management for Kipoi

Following the Keras configuration management:
https://github.com/fchollet/keras/blob/6f3e6bb6fc97e706f37dc078ae821f490b78035b/keras/backend/__init__.py#L14-L43
"""
import os
import yaml
from collections import OrderedDict
import six
from .remote import load_source, GitLFSModelSource

_mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

# --------------------------------------------
# allow yaml to use orderedDict


def dict_representer(dumper, data):
    return dumper.represent_dict(six.iteritems(data))


def dict_constructor(loader, node):
    return OrderedDict(loader.construct_pairs(node))


yaml.add_representer(OrderedDict, dict_representer)
yaml.add_constructor(_mapping_tag, dict_constructor)
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


def model_sources():
    return _MODEL_SOURCES


def model_sources_dict():
    return OrderedDict([(k, v.get_config())
                        for k, v in six.iteritems(model_sources())])


def set_model_sources(_model_sources):
    global _MODEL_SOURCES

    _MODEL_SOURCES = _model_sources


# Attempt to read Kipoi config file.
_config_path = os.path.expanduser(os.path.join(_kipoi_dir, 'config.yaml'))
if os.path.exists(_config_path):
    try:
        _config = yaml.load(open(_config_path))
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
            f.write(yaml.dump(_config, indent=4, default_flow_style=False))
    except IOError:
        # Except permission denied.
        pass
