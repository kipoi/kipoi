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
from .sources import load_source, GitSource, GitLFSSource, GithubPermalinkSource, LocalSource
from kipoi_utils.utils import yaml_ordered_dump, yaml_ordered_load, du
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# --------------------------------------------


_kipoi_base_dir = os.path.expanduser('~')
if not os.access(_kipoi_base_dir, os.W_OK):
    _kipoi_base_dir = '/tmp'

_kipoi_dir = os.path.join(_kipoi_base_dir, '.kipoi')

# model source container
_MODEL_SOURCES = {}


def model_sources():
    if _MODEL_SOURCES:
        return _MODEL_SOURCES
    else:
        # default model source
        return {
            "kipoi": GitSource(remote_url="https://github.com/kipoi/models.git",
                               local_path=os.path.join(_kipoi_dir, "models/"),
                               auto_update=True),
            "github-permalink": GithubPermalinkSource(local_path=os.path.join(_kipoi_dir, "github-permalink/")),
        }


def model_sources_dict():
    return OrderedDict([(k, v.get_config())
                        for k, v in six.iteritems(model_sources())])


def set_model_sources(_model_sources):
    global _MODEL_SOURCES

    _MODEL_SOURCES = _model_sources


def get_source(source):
    """Get the source object

    # Arguments
        source (str): source string

    # Returns
        Source child class instance: like `kipoi.sources.GitLFSSource`
    """
    if source in model_sources():
        return model_sources()[source]
    else:
        raise ValueError("source={0} needs to be in model_sources()\n".format(source) +
                         "available sources: {0}".
                         format(", ".join(list(model_sources().keys()))))


def add_source(name, obj):
    """Append a custom source to global model_sources

    # Arguments
      name: source name
      obj: source object. Can be a dictionary or a Source instance (say `kipoi.sources.LocalSource("mydir/")`).

    """
    if isinstance(obj, dict):
        # parse the object
        obj = load_source(obj, name)
    c_dict = model_sources()
    c_dict.update({name: obj})
    set_model_sources(c_dict)


def list_sources():
    """Returns a `pandas.DataFrame` of possible sources
    """

    def src2dict(k, s):
        lm = s.list_models()
        return OrderedDict([("source", k),
                            ("type", s.TYPE),
                            ("location", s.local_path),
                            ("local_size", du(s.local_path)),
                            ("n_models", len(lm)),
                            ("n_dataloaders", len(lm)),  # TODO - update
                            # last_updated=TODO - implement?
                            ])

    return pd.DataFrame([src2dict(k, s) for k, s in six.iteritems(model_sources()) if k != "dir"])


def list_models(sources=None):
    """List models

    # Arguments
        sources: list of model sources to use. If None, use all

    # Returns
        pandas.DataFrame
    """

    def get_df(source_name, source):
        df = source.list_models()
        df.insert(0, "source", source_name)
        return df

    if sources is None:
        sources = model_sources()

    pd_list = []
    for name, source in six.iteritems(sources):
        if name != "dir":
            pd_list.append(get_df(name, source))

    return pd.concat(pd_list)[pd_list[0].columns]


def list_dataloaders(sources=None):
    """List dataloaders

    # Arguments
        sources: list of model sources to use. If None, use all

    # Returns
        pandas.DataFrame
    """

    def get_df(source_name, source):
        df = source.list_dataloaders()
        df.insert(0, "source", source_name)
        return df

    if sources is None:
        sources = model_sources()

    pd_list = []
    for name, source in six.iteritems(sources):
        if name != "dir":
            pd_list.append(get_df(name, source))

    return pd.concat(pd_list)[pd_list[0].columns]


# Attempt to read Kipoi config file.
_config_path = os.path.expanduser(os.path.join(_kipoi_dir, 'config.yaml'))
# Set the env DB path
_env_db_path = os.path.join(_kipoi_dir, 'envs.json')
if os.path.exists(_config_path):
    try:
        _config = yaml_ordered_load(open(_config_path, "r"))
    except ValueError:
        logger.warning("Unable to parse the config file: {0}. Using default config".format(_config_path))
        _model_sources = model_sources()
    else:
        _model_sources = _config['model_sources']
        # Try to load DB path from ~/.kipoi/config.yaml
        if 'env_db_path' in _config:
            _env_db_path = os.path.expanduser(_config['env_db_path'])
    # dict  -> Source class
    if "dir" in _model_sources:
        raise ValueError("'dir' is a protected key name in model_sources" +
                         " and hence can't be used")

    _model_sources = OrderedDict([(k, load_source(v, k))
                                  for k, v in six.iteritems(_model_sources)])
    assert isinstance(_model_sources, OrderedDict)
    set_model_sources(_model_sources)

if 'KIPOI_ENV_DB_PATH' in os.environ:
    _env_db_path = os.path.expanduser(os.environ['KIPOI_ENV_DB_PATH'])

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
    logger.info("{0} doesn't exist. Creating the config file".format(_config_path))
    _config = {
        'model_sources': model_sources_dict(),
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(yaml_ordered_dump(_config, indent=4, default_flow_style=False))
    except IOError:
        # Except permission denied.
        pass

# Add dir as a valid source
add_source("dir", LocalSource(None))

if "github-permalink" not in model_sources():
    # backcompatibility with old versions
    add_source("github-permalink",
               GithubPermalinkSource(local_path=os.path.join(_kipoi_dir, "github-permalink/")))
