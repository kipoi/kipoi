"""Configuration management for Kipoi

Following the Keras configuration management:
https://github.com/fchollet/keras/blob/6f3e6bb6fc97e706f37dc078ae821f490b78035b/keras/backend/__init__.py#L14-L43
"""
import os
import yaml

_kipoi_base_dir = os.path.expanduser('~')
if not os.access(_kipoi_base_dir, os.W_OK):
    _kipoi_base_dir = '/tmp'

_kipoi_dir = os.path.join(_kipoi_base_dir, '.kipoi')

# default model directory
_KIPOI_MODELS_REPO = os.path.join(_kipoi_dir, "models/")

# other repositories
_OTHER_MODELS_REPO = []


def kipoi_models_repo():
    return _KIPOI_MODELS_REPO


def set_kipoi_models_repo(kmr):
    global _KIPOI_MODELS_REPO

    _KIPOI_MODELS_REPO = kmr


def other_models_repo():
    return _OTHER_MODELS_REPO


def set_other_models_repo(repo_list):
    global _OTHER_MODELS_REPO

    _OTHER_MODELS_REPO = repo_list


# Attempt to read Kipoi config file.
_config_path = os.path.expanduser(os.path.join(_kipoi_dir, 'config.yaml'))
if os.path.exists(_config_path):
    try:
        _config = yaml.load(open(_config_path))
    except ValueError:
        _config = {}
    _kipoi_models_repo = _config.get('kipoi_models_repo', kipoi_models_repo())
    assert isinstance(_kipoi_models_repo, str)
    _other_models_repo = _config.get('other_models_repo', other_models_repo())
    assert isinstance(_other_models_repo, list)
    if len(_other_models_repo) > 0:
        assert isinstance(_other_models_repo[0], str)

    set_kipoi_models_repo(_kipoi_models_repo)
    set_other_models_repo(_other_models_repo)


# Save config file, if possible.
if not os.path.exists(_kipoi_dir):
    try:
        os.makedirs(_kipoi_dir)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        'kipoi_models_repo': kipoi_models_repo(),
        'other_models_repo': other_models_repo(),
    }
    try:
        with open(_config_path, 'w') as f:
            f.write(yaml.dump(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass
