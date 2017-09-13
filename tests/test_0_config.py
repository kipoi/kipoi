"""Test the Kipoi configuration setup
"""
import modelzoo
import os


def test_config_file_exists():
    assert os.path.exists(os.path.join(os.path.expanduser('~'), ".kipoi"))
    assert os.path.exists(os.path.join(os.path.expanduser('~'), ".kipoi/config.yaml"))


def test_load_config():
    assert modelzoo.config.kipoi_models_repo() == os.path.join(os.path.expanduser('~'), ".kipoi/models/")
    assert modelzoo.config.other_models_repo() == []


def test_load_models():
    # TODO - setup git-lfs in travis
    # TODO - test: pull_kipoi_models
    # TODO - test: pull_kipoi_model
    pass
