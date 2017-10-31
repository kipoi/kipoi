"""Test the Kipoi configuration setup
"""
import kipoi
import os
import config


def test_config_file_exists():
    assert os.path.exists(os.path.join(os.path.expanduser('~'), ".kipoi"))
    assert os.path.exists(os.path.join(os.path.expanduser('~'), ".kipoi/config.yaml"))


def test_load_config():
    assert kipoi.config.model_sources()["kipoi"].local_path == \
        os.path.join(os.path.expanduser('~'), ".kipoi/models/")
