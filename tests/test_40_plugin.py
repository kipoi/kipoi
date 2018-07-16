"""Test the plugins
"""
import os
import subprocess
from kipoi.plugin import list_installed, list_plugins, get_model_yaml_parser, get_dataloader_yaml_parser, get_cli_fn, get_plugin_help, load_plugin, get_plugin_cli_fns


def test_list_installed():
    assert list_installed() == ['kipoi_veff', 'kipoi_interpret']


def test_list_plugins():
    df = list_plugins()
    assert df.plugin.tolist() == ['kipoi_veff', 'kipoi_interpret']
    assert df.installed.all()
    assert df.cli.all()


def test_get_plugin_help():
    h = get_plugin_help()
    assert "Plug" in h
    assert "kipoi_veff" in h
    assert "kipoi_interpret" in h


def test_yaml_parsers():
    plugin = load_plugin("kipoi_veff")
    assert plugin.ModelParser == get_model_yaml_parser("kipoi_veff")
    assert plugin.DataloaderParser == get_dataloader_yaml_parser("kipoi_veff")
    assert get_model_yaml_parser("kipoi_interpret") is None
    assert get_dataloader_yaml_parser("kipoi_interpret") is None


def test_cli_fn():
    fns = get_plugin_cli_fns()
    assert set(fns) == {"kipoi_veff", "kipoi_interpret"}


def test_cli_list_plugins():
    args = ["python",
            os.path.abspath("./kipoi/__main__.py"),
            "list_plugins"]
    returncode = subprocess.call(args=args)
    assert returncode == 0
