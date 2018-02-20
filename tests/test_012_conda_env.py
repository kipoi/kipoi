"""Test conda env installation
"""
from collections import OrderedDict
import pytest
import kipoi
import kipoi.conda
from kipoi.components import Dependencies
from kipoi.conda import install_conda, install_pip, normalize_pip, parse_conda_package


def test_pip_merge():
    pip_list = ["package>=1.1,==1.4", "package2", "package2>=1.5",
                "package>=1.1,==1.4,==1.5", "package5"]
    assert normalize_pip(pip_list) == ['package>=1.1,==1.4,==1.5', 'package2>=1.5', 'package5']


def test_parse_conda_package():
    assert parse_conda_package("package") == ("defaults", "package")
    assert parse_conda_package("channel::package") == ("channel", "package")
    with pytest.raises(ValueError):
        parse_conda_package("channel::package::asds")


def test_Dependencies():
    dep = Dependencies(conda=["conda_pkg1", "conda_pkg2"],
                       pip=["pip_pkg1>=1.1", "pip_pkg2"])
    res = dep.to_env_dict("asd")
    assert res["name"] == "asd"
    assert res["channels"] == ["defaults"]
    assert res["dependencies"][0] == "conda_pkg1"
    assert res["dependencies"][1] == "conda_pkg2"
    assert res["dependencies"][2]["pip"][1] == "pip_pkg2"


def test_Dependencies_merge():
    dep1 = Dependencies(conda=["conda_pkg1", "conda_pkg2"],
                        pip=["pip_pkg1>=1.1", "pip_pkg2"])
    dep2 = Dependencies(conda=["conda_pkg1", "conda_pkg3>=1.1"],
                        pip=["pip_pkg1>=1.0", "pip_pkg2==3.3"])
    dep_merged = dep1.merge(dep2)
    assert dep_merged.conda == ['conda_pkg1',
                                'conda_pkg2',
                                'conda_pkg3>=1.1']
    assert dep_merged.pip == ['pip_pkg1>=1.1,>=1.0',
                              'pip_pkg2==3.3']

    assert dep_merged.conda_channels == []


def test_create_env():
    dependencies = ["python=3.6", "numpy",
                    OrderedDict(pip=["tqdm"])
                    ]

    ENV_NAME = "kipoi-test-env1"
    kipoi.conda.create_env(ENV_NAME, dependencies)
    # check that file exists
    assert kipoi.conda.env_exists(ENV_NAME)
    # remove the environment
    kipoi.conda.remove_env(ENV_NAME)
    assert not kipoi.conda.env_exists(ENV_NAME)


def test_create_env_wrong_dependencies():
    dependencies = ["python=3.6", "numpyxzy"]
    ENV_NAME = "kipoi-test-env2"
    if kipoi.conda.env_exists(ENV_NAME):
        kipoi.conda.remove_env(ENV_NAME)
    with pytest.raises(Exception):
        kipoi.conda.create_env(ENV_NAME, dependencies)


def test_install():
    # TODO - write a conda installation test with a certain channel
    # TODO - add a conda channels for installing
    conda_deps = ["python=3.6", "pep8"]
    pip_deps = ["tqdm"]

    install_conda(conda_deps)
    install_pip(pip_deps)
