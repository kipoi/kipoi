"""Test conda env installation
"""
from collections import OrderedDict
import pytest
import kipoi
import kipoi.conda
from kipoi.specs import Dependencies
from kipoi.conda import (install_conda, install_pip, normalize_pip, parse_conda_package,
                         compatible_versions, is_installed, get_package_version, version_split)


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

    assert dep_merged.conda_channels == ["defaults"]


def test_bioconda_channels():
    dep1 = Dependencies(conda=["conda_pkg1", "bioconda::conda_pkg2"],
                        pip=[])
    channels, packages = dep1._get_channels_packages()
    assert channels == ["defaults", "bioconda", "conda-forge"]
    dep1 = Dependencies(conda=["bioconda::conda_pkg2", "conda_pkg1"],
                        pip=[])
    channels, packages = dep1._get_channels_packages()
    assert channels == ["bioconda", "conda-forge", "defaults"]

    dep1 = Dependencies(conda=["bioconda::conda_pkg2"],
                        pip=[])
    channels, packages = dep1._get_channels_packages()
    assert channels == ["bioconda", "conda-forge", "defaults"]

    dep1 = Dependencies(conda=["conda-forge::conda_pkg2", "bioconda::conda_pkg2"],
                        pip=[])
    channels, packages = dep1._get_channels_packages()
    assert channels == ["conda-forge", "bioconda", "defaults"]

    dep1 = Dependencies(conda=["asd::conda_pkg2", "bioconda::conda_pkg2", "dsa::conda_pkg2"],
                        pip=[])
    channels, packages = dep1._get_channels_packages()
    assert channels == ["asd", "bioconda", "conda-forge", "dsa", "defaults"]


def test_handle_pysam():
    dep1 = Dependencies(conda=["conda_pkg1", "bioconda::pysam"],
                        pip=[])
    channels, packages = dep1._get_channels_packages()
    assert channels == ["bioconda", "conda-forge", "defaults"]

    dep1 = Dependencies(conda=["conda_pkg1", "bioconda::pybedtools"],
                        pip=[])
    channels, packages = dep1._get_channels_packages()
    assert channels == ["defaults", "bioconda", "conda-forge"]


def test_other_channels():
    dep1 = Dependencies(conda=["other::conda_pkg2", "conda_pkg1"],
                        pip=[])
    channels, packages = dep1._get_channels_packages()
    assert channels == ["other", "defaults"]
    dep1 = Dependencies(conda=["conda_pkg1", "other::conda_pkg2"],
                        pip=[])
    channels, packages = dep1._get_channels_packages()
    assert channels == ["defaults", "other"]


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


def test_version_split():
    assert version_split("asdsda>=2.4,==2") == ('asdsda', ['>=2.4', '==2'])
    assert version_split("asdsda>=2.4") == ('asdsda', ['>=2.4'])
    assert version_split("asdsda>=2.4,~=2.3") == ('asdsda', ['>=2.4', '~=2.3'])
    assert version_split("asdsda~=2.4,>=2.3") == ('asdsda', ['~=2.4', '>=2.3'])
    assert version_split("asdsda~=2.4") == ('asdsda', ['~=2.4'])
    assert version_split("asdsda") == ('asdsda', [])


def test_compatible_versions():
    assert compatible_versions("1.10", '>=1.0')
    assert compatible_versions("1.10", '>1.0')
    assert not compatible_versions("1.10", '>=2.0')
    assert not compatible_versions("1.10", '>2.0')
    assert not compatible_versions("1.10", '==1.11')
    assert compatible_versions("1.10", '==1.10')
    assert compatible_versions("1.10", '<=1.10')
    assert not compatible_versions("1.10", '<=1.1')
    assert compatible_versions("1.10", '<=1.11')
    assert compatible_versions("1.10", '<1.11')
    with pytest.raises(ValueError):
        compatible_versions("1.10", '1<1.11')

    assert compatible_versions("0.10", '<1.11')


def test_package_version():
    import numpy as np
    import pandas as pd
    assert get_package_version("kipoi") == kipoi.__version__
    assert get_package_version("numpy") == np.__version__
    assert get_package_version("pandas") == pd.__version__
    assert get_package_version("package_doesnt_exist") is None


def test_is_installed():
    assert is_installed("kipoi>=0.1")
    assert is_installed("kipoi<=10.1")
    assert not is_installed("kipoi>=10.1")
    assert is_installed("kipoi>=0.1,>=0.2")
    assert is_installed("kipoi>=0.1,>0.2")
    assert not is_installed("package_doesnt_exist")


def test_dependencies_all_installed():
    assert Dependencies(conda=["numpy"], pip=["kipoi"]).all_installed()
    assert Dependencies(conda=["numpy"], pip=["kipoi>=0.1"]).all_installed()
    assert Dependencies(conda=["numpy>0.1"], pip=["kipoi>=0.1"]).all_installed()
    assert not Dependencies(conda=["numpy>0.1"], pip=["kipoi>=10.1"]).all_installed()
    assert not Dependencies(conda=["numpy>0.1"], pip=["kipoi>=10.1"]).all_installed(verbose=True)
    assert not Dependencies(conda=["package_doesnt_exist>0.1"], pip=["kipoi>=10.1"]).all_installed(verbose=True)
