"""Test conda env installation
"""

from collections import OrderedDict
import pytest
import kipoi
import kipoi.conda


# TODO - pytest run in a special conda environment for other CLI tests
def test_create_env():
    dependencies = ["python=3.6", "numpy",
                    OrderedDict(pip=["tqdm"])
                    ]

    ENV_NAME = "kipoi-test-env1"
    kipoi.conda.create_env(ENV_NAME, dependencies)
    dependencies = ["python=3.6", "numpyxzy"]
    # check that file exists
    assert kipoi.conda.env_exists(ENV_NAME)
    # remove the environment
    kipoi.conda.remove_env("kipoi-test-env1")
    assert not kipoi.conda.env_exists(ENV_NAME)

    with pytest.raises(Exception):
        kipoi.conda.create_env("kipoi-test-env2", dependencies)
