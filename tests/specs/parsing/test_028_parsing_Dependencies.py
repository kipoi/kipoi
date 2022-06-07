"""Test parsing dependencies
"""
import pytest
from pytest import raises
from kipoi_utils.utils import read_txt
from kipoi.specs import Dependencies
from related import from_yaml

CLS = Dependencies

GOOD_EXAMPLES = [
    ("""
conda:
  - conda_dep1
  - conda_dep2
pip:
  - pip_dep1
  - pip_dep2
""", ["conda_dep1", "conda_dep2"],
     ["pip_dep1", "pip_dep2"]
     ),
    ("""
conda:
  - conda_dep1
  - conda_dep2
pip:
  - pip_dep1
  - pip_dep2
conda_file: tests/data/conda-env.yaml
""", ["conda_dep3", "conda_dep4"],
     ["pip_dep3", "pip_dep4"]
     ),
    ("""
conda: tests/data/conda_requirements.txt
pip: tests/data/pip_requirements.txt
""", ["conda_dep1", "conda_dep2"],
     ["pip_dep1", "pip_dep2"]
     )
]


def test_read_txt():
    lines = read_txt("tests/data/conda_requirements.txt")
    assert lines == ["conda_dep1", "conda_dep2"]


@pytest.mark.parametrize("info_str,conda,pip", GOOD_EXAMPLES)
def test_parse_correct_info(info_str, conda, pip):
    # loading works
    deps = CLS.from_config(from_yaml(info_str))

    assert deps.conda == conda
    assert deps.pip == pip


def test_gpu():
    # tensorflow
    deps = Dependencies(pip=["tensorflow==1.4"])
    assert deps.gpu().pip == ["tensorflow-gpu==1.4"]

    # pytorch
    deps = Dependencies(conda=["pytorch::pytorch-cpu"])
    assert deps.gpu().conda == ["pytorch"]

    # nothing changed
    deps = Dependencies(pip=["foo"], conda=["bar"])
    assert deps.gpu() == deps.normalized()

def test_pytorch_cpu():
    deps = Dependencies(conda=['python=3.7', 'numpy=1.19.2', 'pytorch-cpu=1.3.1', 'torchvision-cpu=0.3.0', 'pip=21.0.1', 'numpy'])
    print(sorted(deps.normalized().conda))
    assert sorted(deps.normalized().conda) == ['cpuonly', 'numpy', 'numpy=1.19.2', 'pip=21.0.1', 'python=3.7', 'pytorch=1.3.1', 'torchvision=0.3.0']