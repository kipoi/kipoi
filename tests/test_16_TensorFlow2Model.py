"""TensorFlowModel tests
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import kipoi
from kipoi.model import TensorFlow2Model


def test_loading():
    if sys.version_info.major == 3 and sys.version_info.minor < 8:
        pytest.skip("example is only supported >= python 3.8 ")

    savedmodel_path = "example/models/iris_tensorflow2/model_files/"
    a = TensorFlow2Model(savedmodel_path=savedmodel_path)
    o = a.predict_on_batch(np.ones((3, 4)))
    assert o.shape == (3, 3)

def test_loading_from_kipoi():
    if sys.version_info.major == 3 and sys.version_info.minor < 8:
        pytest.skip("example is only supported >= python 3.8 ")

    m = kipoi.get_model("example/models/iris_tensorflow2", source="dir")
    o = m.predict_on_batch(np.ones((3, 4)))
    assert o.shape == (3, 3)
    
def test_test_example(tmpdir):
    """kipoi test ..., add also output file writing
    """
    if sys.version_info.major == 3 and sys.version_info.minor < 8:
        pytest.skip("example is only supported >= python 3.8 ")

    args = ["python", "./kipoi/__main__.py", "test",
            "--batch_size=4",
            "--keep_metadata",
            "example/models/iris_tensorflow2"]
    returncode = subprocess.call(args=args)
    assert returncode == 0

