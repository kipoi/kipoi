"""TensorFlow2Model tests
"""
import os
import subprocess

import numpy as np
import pytest

from config import pythonversion
import kipoi
from kipoi.model import TensorFlow2Model

@pythonversion
def test_loading():
    savedmodel_path = "example/models/iris_tensorflow2/model_files/"
    a = TensorFlow2Model(savedmodel_path=savedmodel_path)
    o = a.predict_on_batch([np.ones((3, 4))])
    assert o.shape == (1, 3, 3)

@pythonversion
def test_loading_from_kipoi():
    m = kipoi.get_model("example/models/iris_tensorflow2", source="dir")
    o = m.predict_on_batch([np.ones((3, 4))])
    assert o.shape == (1, 3, 3)
    
@pythonversion
def test_test_example(tmpdir):
    """kipoi test ..., add also output file writing
    """
    args = ["python", "./kipoi/__main__.py", "test",
            "--batch_size=4",
            "--keep_metadata",
            "example/models/iris_tensorflow2"]
    returncode = subprocess.call(args=args)
    assert returncode == 0

