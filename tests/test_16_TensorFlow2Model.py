"""TensorFlowModel tests
"""
import os
import sys

import numpy as np
import pytest

import kipoi
from kipoi.model import TensorFlow2Model


def test_loading():
    if sys.version_info.major == 3 and sys.version_info.minor < 8:
        pytest.skip("example is only supported >= python 3.8 ")

    import tensorflow as tf
    savedmodel_path = "example/models/iris_tensorflow2/model_files/"
    a = TensorFlow2Model(savedmodel_path=savedmodel_path)
    o = a.predict_on_batch(np.ones((3, 4)))
    assert o.shape == (3, 3)

def test_loading_from_kipoi():
    if sys.version_info.major == 3 and sys.version_info.minor < 8:
        pytest.skip("example is only supported >= python 3.8 ")

    import tensorflow as tf
    m = kipoi.get_model("example/models/iris_tensorflow2", source="dir")

