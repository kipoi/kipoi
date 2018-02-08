"""TensorFlowModel tests
"""
import pytest
import os
import tensorflow as tf
import numpy as np
from kipoi.model import TensorFlowModel
# fixture


def test_loading():
    checkpoint_path = "examples/iris_tensorflow/model_files/model.ckpt"
    const_feed_dict_pkl = "examples/iris_tensorflow/model_files/const_feed_dict.pkl"

    # dict of variables
    # input = list
    a = TensorFlowModel(input_nodes="inputs",
                        target_nodes="probas",
                        checkpoint_path=checkpoint_path
                        )
    o = a.predict_on_batch(np.ones((3, 4)))
    assert o.shape == (3, 3)

    # input = dict
    a = TensorFlowModel(input_nodes={"out_name": "inputs"},
                        target_nodes="probas",
                        checkpoint_path=checkpoint_path
                        )
    with pytest.raises(AssertionError):
        o = a.predict_on_batch(np.ones((3, 4)))

    o = a.predict_on_batch({"out_name": np.ones((3, 4))})
    assert o.shape == (3, 3)

    # input = list
    a = TensorFlowModel(input_nodes=["inputs"],
                        target_nodes="probas",
                        checkpoint_path=checkpoint_path
                        )
    with pytest.raises(AssertionError):
        o = a.predict_on_batch(np.ones((3, 4)))

    o = a.predict_on_batch([np.ones((3, 4))])
    assert o.shape == (3, 3)

    # output = dict
    a = TensorFlowModel(input_nodes="inputs",
                        target_nodes={"out_name": "probas"},
                        checkpoint_path=checkpoint_path
                        )
    o = a.predict_on_batch(np.ones((3, 4)))
    assert isinstance(o, dict)
    assert list(o.keys()) == ["out_name"]
    assert o['out_name'].shape == (3, 3)

    # output = list
    a = TensorFlowModel(input_nodes="inputs",
                        target_nodes=["probas"],
                        checkpoint_path=checkpoint_path
                        )
    o = a.predict_on_batch(np.ones((3, 4)))
    assert isinstance(o, list)
    assert len(o) == 1
    assert o[0].shape == (3, 3)

    # test with the extra input
    a = TensorFlowModel(input_nodes="inputs",
                        target_nodes="probas",
                        checkpoint_path=checkpoint_path,
                        const_feed_dict_pkl=const_feed_dict_pkl
                        )
    o = a.predict_on_batch(np.ones((3, 4)))
    assert o.shape == (3, 3)
