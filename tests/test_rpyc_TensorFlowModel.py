"""TensorFlowModel tests
"""
import pytest
import os
import numpy as np
import time
from kipoi.model import TensorFlowModel

from kipoi.rpyc_model import *

from utils import *

PORTS =  [18838, 18839, 18838]

@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("port", PORTS)
def test_loading(port):

    with port_filelock(port):

        import tensorflow as tf
        checkpoint_path = "example/models/iris_tensorflow/model_files/model.ckpt"
        const_feed_dict_pkl = "example/models/iris_tensorflow/model_files/const_feed_dict.pkl"


        s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=True, address='localhost', port=port, logging_level=0)


        # dict of variables
        # input = list
        for x in range(4):
            with RemoteTensorFlowModel(s,input_nodes="inputs",target_nodes="probas",checkpoint_path=checkpoint_path) as a:
                o = a.predict_on_batch(np.ones((3, 4)))
                assert o.shape == (3, 3)

    # # input = dict
    # with RemoteTensorFlowModel(s,input_nodes={"out_name": "inputs"}, target_nodes="probas",
    #                     checkpoint_path=checkpoint_path
    #                     ) as a:
    #     with pytest.raises(AssertionError):
    #         o = a.predict_on_batch(np.ones((3, 4)))

    #     o = a.predict_on_batch({"out_name": np.ones((3, 4))})
    #     assert o.shape == (3, 3)

    # # input = list
    # with RemoteTensorFlowModel(s,input_nodes=["inputs"], target_nodes="probas",
    #                     checkpoint_path=checkpoint_path
    #                     ) as a:
    #     with pytest.raises(AssertionError):
    #         o = a.predict_on_batch(np.ones((3, 4)))

    #     o = a.predict_on_batch([np.ones((3, 4))])
    #     assert o.shape == (3, 3)

    # # output = dict
    # with RemoteTensorFlowModel(s,input_nodes="inputs", target_nodes={"out_name": "probas"},
    #                     checkpoint_path=checkpoint_path
    #                     ) as a:
    #     o = a.predict_on_batch(np.ones((3, 4)))
    #     assert isinstance(o, dict)
    #     assert list(o.keys()) == ["out_name"]
    #     assert o['out_name'].shape == (3, 3)

    # # output = list
    # with RemoteTensorFlowModel(s,input_nodes="inputs",target_nodes=["probas"],
    #                     checkpoint_path=checkpoint_path
    #                     ) as a:
    #     o = a.predict_on_batch(np.ones((3, 4)))
    #     assert isinstance(o, list)
    #     assert len(o) == 1
    #     assert o[0].shape == (3, 3)

    # # test with the extra input
    # with RemoteTensorFlowModel(s,input_nodes="inputs",target_nodes="probas",
    #                     checkpoint_path=checkpoint_path,
    #                     const_feed_dict_pkl=const_feed_dict_pkl
    #                     ) as a:
    #     o = a.predict_on_batch(np.ones((3, 4)))
    #     assert o.shape == (3, 3)


# class DummySlice():
#     def __getitem__(self, key):
#         return key


# def test_grad_tens_generation():
#     import tensorflow as tf
#     checkpoint_path = "example/models/iris_tensorflow/model_files/model.ckpt"
#     a = TensorFlowModel(input_nodes="inputs",
#                         target_nodes="probas",
#                         checkpoint_path=checkpoint_path
#                         )
#     fwd_values = a.predict_on_batch(np.ones((3, 4)))
#     assert np.all(a.get_grad_tens(fwd_values, DummySlice()[:, 0:1], "min")[0, :] == np.array([1, 0, 0]))
#     assert np.all(a.get_grad_tens(fwd_values, DummySlice()[:, 0:2], "min")[0, :] == np.array([1, 0, 0]))
#     assert np.all(a.get_grad_tens(fwd_values, DummySlice()[:, 0:2], "max")[0, :] == np.array([0, 1, 0]))
#     assert np.all(
#         a.get_grad_tens(fwd_values, DummySlice()[0:2], "max")[0, :] == a.get_grad_tens(fwd_values, DummySlice()[:, 0:2],
#                                                                                        "max")[0, :])


# def test_activation_on_batch():
#     checkpoint_path = "example/models/iris_tensorflow/model_files/model.ckpt"
#     const_feed_dict_pkl = "example/models/iris_tensorflow/model_files/const_feed_dict.pkl"

#     # dict of variables
#     # input = list
#     a = TensorFlowModel(input_nodes="inputs",
#                         target_nodes="probas",
#                         checkpoint_path=checkpoint_path
#                         )
#     a.predict_activation_on_batch(np.ones((3, 4)), layer="logits")
