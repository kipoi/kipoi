"""TensorFlowModel tests
"""
import pytest
import os
import numpy as np
import time
from kipoi.model import TensorFlowModel

from kipoi.rpyc_model import *

from utils import *

@pytest.mark.flaky(max_runs=20)
class TestTensorflowRpyc(object):


    @pytest.mark.parametrize("port", [3000,3010])
    def test_loading(self, port):

        import tensorflow as tf
        checkpoint_path = "example/models/iris_tensorflow/model_files/model.ckpt"
        const_feed_dict_pkl = "example/models/iris_tensorflow/model_files/const_feed_dict.pkl"


        s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=True, address='localhost', port=port, logging_level=1)


        # dict of variables
        # input = list
        for x in range(1):
            with RemoteTensorFlowModel(s,input_nodes="inputs",target_nodes="probas",checkpoint_path=checkpoint_path) as a:
                o = a.predict_on_batch(np.ones((3, 4)))
                assert o.shape == (3, 3)

        