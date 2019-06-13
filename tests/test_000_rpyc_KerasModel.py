import warnings

warnings.filterwarnings('ignore')
import pytest
import sys
import os
import yaml
from contextlib import contextmanager
import kipoi
from kipoi.pipeline import install_model_requirements
from kipoi_utils.utils import _call_command
import config
import numpy as np
from kipoi_utils.utils import cd
from kipoi_conda.utils import get_kipoi_bin
from kipoi.rpyc_model import *
from kipoi.cli.env import *


from utils import *


import subprocess


from test_16_KerasModel import (get_sample_functional_model,cd, get_sample_functional_model_input,get_test_kwargs,read_json_yaml, get_sample_sequential_model)


INSTALL_REQ = config.install_req

EXAMPLES_TO_RUN = ["extended_coda","rbp"]  


E_TO_OFFSET = {
    "extended_coda": 0,
    "rbp": 4    
}
    
@pytest.mark.flaky(max_runs=20)
class TestKerasRpyc(object):

    @pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
    @pytest.mark.parametrize("use_current_python",  [True, False])
    @pytest.mark.parametrize("port", [1030,1040])
    def test_activation_function_model(self, example, use_current_python, port):

        port += E_TO_OFFSET[example] + int(use_current_python)
        
        example_dir = "example/models/{0}".format(example)
        Dl = kipoi.get_dataloader_factory(example_dir, source="dir")
        test_kwargs = get_test_kwargs(example_dir)

        
        # ensure we got the env
        env_name = create_env_if_not_exist(bypass=use_current_python, model=example_dir, source='dir')

        # get remote model
        s = kipoi.rpyc_model.ServerArgs(env_name=env_name,  address='localhost', port=port, logging_level=0, use_current_python=use_current_python)
        with  kipoi.get_model(example_dir, source="dir", server_settings=s) as remote_model:

                with cd(example_dir + "/example_files"):

                    # initialize the dataloader
                    dataloader = Dl(**test_kwargs)
                    #
                    it = dataloader.batch_iter()
                    batch = next(it)


                    #bar = remote_model.predict_fobar('x')

                    # predict with a model
                    remote_model.predict_on_batch(batch["inputs"])
                    remote_model.predict_activation_on_batch(batch["inputs"], layer=len(remote_model.model.layers) - 2)

                    if example == "rbp":
                        remote_model.predict_activation_on_batch(batch["inputs"], layer="flatten_6")

    @pytest.mark.parametrize("port", [1050, 1060])
    def test_keras_get_layers_and_outputs(self, port):

        s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=True, address='localhost', port=port, logging_level=0)
        with kipoi.model.RemoteKerasModel(s, *get_sample_functional_model()) as model:

            backend = model.keras_backend
            selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs("shared_lstm")
            assert len(selected_layers) == 1
            assert selected_layers[0].name == "shared_lstm"
            assert len(sel_outputs) == 2
            assert len(sel_output_dims) == 2
            if backend != 'theano':
                with pytest.raises(Exception):  # expect exception
                    # LSTM activation layer has non-trivial input
                    selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs("shared_lstm",
                                                                                                 pre_nonlinearity=True)
                selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs(use_final_layer=True,
                                                                                             pre_nonlinearity=True)
                assert len(selected_layers) == 1
                assert selected_layers[0].name == "final_layer"
                assert len(sel_outputs) == 1
                assert sel_outputs[0] != selected_layers[0].output
                assert len(sel_output_dims) == 1
            selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs(use_final_layer=True,
                                                                                         pre_nonlinearity=False)
            assert len(selected_layers) == 1
            assert selected_layers[0].name == "final_layer"
            assert len(sel_outputs) == 1
            assert sel_outputs[0] == selected_layers[0].output
            assert len(sel_output_dims) == 1
            
            # using the sequential model
            model = kipoi.model.KerasModel(*get_sample_sequential_model())
            selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs(2)
            assert len(selected_layers) == 1
            assert selected_layers[0].name == "hidden"
            assert len(sel_outputs) == 1
            assert len(sel_output_dims) == 1
            if backend != 'theano':
                selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs(use_final_layer=True,
                                                                                             pre_nonlinearity=True)
                assert len(selected_layers) == 1
                assert selected_layers[0].name == "final"
                assert len(sel_outputs) == 1
                assert sel_outputs[0] != selected_layers[0].output
                assert len(sel_output_dims) == 1
            selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs(use_final_layer=True,
                                                                                         pre_nonlinearity=False)
            assert len(selected_layers) == 1
            assert selected_layers[0].name == "final"
            assert len(sel_outputs) == 1
            assert sel_outputs[0] == selected_layers[0].output
            assert len(sel_output_dims) == 1



    @pytest.mark.parametrize("example",             EXAMPLES_TO_RUN)
    @pytest.mark.parametrize("use_current_python",  [True, False])
    @pytest.mark.parametrize("port",  [1080, 1090])
    def test_predict_on_batch(self, example, use_current_python, port):

        port += E_TO_OFFSET[example] + int(use_current_python)


        example_dir = "example/models/{0}".format(example)
        Dl = kipoi.get_dataloader_factory(example_dir, source="dir")
        test_kwargs = get_test_kwargs(example_dir)


        env_name = create_env_if_not_exist(bypass=use_current_python, model=example_dir, source='dir')
        # get remote model
        s = kipoi.rpyc_model.ServerArgs(env_name=env_name,use_current_python=use_current_python,  address='localhost', port=port, logging_level=0)
        with  kipoi.get_model(example_dir, source="dir", server_settings=s) as remote_model:
            
            with cd(example_dir + "/example_files"):
                # initialize the dataloader
                dataloader = Dl(**test_kwargs)
                
                # sample a batch of data
                it = dataloader.batch_iter()
                batch = next(it)
                remote_res = remote_model.predict_on_batch(batch["inputs"])


    @pytest.mark.parametrize("example",             EXAMPLES_TO_RUN)
    @pytest.mark.parametrize("use_current_python",  [True, False])
    @pytest.mark.parametrize("port",  [1120, 1130])
    def test_pipeline(self, example, use_current_python, port):
        """Test extractor
        """
        port += E_TO_OFFSET[example] + int(use_current_python)

        example_dir = "example/models/{0}".format(example)
        test_kwargs = get_test_kwargs(example_dir)

        # get model
        env_name = create_env_if_not_exist(bypass=use_current_python, model=example_dir, source='dir')

        # get remote model
        s = kipoi.rpyc_model.ServerArgs(env_name=env_name,use_current_python=use_current_python,  address='localhost', port=port, logging_level=0)
        with  kipoi.get_model(example_dir, source="dir", server_settings=s) as remote_model:
            
            newdir = example_dir + "/example_files"
            assert not os.path.isabs(example_dir)
            assert not os.path.isabs(newdir)
            with remote_model.cd_local_and_remote(newdir):

                pipeline = remote_model.pipeline
                the_pred = pipeline.predict(dataloader_kwargs=test_kwargs)
                assert the_pred is not None

    def test_multi_server_pipeline(self):

        example_dir_0= "example/models/extended_coda"
        example_dir_1= "example/models/rbp"
        test_kwargs_0 = get_test_kwargs(example_dir_0)
        test_kwargs_1 = get_test_kwargs(example_dir_1)
        env_name_0 = create_env_if_not_exist(model=example_dir_0, source='dir')
        env_name_1 = create_env_if_not_exist(model=example_dir_1, source='dir')
        # ports = [1140, 1150]    
        # get remote model
        assert ports[0] != ports[1]
        s0 = kipoi.rpyc_model.ServerArgs(env_name=env_name_0,use_current_python=False,  address='localhost', port=ports[0], logging_level=0)
        s1 = kipoi.rpyc_model.ServerArgs(env_name=env_name_1,use_current_python=False,  address='localhost', port=ports[1], logging_level=0)


        with  kipoi.get_model(example_dir_0, source="dir", server_settings=s0) as remote_model_0:
            with  kipoi.get_model(example_dir_1, source="dir", server_settings=s1) as remote_model_1:   

                with remote_model_0.cd_remote(example_dir_0 + "/example_files"):
                    with remote_model_1.cd_remote(example_dir_1+ "/example_files"):

                        pipeline_0 = remote_model_0.pipeline
                        pipeline_1 = remote_model_1.pipeline
                        the_pred_0 = pipeline_0.predict(dataloader_kwargs=test_kwargs_0)
                        the_pred_1 = pipeline_1.predict(dataloader_kwargs=test_kwargs_1)
                        assert the_pred_0 is not None
                        assert the_pred_1 is not None


    #@pytest.mark.flaky(max_runs=1)
    @pytest.mark.parametrize("port",  [1161, 1171])
    def test_returned_gradient_fmt(self, port):

        use_current_python = True
        s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=use_current_python, address='localhost', port=port, logging_level=0)
        with kipoi.model.RemoteKerasModel(s, *get_sample_functional_model()) as model:
            sample_input = get_sample_functional_model_input(kind="list")
            grad_out = model.input_grad(sample_input, final_layer=True, avg_func="absmax")
            assert isinstance(grad_out, type(sample_input))
            assert len(grad_out) == len(sample_input)
            sample_input = get_sample_functional_model_input(kind="dict")
            grad_out = model.input_grad(sample_input, final_layer=True, avg_func="absmax")
            assert isinstance(grad_out, type(sample_input))
            assert len(grad_out) == len(sample_input)
            assert all([k in grad_out for k in sample_input])

