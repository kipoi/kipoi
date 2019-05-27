import warnings

warnings.filterwarnings('ignore')
import pytest
import sys
import os
import yaml
from contextlib import contextmanager
import kipoi
from kipoi.pipeline import install_model_requirements
from kipoi_utils.utils import Slice_conv,_call_command
import config
import numpy as np
from kipoi_utils.utils import cd
from kipoi_conda.utils import get_kipoi_bin
from kipoi.rpyc_model import *
from kipoi.cli.env import *





import subprocess


from test_16_KerasModel import (get_sample_functional_model,cd, get_sample_functional_model_input,get_test_kwargs,read_json_yaml, get_sample_sequential_model)


INSTALL_REQ = config.install_req

EXAMPLES_TO_RUN = ["extended_coda","rbp"]  # "pyt" not used as gradients are not yet supported for pytorch model.
PORTS =  [18838]

MODEL_PORT_OFFSET = {
    "extended_coda" : 0,
    "rbp":2
}

@contextmanager
def cd_local_and_remote(newdir, remote_model):
    """Temporarily change the directory
    """
    prevdir = os.getcwd()
    nd = os.path.expanduser(newdir)


    remote_model.cd(nd)
    os.chdir(nd)
    
    try:
        yield
    finally:
        remote_model.cd(prevdir)
        os.chdir(prevdir)
        




def create_model_env(model,source,tmpdir=None):
    if isinstance(model, str):
        model = [model]
    import uuid

    class Args(object):
        def __init__(self, model, source):
            self.model = model
            self.source = source        
        def _get_kwargs(self):
            kwargs = {"dataloader": [], "env":None, "gpu": False, "model": self.model, "source": self.source,
                  "tmpdir": "something", "vep": False}
            return kwargs

    # create the tmp dir
    if tmpdir is None:
        tmpdir = "/tmp/kipoi/envfiles/" + str(uuid.uuid4())[:8]
    else:
        tmpdir = args.tmpdir
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    # write the env file
    logger.info("Writing environment file: {0}".format(tmpdir))


    env, env_file = export_env(model,
                               None,
                               source,
                               env_file=None,
                               env_dir=tmpdir,
                               env=None,
                               vep=False,
                               interpret=False,
                               gpu=False)

    args = Args(model=model, source=source)
    env_db_entry = generate_env_db_entry(args, args_env_overload=env)
    envdb = get_model_env_db()
    envdb.append(env_db_entry)
    envdb.save()

    # setup the conda env from file
    kipoi_conda.create_env_from_file(env_file)
    env_db_entry.successful = True

    # env is environment name
    env_db_entry.cli_path = get_kipoi_bin(env)
    get_model_env_db().save()



@contextmanager
def create_tmp_env(model, use_current_python, source, delete_env=True):
    env_name = get_env_name(model,source=source)
    if not use_current_python:

        # this SHOULD not be needed since we delete the envs at the end...
        try:

            args = ["env","remove","-n",env_name]
            _call_command('conda' ,extra_args=args)

        except:
            pass

        # create the env for the model
        create_model_env(model=model, source=source)
        #args = ["./kipoi/__main__.py", "env", "create", model, "--source",source]
        #_call_command('python' ,extra_args=args,cwd=os.getcwd(),use_stdout=True)

        
    else:
        env_name = None

    try:
        yield env_name
    finally:
        if delete_env and not use_current_python:

            try:
                pass
                # env of that model
                model_env = get_envs_by_model(model, source=source, only_most_recent=True, only_valid=True)[-1]
                env_name = model_env.create_args.env
                #delete_envs([model_env])
            except:
                pass





@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
@pytest.mark.parametrize("port", [18838])
def test_create_tmp_env(example, port):
    """Test extractor
    """

    port = port + MODEL_PORT_OFFSET[example]

    import keras
    backend = keras.backend._BACKEND
    if backend == 'theano' and example == "rbp":
        pytest.skip("extended_coda example not with theano ")
    #
    example_dir = "example/models/{0}".format(example)
    # install the dependencies
    # - TODO maybe put it implicitly in load_dataloader?
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)
    #
    Dl = kipoi.get_dataloader_factory(example_dir, source="dir")
    #
    test_kwargs = get_test_kwargs(example_dir)
    #
    # install the dependencies
    # - TODO maybe put it implicitly in load_extractor?
    if INSTALL_REQ:
        install_model_requirements(example_dir, source="dir")
    #
    # get model
    model = kipoi.get_model(example_dir, source="dir")
    


    with create_tmp_env(use_current_python=False, model=example_dir, source='dir') as env_name:

        

        pass







@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
@pytest.mark.parametrize("use_current_python",  [False])
@pytest.mark.parametrize("port", [18838])
def test_activation_function_model(example, use_current_python, port):
    """Test extractor
    """

    port = port + int(use_current_python)
    port = port + MODEL_PORT_OFFSET[example]




    import keras
    backend = keras.backend._BACKEND
    if backend == 'theano' and example == "rbp":
        pytest.skip("extended_coda example not with theano ")
    #
    example_dir = "example/models/{0}".format(example)
    # install the dependencies
    # - TODO maybe put it implicitly in load_dataloader?
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)
    #
    Dl = kipoi.get_dataloader_factory(example_dir, source="dir")
    #
    test_kwargs = get_test_kwargs(example_dir)
    #
    # install the dependencies
    # - TODO maybe put it implicitly in load_extractor?
    if INSTALL_REQ:
        install_model_requirements(example_dir, source="dir")
    
    # get model
    model = kipoi.get_model(example_dir, source="dir")
    


    with create_tmp_env(use_current_python=use_current_python, model=example_dir, source='dir') as env_name:

        

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

@pytest.mark.parametrize("port", [18842])
def test_keras_get_layers_and_outputs(port):


    use_current_python = True
    s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=use_current_python, address='localhost', port=port, logging_level=0)
    with kipoi.model.RemoteKerasModel(s, *get_sample_functional_model()) as model:

        backend = model.keras_backend
        selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs("shared_lstm")
        print("SEL",selected_layers)
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
@pytest.mark.parametrize("port",  [18846])
def test_predict_on_batch(example, use_current_python, port):
    """Test extractor
    """
    port = port + int(use_current_python)
    port = port + MODEL_PORT_OFFSET[example]

    import keras
    backend = keras.backend._BACKEND




    if backend == 'theano' and example == "rbp":
        pytest.skip("extended_coda example not with theano ")
    #
    example_dir = "example/models/{0}".format(example)
    # install the dependencies
    # - TODO maybe put it implicitly in load_dataloader?
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)
    #
    Dl = kipoi.get_dataloader_factory(example_dir, source="dir")
    #
    test_kwargs = get_test_kwargs(example_dir)
    #    # install the dependencies
    # - TODO maybe put it implicitly in load_extractor?
    if INSTALL_REQ:
        install_model_requirements(example_dir, source="dir")
    




    with create_tmp_env(use_current_python=use_current_python, model=example_dir, source='dir') as env_name:

        # get remote model
        s = kipoi.rpyc_model.ServerArgs(env_name=env_name,use_current_python=use_current_python,  address='localhost', port=port, logging_level=0)
        with  kipoi.get_model(example_dir, source="dir", server_settings=s) as remote_model:
            
            with cd(example_dir + "/example_files"):
                # initialize the dataloader
                dataloader = Dl(**test_kwargs)
                #
                # sample a batch of data
                it = dataloader.batch_iter()
                batch = next(it)
                # predict with a model
                #res = model.predict_on_batch(batch["inputs"])
                remote_res = remote_model.predict_on_batch(batch["inputs"])

                #numpy.testing.assert_allclose(res, remote_res)




@pytest.mark.parametrize("example",             EXAMPLES_TO_RUN)
@pytest.mark.parametrize("use_current_python",  [True, False])
@pytest.mark.parametrize("port",  [18850])
def test_pipeline(example, use_current_python, port):
    """Test extractor
    """

    port = port + int(use_current_python)
    port = port + MODEL_PORT_OFFSET[example]

    import keras
    backend = keras.backend._BACKEND


    if backend == 'theano' and example == "rbp":
        pytest.skip("extended_coda example not with theano ")
    #
    example_dir = "example/models/{0}".format(example)
    # install the dependencies
    # - TODO maybe put it implicitly in load_dataloader?
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)
    #
    Dl = kipoi.get_dataloader_factory(example_dir, source="dir")
    #
    test_kwargs = get_test_kwargs(example_dir)
    #    # install the dependencies
    # - TODO maybe put it implicitly in load_extractor?
    if INSTALL_REQ:
        install_model_requirements(example_dir, source="dir")
    


    # get model
    model = kipoi.get_model(example_dir, source="dir")
    env_name = None




    with create_tmp_env(use_current_python=use_current_python, model=example_dir, source='dir') as env_name:

        # get remote model
        s = kipoi.rpyc_model.ServerArgs(env_name=env_name,use_current_python=use_current_python,  address='localhost', port=port, logging_level=0)
        with  kipoi.get_model(example_dir, source="dir", server_settings=s) as remote_model:
            
            newdir = example_dir + "/example_files"
            assert not os.path.isabs(example_dir)
            assert not os.path.isabs(newdir)
            with cd_local_and_remote(newdir, remote_model):
                # initialize the dataloader
                dataloader = Dl(**test_kwargs)
                

                pipeline = remote_model.pipeline
                the_pred = pipeline.predict(dataloader_kwargs=test_kwargs)


@pytest.mark.parametrize("port",  [18854])
def test_returned_gradient_fmt(port):

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

