import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
import sys
import json
import os
import signal
import subprocess
import sched,time
from collections import OrderedDict, Mapping
from subprocess import Popen, PIPE, STDOUT, check_output
import atexit
import numpy
from . base_model import BaseModel
from kipoi_conda import get_envs, get_env_path
from kipoi.cli.env import get_envs_by_model
from kipoi_utils import (load_module, cd, merge_dicts, read_pickle, override_default_kwargs,
                    load_obj, inherits_from, infer_parent_class, makedir_exist_ok)

import logging




def call_script_in_env(filename, env_name=None, use_current_python=False, args=None, cwd=None): 
    """run an python script in a certain conda enviroment in a background
    process.
    
    Args:
        filename (str): path to the python script
        env_name (str or None, optional): Name of the conda enviroment.
        use_current_python (bool, False) If true, the current python executable will be used
        args (None, optional): args to pass to the script
    
    Returns:
        Popen: instance of Popen / running program
    """


    def activate_env(env_name=None,env_path=None):
        if env_path is None:
            assert env_name is not None
            env_path = get_env_path(env_name)
        bin_path = os.path.join(env_path,'bin')
        new_env = os.environ.copy()
        new_env['PATH'] = bin_path + os.pathsep + new_env['PATH']

    if use_current_python:
        python_path = sys.executable
    else:
        env_path = get_env_path(env_name=env_name)
        activate_env(env_path=env_path)
        python_path = os.path.join(env_path,'bin','python')

        #subprocess.run(, shell=True)
    # The os.setsid() is passed in the argument preexec_fn so
    # it's run after the fork() and before  exec() to run the shell.
    if args is None:
        args = []
    
    pro = subprocess.Popen([python_path, filename] + list(args), stdout=subprocess.PIPE, 
                           shell=False, close_fds=True,  preexec_fn=os.setsid,cwd=cwd)
    return pro




class ModelArgs(object):
    def __init__(self, *args, **kwargs):
        self.args = list(*args)
        self.kwargs = dict(**kwargs)

class ServerArgs(dict):
    def __init__(self,env_name, address, port, use_current_python=False,connection_timeout=1.0, logging_level=None, cwd=None):
        super(ServerArgs, self).__init__()


        self['address'] = str(address)
        self['port'] = int(port)
        self['env_name'] = env_name
        self['use_current_python'] = use_current_python
        self['connection_timeout'] = float(connection_timeout)
        self['logging_level'] = logging_level
        if logging_level is None:
            self['logging_level']= logging.getLogger().level
        self['cwd'] = cwd
        


    def setEnv(self, env_name):
        self['env_name'] = str(env_name)




class RpycServer(object):

    def __init__(self, env_name, model_type, address, port, use_current_python=False, connection_timeout=1.0, logging_level=None, cwd=None):
        
        self.address = str(address)
        self.port = int(port)
        self.env_name = env_name
        self.model_type = model_type
        self.connection_timeout = float(connection_timeout)
        self.use_current_python = bool(use_current_python)
        self.logging_level = logging_level
        if logging_level is None:
            self.logging_level = logging.getLogger().level

        self.server_process = None
        self.cwd = cwd
        if cwd is None:
            self.cwd = os.getcwd()

        # in any case, we want the server to stop when we stop the python session 
        def killserver_atexit():
            try:
                self.connection.close()  
            except:
                pass
            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)  
            except:
                pass
        atexit.register(killserver_atexit)

    @property
    def is_running(self):
        return self.server_process is not None
    

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self):
        if self.server_process is not None:
            raise RuntimeError("server process already running")
        # start the server 
        server_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rpyc_server.py')
        args = [ 
            str(self.port), 
            str(self.model_type),
            '--logging_level', 
            str(self.logging_level)
        ]
        print('cwd',self.cwd)
        self.server_process =  call_script_in_env(env_name=self.env_name, use_current_python=self.use_current_python,
            filename=server_script_path, args=args, cwd=self.cwd)

        self.connection = self.__get_connection()


    def stop(self):
        if self.server_process is None:
            raise RuntimeError("server process is not running")

        try:
            self.connection.close()  
        except:
            pass
        try:
            os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)  
        except:
            pass


    def __get_connection(self):
        t0 = time.clock()

        while True:
            try:
                config=dict(allow_all_attrs=True, allow_public_attrs=True)
                c = rpyc.connect(self.address, self.port, config=config)
                return c
            except ConnectionRefusedError:
                pass
            
            t1 = time.clock()
            d = t1 - t0

            if d > float(self.connection_timeout):
                raise TimeoutError("try_connect time out: cannot connect")


    def echo(self, x):
        if not self.is_running:
            raise RuntimeError("server is not running")
        res = self.connection.root.echo(x)
        return res




class RemoteGradientMixin(object):
    allowed_functions = ["sum", "max", "min", "absmax"]

    def input_grad(self, x, filter_idx=None, avg_func=None, layer=None, final_layer=True,
                   selected_fwd_node=None, pre_nonlinearity=False):
        return self._remote.input_grad(x=x, filter_idx=filter_idx, avg_func=avg_func,
                   layer=layer, final_layer=final_layer,selected_fwd_node=selected_fwd_node, 
                   pre_nonlinearity=pre_nonlinearity)





class RemoteLayerActivationMixin(object):
    #allowed_functions = ["sum", "max", "min", "absmax"]

    def predict_activation_on_batch(self, x, layer, pre_nonlinearity=False):
        return self._remote.predict_activation_on_batch(x=x, layer=layer, 
                   pre_nonlinearity=pre_nonlinearity)

class RemoteModel(BaseModel):
    def __init__(self, server_settings, model_type, model_args, cwd=None):
        super(RemoteModel, self).__init__()
        self.server = RpycServer(model_type=model_type, **server_settings)
        self.server.start()
        self._c = self.server.connection
        self._remote = self._c.root
        self._model_type = model_type
        self._model_args = model_args

        self.cwd = cwd
        if cwd is None:
            self.cwd = os.getcwd()

        # cd to the right dir
        self.cd(self.cwd)

        # initalize the model 
        self._remote._initialize(*self._model_args.args, **self._model_args.kwargs)
    

    def cd(self, newdir):
        try:
            newdir_init = newdir
            if not os.path.isabs(newdir):
                newdir = os.path.abspath(newdir)
            self._remote.cd(newdir)
        except Exception as e:
            raise RuntimeError("{} newdir init {} newdir {} ".format(str(e),newdir_init, newdir))



    def predict_on_batch(self, x):
        return self._remote.predict_on_batch(x)

    def _populate_model(self, **kwargs):
        self._remote._populate_model(**kwargs)

    @property
    def pipeline(self):
        return self._remote.get_pipeline()


    def _init_pipeline(self, default_dataloader, source, model, cwd):
        self._remote._init_pipeline(default_dataloader, source, model, cwd)


    def stop_server(self):
        self.server.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_server()



class RemoteKerasModel(RemoteModel, RemoteGradientMixin, RemoteLayerActivationMixin):
    def __init__(self, server_settings, weights, arch=None, custom_objects=None, backend=None, image_dim_ordering=None, cwd=None):
        model_args = ModelArgs(weights=weights, arch=arch, custom_objects=custom_objects, backend=backend, image_dim_ordering=image_dim_ordering)
        super().__init__(cwd=cwd,server_settings=server_settings, model_type="keras", model_args=model_args)


    def get_layers_and_outputs(self, layer=None, use_final_layer=False, pre_nonlinearity=False):
        return  self._remote.get_layers_and_outputs( layer=layer, use_final_layer=use_final_layer, pre_nonlinearity=pre_nonlinearity)


    @property
    def keras_backend(self):
        return self._remote.get_keras_backend()
    
    @property
    def model(self):
        return self._remote.get_model()
    

    

class RemotePyTorchModel(RemoteModel, RemoteGradientMixin, RemoteLayerActivationMixin):
    def __init__(self, server_settings, weights, module_class=None, module_kwargs=None, module_obj=None, module_file=None,
                 auto_use_cuda=True, cwd=None):
        model_args = ModelArgs(weights=weights, module_class=module_class, module_kwargs=module_kwargs, module_obj=module_obj, module_file=module_file,
                 auto_use_cuda=auto_use_cuda)
        super().__init__(cwd=cwd,server_settings=server_settings, model_type="pytorch", model_args=model_args)


class RemoteOldPyTorchModel(RemoteModel, RemoteGradientMixin, RemoteLayerActivationMixin):
    def __init__(self, server_settings,file=None, build_fn=None, weights=None, auto_use_cuda=True, cwd=None):
        model_args = ModelArgs(file=file, build_fn=build_fn, weights=weights, auto_use_cuda=auto_use_cuda)
        super().__init__(cwd=cwd,server_settings=server_settings, model_type="old_pytorch", model_args=model_args)


class RemoteTensorFlowModel(RemoteModel, RemoteGradientMixin, RemoteLayerActivationMixin):
    def __init__(self, server_settings, input_nodes, target_nodes, checkpoint_path, const_feed_dict_pkl=None, cwd=None):
        model_args = ModelArgs(input_nodes=input_nodes, target_nodes=target_nodes, checkpoint_path=checkpoint_path, const_feed_dict_pkl=const_feed_dict_pkl)
        super().__init__(cwd=cwd,server_settings=server_settings, model_type="tensorflow", model_args=model_args)


class RemoteSklearnModel(RemoteModel):
    def __init__(self, server_settings, pkl_file, predict_method="predict",cwd=None):
        model_args = ModelArgs(pkl_file=pkl_file, predict_method=predict_method)
        super().__init__(cwd=cwd,server_settings=server_settings, model_type="sklearn", model_args=model_args)

# the custom model may or may not implement 
# input_grad and predict_activation_on_batch
# therefore it needs the mixin
class RemoteCustomModel(RemoteModel, RemoteGradientMixin, RemoteLayerActivationMixin):
    def __init__(self, server_settings,cwd=None, **kwargs):
        model_args = ModelArgs(**kwargs)
        super().__init__(cwd=cwd,server_settings=server_settings, model_type="custom", model_args=model_args)

AVAILABLE_REMOTE_MODELS = OrderedDict([
                                ("keras", RemoteKerasModel),
                                ("pytorch", RemotePyTorchModel),
                                ("sklearn", RemoteSklearnModel),
                                ("tensorflow", RemoteTensorFlowModel),
                                ("custom", RemoteCustomModel)
                                ])