import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
import sys
import json
import os
import signal
import subprocess
import sched,time
from subprocess import Popen, PIPE, STDOUT, check_output
import atexit
import numpy
from . base_model import BaseModel
from kipoi_conda import get_envs, get_env_path

from kipoi_utils import (load_module, cd, merge_dicts, read_pickle, override_default_kwargs,
                    load_obj, inherits_from, infer_parent_class, makedir_exist_ok)

import logging




def call_script_in_env(filename, env_name=None, args=None, cwd=None): 
    """run an python script in a certain conda enviroment in a background
    process.
    
    Args:
        filename (str): path to the python script
        env_name (str or None, optional): Name of the conda enviroment.
        If this is None, the current python executable will be used
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

    if env_name is None:
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
    def __init__(self,env_name, address, port, connection_timeout=1.0, logging_level=None, cwd=None):
        super(ServerArgs, self).__init__()


        self['address'] = str(address)
        self['port'] = int(port)
        self['env_name'] = env_name
        self['connection_timeout'] = float(connection_timeout)
        self['logging_level'] = logging_level
        if logging_level is None:
            self['logging_level']= logging.getLogger().level
        self['cwd'] = cwd
        


    def setEnv(self, env_name):
        self['env_name'] = str(env_name)




class RpycServer(object):

    def __init__(self, env_name, address, port, connection_timeout=1.0, logging_level=None, cwd=None):
        
        self.address = str(address)
        self.port = int(port)
        self.env_name = env_name
        self.connection_timeout = float(connection_timeout)
        # self.use_current_python = env_name is None
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
            '--logging_level', 
            str(self.logging_level)
        ]
        print('cwd',self.cwd)
        self.server_process =  call_script_in_env(env_name=self.env_name, 
            filename=server_script_path, args=args, cwd=self.cwd)

        self.connection = self.__get_connection()


    def stop(self):
        if self.server_process is None:
            raise RuntimeError("server process is not running")

        os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)  


    def __get_connection(self):
        t0 = time.clock()

        while True:
            try:
                c = rpyc.connect(self.address, self.port)
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



class RemoteModel(BaseModel):
    def __init__(self, server_settings, model_type, model_args, cwd=None):
        super(RemoteModel, self).__init__()
        self.server = RpycServer(**server_settings)
        self.server.start()
        self._c = self.server.connection
        self._model_type = model_type
        self._model_args = model_args

        self.cwd = cwd
        if cwd is None:
            self.cwd = os.getcwd()

        # cd to the right dir
        self._c.root.cd(self.cwd)

        # specify model type
        self._c.root._set_model_type(model_type=self._model_type)

        # initalize the model 
        self._c.root._initialize(*self._model_args.args, **self._model_args.kwargs)


    def predict_on_batch(self, x):
        return self._c.root.predict_on_batch(x)

    def input_grad(self, x, filter_idx=None, avg_func=None, layer=None, final_layer=True,
                   selected_fwd_node=None, pre_nonlinearity=False):
        return self._c.root.input_grad(x=x, filter_idx=filter_idx, avg_func=avg_func,
                   layer=layer, final_layer=final_layer,selected_fwd_node=selected_fwd_node, 
                   pre_nonlinearity=pre_nonlinearity)

    def predict_activation_on_batch(self, x, layer, pre_nonlinearity=False):
        return self._c.root.predict_activation_on_batch(x=x, layer=layer, 
                   pre_nonlinearity=pre_nonlinearity)

    def stop_server(self):
        self.server.stop()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_server()


