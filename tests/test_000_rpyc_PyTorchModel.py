import numpy as np
from kipoi.model import PyTorchModel, OldPyTorchModel, infer_pyt_class
from kipoi.rpyc_model import *


from kipoi_utils.utils import cd
import torch
from torch import nn
import torch.nn.functional as F
import kipoi
import pytest
import os

from utils import *

DUMMY_MODEL_WEIGHTS_FILE = "tests/data/pyt_dummy_model_weight.pth"
PYT_SEQUENTIAL_MODEL_WEIGHTS_FILE = "tests/data/pyt_sequential_model_weights.pth"
PYT_NET_MODEL_WEIGHTS_FILE = "tests/data/pyt_net_model_weights.pth"
PYT_SUMMY_MULTI_I_MODEL_WEIGHTS_FILE = "tests/data/pyt_dummy_multi_i_model_weight.pth"
CHECKING_MODEL_WEIGHTS_FILE = "tests/data/pyt_checking_model_weight.pth"
THISFILE = "tests/test_000_rpyc_PyTorchModel.py"



# Todo - test the kwargs argument
# todo - self-refer to this file to load the model classes

def check_same_weights(dict1, dict2):
    for k in dict1:
        if dict1[k].is_cuda:
            vals1 = dict1[k].cpu().numpy()
        else:
            vals1 = dict1[k].numpy()
        if dict2[k].is_cuda:
            vals2 = dict2[k].cpu().numpy()
        else:
            vals2 = dict2[k].numpy()
        assert np.all(vals1 == vals2)


D_in, H, D_out = 1000, 100, 1
SimpleModel = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid(),
)

def get_simple_model():
    return SimpleModel

class CheckingModel(torch.nn.Module):
    original_input = None
    def __init__(self):
        super(CheckingModel, self).__init__()
    #
    def forward(self, *args, **kwargs):
        if (len(args) != 0) and (len(kwargs) != 0):
            raise Exception("Mix of positional and keyword inputs should not happen!")
        if len(args) != 0:
            if isinstance(self.original_input, np.ndarray):
                assert all([np.all(get_np(el) == self.original_input) for el in args])
            else:
                assert all([np.all(get_np(el) == el2) for el, el2 in zip(args, self.original_input)])
            return args
        #
        if len(kwargs) != 0:
            assert set(kwargs.keys()) == set(self.original_input.keys())
            for k in self.original_input:
                assert np.all(get_np(kwargs[k]) == self.original_input[k])
            # at the moment (pytorch 0.2.0) pytorch doesn't support dictionary outputs from models
            return [kwargs[k] for k in sorted(list(kwargs))]

checking_model = CheckingModel().double()

def store_checking_model_weights(fname=CHECKING_MODEL_WEIGHTS_FILE):
    torch.save(checking_model.state_dict(), fname)

def get_np(var):
    if var.is_cuda:
        return var.cpu().data.numpy()
    else:
        return var.data.numpy()


@pytest.mark.flaky(max_runs=20)
class TestPyTorchRpyc(object):



    @pytest.mark.parametrize("port",  [2000, 2010])
    def test_loading_a(self, port):
        

        s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=True, address='localhost', port=port, logging_level=0)
        model_path = "example/models/pyt/model_files/"
        model_path_class_model = "example/models/pyt_class/"
        with cd(model_path):
            with RemotePyTorchModel(s, module_obj="pyt.simple_model", weights="only_weights.pth") as model:
                pass



    @pytest.mark.parametrize("port",  [2020, 2030])
    def test_loading_b(self, port):

        s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=True, address='localhost', port=port, logging_level=0)
        model_path = "example/models/pyt/model_files/"
        model_path_class_model = "example/models/pyt_class/"


        with RemotePyTorchModel(s,module_file=model_path + "pyt.py", weights=model_path + "only_weights.pth", module_obj="simple_model") as m1:
            pass


    @pytest.mark.parametrize("port",  [2040, 2050])
    def test_loading_c(self, port):
       

        s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=True, address='localhost', port=port, logging_level=0)

        model_path = "example/models/pyt/model_files/"
        model_path_class_model = "example/models/pyt_class/"


        with RemotePyTorchModel(s,module_file=THISFILE, weights=PYT_NET_MODEL_WEIGHTS_FILE, module_class="PyTNet") as m1:
            pass

    @pytest.mark.parametrize("port",  [2060, 2070])
    def test_loading_c(self, port):

        s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=True, address='localhost', port=port, logging_level=0)




        model_path = "example/models/pyt/model_files/"
        model_path_class_model = "example/models/pyt_class/"

        with RemotePyTorchModel(s,module_file=THISFILE, weights=PYT_NET_MODEL_WEIGHTS_FILE, module_class="PyTNet", module_kwargs={}) as m1:
            pass


    @pytest.mark.parametrize("port",  [2080, 2090])
    def test_loading_e(self, port):

       
        s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=True, address='localhost', port=port, logging_level=0)

        model_path = "example/models/pyt/model_files/"
        model_path_class_model = "example/models/pyt_class/"

        with RemotePyTorchModel(s,module_file=THISFILE, weights=PYT_NET_MODEL_WEIGHTS_FILE, module_class="PyTNet", module_kwargs="{}") as m1:
            pass


    @pytest.mark.parametrize("port",  [2100, 2110])
    def test_loading_f(self, port):


        s = kipoi.rpyc_model.ServerArgs(env_name=None, use_current_python=True, address='localhost', port=port, logging_level=0)




        model_path = "example/models/pyt/model_files/"
        model_path_class_model = "example/models/pyt_class/"


        
        with kipoi.get_model(model_path_class_model, "dir", server_settings=s) as mh:
            # Load the test files from model source
            mh.pipeline.predict_example(batch_size=3)



  
    @pytest.mark.parametrize("port",  [2120, 2130])
    def test_loading_g(self, port):

        model_path = "example/models/pyt/model_files/"
        model_path_class_model = "example/models/pyt_class/"
        
        # ensure we got the env
        env_name = create_env_if_not_exist(bypass=False , model=model_path_class_model, source='dir')
        s = kipoi.rpyc_model.ServerArgs(env_name=env_name, use_current_python=False, address='localhost', port=port, logging_level=0)
        with kipoi.get_model(model_path_class_model, "dir", server_settings=s) as mh:
            # Load the test files from model source
            mh.pipeline.predict_example(batch_size=3) 




class PyTNet(nn.Module):
    def __init__(self):
        super(PyTNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(240, 2)
        self.fc2 = nn.Linear(2, 1)
    def forward(self, x):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(F.sigmoid(self.conv1(x)))
        x1 = x1.view(-1, 240)
        x2 = x2.view(-1, 240)
        x = torch.cat((x1, x2), 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

