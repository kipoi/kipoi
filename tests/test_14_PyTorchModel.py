import numpy as np
from kipoi.model import PyTorchModel
from kipoi.utils import cd
import torch
from torch import nn
import torch.nn.functional as F
import kipoi
import pytest
DUMMY_MODEL_WEIGHTS_FILE = "tests/data/pyt_dummy_model_weight.pth"
PYT_SEQUENTIAL_MODEL_WEIGHTS_FILE = "tests/data/pyt_sequential_model_weights.pth"
PYT_NET_MODEL_WEIGHTS_FILE = "tests/data/pyt_net_model_weights.pth"
PYT_SUMMY_MULTI_I_MODEL_WEIGHTS_FILE = "tests/data/pyt_dummy_multi_i_model_weight.pth"
CHECKING_MODEL_WEIGHTS_FILE = "tests/data/pyt_checking_model_weight.pth"
THISFILE = "tests/test_14_PyTorchModel.py"


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


# Test the loading of models
def test_loading():
    model_path = "example/models/pyt/model_files/"
    # load model and weights explcitly
    m1 = PyTorchModel(file=model_path + "pyt.py", weights=model_path + "only_weights.pth", model="simple_model")


# Test the input and prediction transformation
def test_prediction_io():
    predict_inputs = {"arr": np.random.randn(1000, 20, 1, 4)}
    predict_inputs["list"] = [predict_inputs["arr"]] * 3
    # at the moment (pytorch 0.2.0) pytorch doesn't support dictionary outputs from models
    predict_inputs["dict"] = {"in%d" % i: predict_inputs["arr"] for i in range(10)}
    for k in predict_inputs:
        m_in = predict_inputs[k]
        m = PyTorchModel(model="checking_model", file=THISFILE, weights=CHECKING_MODEL_WEIGHTS_FILE)
        m.model.original_input = m_in
        pred = m.predict_on_batch(m_in)
        if isinstance(m_in, np.ndarray):
            assert np.all(pred == m_in)
        elif isinstance(m_in, list):
            assert all([np.all(el == el2) for el, el2 in zip(pred, m_in)])
        elif isinstance(m_in, dict):
            m_expected = [m_in[k] for k in sorted(list(m_in))]
            assert all([np.all(el == el2) for el, el2 in zip(pred, m_expected)])



"""
class PyTConvNet(torch.nn.Module):
    # Taken from https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/5_convolutional_net.py

    def __init__(self, output_dim):
        super(PyTConvNet, self).__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(1, 10, kernel_size=5))
        self.conv.add_module("maxpool_1", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_1", torch.nn.ReLU())
        self.conv.add_module("conv_2", torch.nn.Conv2d(10, 20, kernel_size=5))
        self.conv.add_module("dropout_2", torch.nn.Dropout())
        self.conv.add_module("maxpool_2", torch.nn.MaxPool2d(kernel_size=2))
        self.conv.add_module("relu_2", torch.nn.ReLU())
        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(320, 50))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout())
        self.fc.add_module("fc2", torch.nn.Linear(50, output_dim))

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(-1, 320)
        return self.fc.forward(x)
"""


def get_pyt_sequential_model_input():
    np.random.seed(1)
    return np.random.rand(3, 1, 10)


pyt_sequential_model = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=5, padding=2),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Linear(5, 24),
        nn.ReLU()
    ).double()

def store_pyt_sequential_model_weights(fname=PYT_SEQUENTIAL_MODEL_WEIGHTS_FILE):
    torch.save(pyt_sequential_model.state_dict(), fname)

def get_dummy_model_input():
    np.random.seed(1)
    return np.random.rand(20, 1)

# Test layer activation
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.first = nn.Linear(1, 1)
        self.second = nn.Linear(1, 1)
        self.final = nn.Sigmoid()
    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        x = self.final(x)
        return x

dummy_model = DummyModel().double()

def dummy_model_bf():
    from torch import nn
    import torch
    dummy_model = DummyModel().double()
    init_state = dummy_model.state_dict()
    wt_1 = torch.FloatTensor(1, 1)
    wt_1[:] = 0.5
    b_1 = torch.FloatTensor(1)
    b_1[:] = 0.0
    wt_2 = torch.FloatTensor(1, 1)
    wt_2[:] = 0.25
    b_2 = torch.FloatTensor(1)
    b_2[:] = 0.0
    init_state["first.weight"] = wt_1
    init_state["second.weight"] = wt_2
    init_state["first.bias"] = b_1
    init_state["second.bias"] = b_2
    dummy_model.load_state_dict(init_state)
    dummy_model.eval()
    return dummy_model

def store_dummy_model_weights(fname=DUMMY_MODEL_WEIGHTS_FILE):
    dummy_model = dummy_model_bf()
    torch.save(dummy_model.state_dict(), fname)


def get_dummy_multi_input(kind="list"):
    np.random.seed(1)
    if kind == "list":
        return [np.random.rand(20, 1), np.random.rand(20, 1)]
    elif kind == "dict":
        return {"FirstInput": np.random.rand(20, 1), "SecondInput": np.random.rand(20, 1)}


class DummyMultiInput(nn.Module):
    def __init__(self):
        super(DummyMultiInput, self).__init__()
        self.first = nn.Linear(1, 1)
        self.second = nn.Linear(1, 1)
        self.final = nn.Sigmoid()
    def forward(self, FirstInput, SecondInput):
        x_1 = self.first(FirstInput)
        x_2 = self.first(SecondInput)
        x = self.second(torch.cat((x_1, x_2), 0))
        x = self.final(x)
        return x

dummy_multi_input_model = DummyMultiInput().double()

def store_dummy_multi_input_weights(fname=PYT_SUMMY_MULTI_I_MODEL_WEIGHTS_FILE):
    torch.save(dummy_multi_input_model.state_dict(), fname)


def get_pyt_complex_model_input():
    np.random.seed(1)
    return np.random.rand(3, 1, 10)


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

pyt_net = PyTNet().double()

def store_pyt_net_weights(fname=PYT_NET_MODEL_WEIGHTS_FILE):
    torch.save(pyt_net.state_dict(), fname)


def test_get_layer():
    dummy_model = kipoi.model.PyTorchModel(model="dummy_model", file = THISFILE, weights=DUMMY_MODEL_WEIGHTS_FILE)
    sequential_model = kipoi.model.PyTorchModel(model="pyt_sequential_model", file = THISFILE,
                                                weights = PYT_SEQUENTIAL_MODEL_WEIGHTS_FILE)
    complex_model = kipoi.model.PyTorchModel(model="pyt_net", file = THISFILE, weights=PYT_NET_MODEL_WEIGHTS_FILE)
    # test get layer
    assert dummy_model.get_layer("first") == dummy_model.model.first
    assert sequential_model.get_layer("0") == getattr(sequential_model.model, "0")
    assert complex_model.get_layer("fc1") == complex_model.model.fc1


def test_predict_activation_on_batch():
    dummy_model = kipoi.model.PyTorchModel(model="dummy_model", file = THISFILE, weights=DUMMY_MODEL_WEIGHTS_FILE)
    complex_model = kipoi.model.PyTorchModel(model="pyt_net", file = THISFILE, weights=PYT_NET_MODEL_WEIGHTS_FILE)
    acts_dummy = dummy_model.predict_activation_on_batch(get_dummy_model_input(), layer="first")

    acts = complex_model.predict_activation_on_batch(get_pyt_complex_model_input(), layer="conv1")
    assert isinstance(acts, list)
    assert isinstance(acts[0], list)
    assert isinstance(acts[0][0], np.ndarray)
    with pytest.raises(Exception):
        # This has to raise an exception - pre_nonlinearity not implemented
        acts = dummy_model.predict_activation_on_batch(get_dummy_model_input(),
                                                       layer="final", pre_nonlinearity=True)


def test_gradients():
    import kipoi
    dummy_model = kipoi.model.PyTorchModel(model="dummy_model", file = THISFILE, weights=DUMMY_MODEL_WEIGHTS_FILE)
    assert dummy_model.input_grad(np.array([[1.0]]), avg_func="sum", layer="second")[0][0] == 0.125
    complex_model = kipoi.model.PyTorchModel(model="pyt_net", file = THISFILE, weights=PYT_NET_MODEL_WEIGHTS_FILE)

    gT2 = complex_model.input_grad(get_pyt_complex_model_input(), avg_func="sum", layer="conv1",
                                   selected_fwd_node=None)
    gF2 = complex_model.input_grad(get_pyt_complex_model_input(), avg_func="sum", layer="conv1",
                                   selected_fwd_node=1)

    assert np.all(gF2 * 2 == gT2)


def test_returned_gradient_fmt():
    import kipoi
    multi_input_model = kipoi.model.PyTorchModel(model="dummy_multi_input_model", file=THISFILE,
                                                 weights=PYT_SUMMY_MULTI_I_MODEL_WEIGHTS_FILE)
    # try first whether the prediction actually works..
    sample_input = get_dummy_multi_input("list")
    grad_out = multi_input_model.input_grad(sample_input, avg_func="sum", layer="second",
                                            selected_fwd_node=None)
    assert isinstance(grad_out, type(sample_input))
    assert len(grad_out) == len(sample_input)
    sample_input = get_dummy_multi_input("dict")
    grad_out = multi_input_model.input_grad(sample_input, avg_func="sum", layer="second",
                                            selected_fwd_node=None)
    assert isinstance(grad_out, type(sample_input))
    assert len(grad_out) == len(sample_input)
    assert all([k in grad_out for k in sample_input])


def test_gradients_functions():
    import kipoi
    multi_input_model = kipoi.model.PyTorchModel(model="dummy_multi_input_model", file=THISFILE,
                                                 weights=PYT_SUMMY_MULTI_I_MODEL_WEIGHTS_FILE)
    sample_input = get_dummy_multi_input("dict")
    multi_input_model.input_grad(sample_input, avg_func="max", layer="first", selected_fwd_node=None)


class DummySlice():
    def __getitem__(self, key):
        return key


def test_grad_tens_generation():
    model = kipoi.model.PyTorchModel(model="pyt_sequential_model", file = THISFILE,
                                     weights=PYT_SEQUENTIAL_MODEL_WEIGHTS_FILE, auto_use_cuda=True)
    fwd_hook_obj, removable_hook_obj = model._register_fwd_hook(model.get_layer("4"))
    fwd_values, x_in = model.np_run_pred(get_pyt_sequential_model_input(), requires_grad=True)
    removable_hook_obj.remove()

    assert np.all(model.get_grad_tens(fwd_values, DummySlice()[:, 0:3, :], "sum").cpu().numpy()[0, ...] == np.array(
        [[1] * 24] * 3 + [[0] * 24] * 13))
    assert np.all(model.get_grad_tens(fwd_values, DummySlice()[:, 0:3, 0:2], "sum").cpu().numpy()[0, ...] == np.array(
        [[1] * 2 + [0] * 22] * 3 + [[0] * 24] * 13))
    assert np.all(model.get_grad_tens(fwd_values, DummySlice()[0:3, :], "sum").cpu().numpy()[0, ...] == np.array(
        [[1] * 24] * 3 + [[0] * 24] * 13))
    assert np.all(model.get_grad_tens(fwd_values, DummySlice()[0:3, 0:2], "sum").cpu().numpy()[0, ...] == np.array(
        [[1] * 2 + [0] * 22] * 3 + [[0] * 24] * 13))
    # Filter is 2D
    with pytest.raises(Exception):
        model.get_grad_tens(fwd_values, DummySlice()[0:2], "max")
