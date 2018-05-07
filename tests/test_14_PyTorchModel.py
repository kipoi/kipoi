import numpy as np
from kipoi.model import PyTorchModel
from kipoi.utils import cd
import torch
from torch import nn
import torch.nn.functional as F
import kipoi
import pytest


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


def get_simple_model():
    import torch
    D_in, H, D_out = 1000, 100, 1
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid(),
    )
    return model


def get_np(var):
    if var.is_cuda:
        return var.cpu().data.numpy()
    else:
        return var.data.numpy()


# Test the loading of models
def test_loading(tmpdir):
    import torch
    # load model in different ways...
    with pytest.raises(Exception):
        PyTorchModel()
    PyTorchModel(build_fn=lambda: get_simple_model())
    model_path = "examples/pyt/model_files/"
    # load model and weights explcitly
    m1 = PyTorchModel(file=model_path + "pyt.py", weights=model_path + "only_weights.pth", build_fn="get_model")
    # load model and weights through model loader
    with cd("examples/pyt"):
        m2 = PyTorchModel(file="model_files/pyt.py", build_fn="get_model_w_weights")
    # assert that's identical
    check_same_weights(m1.model.state_dict(), m2.model.state_dict())
    # now test whether loading a full model works
    tmpfile = str(tmpdir.mkdir("pytorch").join("full_model.pth"))
    m = get_simple_model()
    torch.save(m, tmpfile)
    km = PyTorchModel(weights=tmpfile)
    check_same_weights(m.state_dict(), km.model.state_dict())


# Test the input and prediction transformation
def test_prediction_io():
    import torch

    class checking_model(torch.nn.Module):

        def __init__(self, original_input):
            super(checking_model, self).__init__()
            self.original_input = original_input
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

    predict_inputs = {"arr": np.random.randn(1000, 20, 1, 4)}
    predict_inputs["list"] = [predict_inputs["arr"]] * 3
    # at the moment (pytorch 0.2.0) pytorch doesn't support dictionary outputs from models
    predict_inputs["dict"] = {"in%d" % i: predict_inputs["arr"] for i in range(10)}
    for k in predict_inputs:
        m_in = predict_inputs[k]
        m = PyTorchModel(build_fn=lambda: checking_model(m_in))
        pred = m.predict_on_batch(m_in)
        if isinstance(m_in, np.ndarray):
            assert np.all(pred == m_in)
        elif isinstance(m_in, list):
            assert all([np.all(el == el2) for el, el2 in zip(pred, m_in)])
        elif isinstance(m_in, dict):
            m_expected = [m_in[k] for k in sorted(list(m_in))]
            assert all([np.all(el == el2) for el, el2 in zip(pred, m_expected)])


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


def get_pyt_sequential_model_input():
    np.random.seed(1)
    return np.random.rand(3, 1, 10)


def pyt_sequential_model_bf():
    new_model = nn.Sequential(
        nn.Conv1d(1, 16, kernel_size=5, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Linear(5, 24),
        nn.ReLU()
    ).double()
    # new_model.load_state_dict(new_model.state_dict())
    return new_model


def get_dummy_model_input():
    np.random.seed(1)
    return np.random.rand(20, 1)


def dummy_model_bf():
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


def get_pyt_complex_model_input():
    np.random.seed(1)
    return np.random.rand(3, 1, 10)


def pyt_complex_model_bf():
    return PyTNet().double()


def test_get_layer():
    dummy_model = kipoi.model.PyTorchModel(build_fn=dummy_model_bf)
    sequential_model = kipoi.model.PyTorchModel(build_fn=pyt_sequential_model_bf)
    complex_model = kipoi.model.PyTorchModel(build_fn=pyt_complex_model_bf)
    # test get layer
    assert dummy_model.get_layer("first") == dummy_model.model.first
    assert sequential_model.get_layer("0") == getattr(sequential_model.model, "0")
    assert complex_model.get_layer("fc1") == complex_model.model.fc1


def test_predict_activation_on_batch():
    dummy_model = kipoi.model.PyTorchModel(build_fn=dummy_model_bf)
    complex_model = kipoi.model.PyTorchModel(build_fn=pyt_complex_model_bf)
    acts_dummy = dummy_model.predict_activation_on_batch(get_dummy_model_input(), layer="first")

    acts = complex_model.predict_activation_on_batch(get_pyt_complex_model_input(), layer="conv1")
    assert isinstance(acts, list)
    assert isinstance(acts[0], list)
    assert isinstance(acts[0][0], np.ndarray)
    with pytest.raises(Exception):
        # This has to raise an exception - pre_nonlinearity not implemented
        acts = dummy_model.predict_activation_on_batch(get_dummy_model_input(),
                                                       layer="final", pre_nonlinearity=True)
