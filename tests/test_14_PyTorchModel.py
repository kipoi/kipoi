import torch
import numpy as np
from kipoi.model import PyTorchModel
import pytest
from kipoi.utils import cd

def check_same_weights(dict1, dict2):
    for k in dict1:
        assert np.all(dict1[k].numpy() == dict2[k].numpy())


def get_simple_model():
    D_in, H, D_out = 1000, 100, 1
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid(),
    )
    return model

class checking_model(torch.nn.Module):
    def __init__(self, original_input):
        super(checking_model, self).__init__()
        self.original_input = original_input
    #
    def forward(self, *args, **kwargs):
        if (len(args) != 0) and (len(kwargs) !=0):
            raise Exception("Mix of positional and keyword inputs should not happen!")
        if len(args) !=0:
            if isinstance(self.original_input, np.ndarray):
                assert all([np.all(el.data.numpy() == self.original_input) for el in args])
            else:
                assert all([np.all(el.data.numpy() == el2) for el, el2 in zip(args, self.original_input)])
            return args
        #
        if len(kwargs) !=0:
            assert set(kwargs.keys()) == set(self.original_input.keys())
            for k in self.original_input:
                assert np.all(kwargs[k].data.numpy() == self.original_input[k])
            # at the moment (pytorch 0.2.0) pytorch doesn't support dictionary outputs from models
            return [kwargs[k] for k in sorted(list(kwargs))]


# Test the loading of models
def test_loading(tmpdir):
    # load model in different ways...
    with pytest.raises(Exception):
        PyTorchModel()
    PyTorchModel(gen_fn=lambda: get_simple_model())
    model_path = "examples/pyt/model_files/"
    # load model and weights explcitly
    m1 = PyTorchModel(weights=model_path + "only_weights.pth", gen_fn=model_path + "pyt.py::get_model")
    # load model and weights through model loader
    with cd("examples/pyt"):
        m2 = PyTorchModel(gen_fn="model_files/pyt.py::get_model_w_weights")
    # assert that's identical
    check_same_weights(m1.model.state_dict(), m2.model.state_dict())
    #now test whether loading a full model works
    tmpfile = str(tmpdir.mkdir("pytorch").join("full_model.pth"))
    m = get_simple_model()
    torch.save(m, tmpfile)
    km = PyTorchModel(weights=tmpfile)
    check_same_weights(m.state_dict(), km.model.state_dict())


# Test the input and prediction transformation
def test_prediction_io():
    predict_inputs = {"arr": np.random.randn(1000,20,1,4)}
    predict_inputs["list"] = [predict_inputs["arr"]] * 3
    # at the moment (pytorch 0.2.0) pytorch doesn't support dictionary outputs from models
    predict_inputs["dict"] = {"in%d"%i:predict_inputs["arr"] for i in range(10)}
    for k in predict_inputs:
        m_in = predict_inputs[k]
        m = PyTorchModel(gen_fn = lambda : checking_model(m_in))
        pred = m.predict_on_batch(m_in)
        if isinstance(m_in, np.ndarray):
            assert np.all(pred == m_in)
        elif isinstance(m_in, list):
            assert all([np.all(el == el2) for el, el2 in zip(pred, m_in)])
        elif isinstance(m_in, dict):
            m_expected = [m_in[k] for k in sorted(list(m_in))]
            assert all([np.all(el == el2) for el, el2 in zip(pred, m_expected)])

