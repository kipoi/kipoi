import numpy as np

import kipoi
from kipoi_utils.utils import cd

def test_model():
    example_dir = "example/models/mdcexample"
    model = kipoi.get_model(example_dir, source="dir")
    # model = kipoi.get_model("APARENT/site_probabilities", source="kipoi")

def test_model_predict_example():
    example_dir = "example/models/mdcexample"
    model = kipoi.get_model(example_dir, source="dir")
    # model = kipoi.get_model("APARENT/site_probabilities", source="kipoi")
    pred = model.pipeline.predict_example(batch_size=32)


def test_model_predict():
    example_dir = "example/models/mdcexample"
    model = kipoi.get_model(example_dir, source="dir")
    # model = kipoi.get_model("APARENT/site_probabilities", source="kipoi")
    dl_kwargs = model.default_dataloader.example_kwargs
    with cd(model.source_dir):
        ret = model.pipeline.predict(dl_kwargs)
    assert isinstance(ret, dict)
    assert list(ret.keys()) == ["distal_prop", "site_props"]
    assert isinstance(ret["distal_prop"], np.ndarray)
    assert isinstance(ret["site_props"], np.ndarray)
    assert ret["distal_prop"].shape == (618, )
    assert ret["site_props"].shape == (618, 205)