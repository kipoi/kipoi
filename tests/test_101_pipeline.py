import numpy as np
from kipoi_utils.utils import cd
import kipoi
import pytest


def test_gradient_pipeline():
    model = kipoi.get_model("Basset", source="kipoi")
    dl_kwargs = model.default_dataloader.example_kwargs
    with cd(model.source_dir):
        ret = model.pipeline.input_grad(dl_kwargs, final_layer=True, avg_func="sum")
    assert all(k in ret for k in ['targets', 'metadata', 'inputs', 'grads'])


def test_predict_pipeline():
    model = kipoi.get_model("Basset", source="kipoi")
    dl_kwargs = model.default_dataloader.example_kwargs
    with cd(model.source_dir):
        ret = model.pipeline.predict(dl_kwargs)
    assert isinstance(ret, np.ndarray)
    with cd(model.source_dir):
        ret = model.pipeline.predict(dl_kwargs, layer="11")
    assert isinstance(ret, list)
    # with a model that does not implement LayerActivationMixin it should fail:
    hal_model = kipoi.get_model("HAL", source="kipoi")
    hal_dl_kwargs = hal_model.default_dataloader.example_kwargs
    with pytest.raises(Exception):
        ret = model.pipeline.predict(hal_dl_kwargs, layer="11")


def test_predict_to_file(tmpdir):
    h5_tmpfile = str(tmpdir.mkdir("example").join("out.h5"))
    model = kipoi.get_model("Basset", source="kipoi")
    dl_kwargs = model.default_dataloader.example_kwargs
    with cd(model.source_dir):
        model.pipeline.predict_to_file(h5_tmpfile, dl_kwargs)
    preds = kipoi.readers.HDF5Reader.load(h5_tmpfile)
    assert 'preds' in preds
