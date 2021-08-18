import numpy as np
import pandas as pd
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


def test_predict_to_file_with_metadata_hdf5(tmpdir):
    h5_tmpfile = str(tmpdir.mkdir("example").join("out.h5"))
    model = kipoi.get_model("Basset", source="kipoi")
    dl_kwargs = model.default_dataloader.example_kwargs
    with cd(model.source_dir):
        model.pipeline.predict_to_file(h5_tmpfile, dl_kwargs,keep_metadata=True)
    preds_and_metadata = kipoi.readers.HDF5Reader.load(h5_tmpfile)
    assert 'preds' in preds_and_metadata
    assert 'metadata' in preds_and_metadata
    assert len(preds_and_metadata['metadata']['ranges']['chr']) == 10
    assert len(preds_and_metadata['preds']) == 10

def test_predict_to_file_with_metadata_tsv(tmpdir):
    tsv_tmpfile_metadata = str(tmpdir.mkdir("example").join("out_with_metadata.tsv"))
    model = kipoi.get_model("Basset", source="kipoi")
    dl_kwargs = model.default_dataloader.example_kwargs
    with cd(model.source_dir):
        model.pipeline.predict_to_file(tsv_tmpfile_metadata, dl_kwargs,keep_metadata=True)
    preds_and_metadata = pd.read_csv(tsv_tmpfile_metadata, sep='\t')
    assert 'metadata/ranges/chr' in preds_and_metadata.columns 
    assert 'preds/100' in preds_and_metadata.columns
    assert len(preds_and_metadata['metadata/ranges/chr']) == 10
    assert len(preds_and_metadata['preds/100']) == 10
    assert preds_and_metadata.at[0,'preds/100'] == pytest.approx(0.4168229, rel=1e-05)


def test_predict_to_file_without_metadata_tsv(tmpdir):
    tsv_tmpfile = str(tmpdir.mkdir("example").join("out.tsv"))
    model = kipoi.get_model("Basset", source="kipoi")
    dl_kwargs = model.default_dataloader.example_kwargs
    with cd(model.source_dir):
        model.pipeline.predict_to_file(tsv_tmpfile, dl_kwargs)
    preds = pd.read_csv(tsv_tmpfile, sep='\t')
    assert 'metadata/ranges/chr' not in preds.columns 
    assert 'preds/100' in preds.columns
    assert preds.at[0,'preds/100'] == pytest.approx(0.4168229, rel=1e-05)
