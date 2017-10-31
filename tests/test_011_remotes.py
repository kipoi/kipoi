"""Test the Kipoi remotes
"""
import kipoi
import os
import pandas as pd


# TODO - add installation dependencies?

def test_load_models_kipoi():
    k = kipoi.config.get_source("kipoi")

    l = k.list_models()  # all the available models

    assert "extended_coda" in list(l.model)
    model = "extended_coda"
    mpath = k.pull_model(model)
    m_dir = os.path.dirname(mpath)

    # load the model
    kipoi.get_model(m_dir, source="dir")

    kipoi.get_model(model, source="kipoi")
    kipoi.get_dataloader_factory(model)


def test_load_models_local():
    model = "examples/extended_coda"
    kipoi.get_model(model, source="dir")
    kipoi.get_dataloader_factory(model, source="dir")


def test_list_models():
    k = kipoi.config.get_source("kipoi")

    df = k.list_models()
    assert isinstance(df, pd.DataFrame)

    # column names
    df_model_columns = ['model', 'name', 'version', 'author', 'descr', 'type', 'inputs', 'targets', 'tags']
    assert df_model_columns == list(df.columns)

    #
    df_all = kipoi.list_models()
    assert ["source"] + df_model_columns == list(df_all.columns)

    kipoi.get_model_info("extended_coda")

    kipoi.get_model_info("extended_coda", source="kipoi")

    # local files
    kipoi.get_model_info("examples/extended_coda", source="dir")
