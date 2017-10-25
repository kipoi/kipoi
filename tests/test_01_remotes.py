"""Test the Kipoi remotes
"""
import kipoi
import os
import pandas as pd


def test_load_models_kipoi():
    k = kipoi.config.get_source("kipoi")

    l = k.list_models()  # all the available models

    assert "extended_coda" in l
    model = "extended_coda"
    m_dir = k.pull_model(model)

    # load the model
    kipoi.Model(m_dir, source="dir")

    kipoi.Model(model, source="kipoi")
    kipoi.DataLoader_factory(model)


def test_load_models_local():
    model = "examples/extended_coda"
    kipoi.Model(model, source="dir")
    kipoi.DataLoader_factory(model, source="dir")


def test_list_models():
    k = kipoi.config.get_source("kipoi")

    df = k.list_models_df()
    assert isinstance(df, pd.DataFrame)

    # column names
    df_model_columns = ['model', 'name', 'version', 'author', 'descr', 'type', 'inputs', 'targets', 'tags']
    assert df_model_columns == list(df.columns)

    #
    df_all = kipoi.list_models()
    assert ["source"] + df_model_columns == list(df_all.columns)

    kipoi.model_info("extended_coda")

    kipoi.model_info("extended_coda", source="kipoi")

    # local files
    kipoi.model_info("examples/extended_coda", source="dir")
