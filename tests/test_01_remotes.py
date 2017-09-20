"""Test the Kipoi remotes
"""
import modelzoo
import os
import pandas as pd


def test_load_models_kipoi():
    k = modelzoo.config.get_source("kipoi")

    l = k.list_models()  # all the available models

    assert "extended_coda" in l
    model = "extended_coda"
    m_dir = k.pull_model(model)

    # load the model
    modelzoo.load_model(m_dir)

    modelzoo.load_model(model)
    modelzoo.load_extractor(model)


def test_load_models_local():
    model = "examples/extended_coda"
    modelzoo.load_model(model, source="dir")
    modelzoo.load_extractor(model, source="dir")


def test_list_models():
    k = modelzoo.config.get_source("kipoi")

    df = k.list_models_df()
    assert isinstance(df, pd.DataFrame)

    # column names
    df_model_columns = ['model', 'name', 'version', 'author', 'description', 'type', 'inputs', 'targets', 'tags']
    assert df_model_columns == list(df.columns)

    #
    df_all = modelzoo.list_models()
    assert ["source"] + df_model_columns == list(df_all.columns)

    modelzoo.model_info("extended_coda")

    modelzoo.model_info("extended_coda", source="kipoi")

    # local files
    modelzoo.model_info("examples/extended_coda", source="dir")
