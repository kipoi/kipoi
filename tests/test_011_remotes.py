"""Test the Kipoi sources
"""
import kipoi
import os
import pandas as pd


def test_load_models_kipoi():
    k = kipoi.config.get_source("kipoi")

    ls = k.list_models()  # all the available models

    assert "HAL" in list(ls.model)
    model = "HAL"
    k.pull_model(model)

    # load the model
    kipoi.get_model(os.path.join(k.local_path, "HAL"), source="dir")

    kipoi.get_model(model, source="kipoi")
    kipoi.get_dataloader_factory(model)


def test_load_models_local():
    model = "example/models/iris_model_template"
    kipoi.get_model(model, source="dir")
    kipoi.get_dataloader_factory(model, source="dir")


def test_list_models():
    k = kipoi.config.get_source("kipoi")

    df = k.list_models()
    assert isinstance(df, pd.DataFrame)

    # column names
    df_model_columns = ['model', 'version', 'authors', 'contributors', 'doc', 'type', 'inputs', 'targets',
                        'veff_score_variants',
                        'license', 'cite_as', 'trained_on', 'training_procedure', 'tags']
    assert df_model_columns == list(df.columns)

    #
    df_all = kipoi.list_models()
    assert ["source"] + df_model_columns == list(df_all.columns)

    kipoi.get_model_descr("extended_coda")

    kipoi.get_model_descr("extended_coda", source="kipoi")

    # local files
    kipoi.get_model_descr("example/models/extended_coda", source="dir")


def test_list_models_group():
    dfg = kipoi.get_source("kipoi").list_models_by_group()
    dfg_columns = ["group", "N_models", "N_subgroups", "is_group", "authors",
                   "contributors",
                   "veff_score_variants",
                   "type", "license", "cite_as", "tags"]
    assert dfg_columns == list(dfg.columns)
    assert len(dfg) > 0
    assert dfg.group.str.contains("^CpGenie$").sum() == 1


def test_github_permalink():
    link = "https://github.com/kipoi/models/tree/7d3ea7800184de414aac16811deba6c8eefef2b6/pwm_HOCOMOCO/human/CTCF"
    kipoi.get_model(link, source="github-permalink")
    kipoi.get_model_descr(link, source="github-permalink")
    assert len(kipoi.get_source("github-permalink").list_models()) == 0
