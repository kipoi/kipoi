"""Test the Kipoi remotes
"""
import kipoi
import os
import pandas as pd


def test_load_models_kipoi():
    k = kipoi.config.get_source("kipoi")

    l = k.list_models()  # all the available models

    assert "HAL" in list(l.model)
    model = "HAL"
    mpath = k.pull_model(model)
    m_dir = os.path.dirname(mpath)

    # load the model
    kipoi.get_model(m_dir, source="dir")

    kipoi.get_model(model, source="kipoi")
    kipoi.get_dataloader_factory(model)


def test_load_models_local():
    model = "examples/iris_model_template"
    kipoi.get_model(model, source="dir")
    kipoi.get_dataloader_factory(model, source="dir")


def test_list_models():
    k = kipoi.config.get_source("kipoi")

    df = k.list_models()
    assert isinstance(df, pd.DataFrame)

    # column names
    df_model_columns = ['model', 'version', 'authors', 'contributors', 'doc', 'type', 'inputs', 'targets',
                        'postproc_score_variants',
                        'license', 'cite_as', 'trained_on', 'training_procedure', 'tags']
    assert df_model_columns == list(df.columns)

    #
    df_all = kipoi.list_models()
    assert ["source"] + df_model_columns == list(df_all.columns)

    kipoi.get_model_descr("extended_coda")

    kipoi.get_model_descr("extended_coda", source="kipoi")

    # local files
    kipoi.get_model_descr("examples/extended_coda", source="dir")


def test_list_models_group():
    dfg = kipoi.get_source("kipoi").list_models_by_group()
    dfg_columns = ["group", "N_models", "N_subgroups", "is_group", "authors",
                   "contributors",
                   "postproc_score_variants",
                   "type", "license", "cite_as", "tags"]
    assert dfg_columns == list(dfg.columns)
    assert len(dfg) > 0
    assert dfg.group.str.contains("^CpGenie$").sum() == 1


def test_github_permalink():
    component = "https://github.com/Avsecz/testlfs/tree/25eee661e75555516b6a7e529857e9ceeecdb711/m1/"
    dl_url = "https://minhaskamal.github.io/DownGit/#/home?url={0}".format(component)
    from kipoi.remote import GithubPermalinkSource
    local_path = "/tmp/kipoi/"

    component_path = GithubPermalinkSource._url_to_dir(component)
    from urllib.request import urlopen
    from tempfile import NamedTemporaryFile
    from shutil import unpack_archive
    zipurl = 'http://stash.compjour.org/data/1800ssa.zip'
    with urlopen(zipurl) as zipresp, NamedTemporaryFile() as tfile:
        tfile.write(zipresp.read())
        tfile.seek(0)
        unpack_archive(tfile.name, '/tmp/mystuff3', format='zip')

    cpath = get_file(fname="file.zip",
                     origin=dl_url,
                     extract=True,
                     archive_format="zip",
                     cache_subdir=component_path,
                     cache_dir=local_path)
    pass
