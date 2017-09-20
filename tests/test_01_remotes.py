"""Test the Kipoi remotes
"""
import modelzoo
import os


def test_load_models():
    k = modelzoo.config.model_sources()["kipoi"]

    l = k.list_models()  # all the available models

    assert "extended_coda" in l
    model = "extended_coda"
    m_dir = k.pull_model(model)

    # load the model
    m = modelzoo.load_model(m_dir)

    m2 = modelzoo.load_model(model)
    d2 = modelzoo.load_extractor(model)
