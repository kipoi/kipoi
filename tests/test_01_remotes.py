"""Test the Kipoi remotes
"""
import modelzoo
import os


def test_load_models():
    k = modelzoo.config.model_sources()["kipoi"]

    l = k.list_models()  # all the available models

    m_dir = k.pull_model(l[1])

    # load the model
    m = modelzoo.load_model(m_dir)

    m2 = modelzoo.load_model(l[1])
    d2 = modelzoo.load_extractor(l[1])
