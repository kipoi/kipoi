"""Test the KipoiModel loading
"""
import pytest
import kipoi
import sys
from kipoi.pipeline import install_model_requirements

EXAMPLES_TO_RUN = ["rbp", "extended_coda"]
# TODO - finish the unit-test
INSTALL_REQ = True


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_load_model(example):
    example_dir = "examples/{0}".format(example)

    if example == "rbp" and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir")
    m = kipoi.Model(example_dir, source="dir")

    m.arch
    m.info
    m.schema
    m.schema.inputs
    m.source
    m.weights
    m.default_dataloader
    m.model
    m.predict_on_batch
