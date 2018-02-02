"""Test the KipoiModel loading
"""
import pytest
import kipoi
import sys
import config
from kipoi.pipeline import install_model_requirements

EXAMPLES_TO_RUN = ["rbp", "extended_coda", "iris_model_template", "pyt"]
# TODO - finish the unit-test
INSTALL_REQ = config.install_req


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_load_model(example):
    example_dir = "examples/{0}".format(example)

    if example in {"rbp", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")

    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir")
    m = kipoi.get_model(example_dir, source="dir")

    if isinstance(m, kipoi.model.KerasModel):
        m.arch
        m.weights
    m.info
    m.schema
    m.schema.inputs
    m.source
    m.default_dataloader
    m.model
    m.predict_on_batch
