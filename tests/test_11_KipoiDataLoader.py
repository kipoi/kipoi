"""Test the KipoiDataLoader loading
"""
import pytest
import kipoi
import sys
from kipoi.pipeline import install_dataloader_requirements
import config
EXAMPLES_TO_RUN = ["rbp", "extended_coda", "iris_model_template"]
# TODO - finish the unit-test

INSTALL_REQ = config.install_req


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_load_model(example):
    example_dir = "examples/{0}".format(example)

    if example in {"rbp", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")

    if INSTALL_REQ:
        install_dataloader_requirements(example_dir, "dir")
    Dl = kipoi.get_dataloader_factory(example_dir, source="dir")

    Dl.type
    Dl.defined_as
    Dl.args
    Dl.info
    Dl.output_schema
    Dl.source
    Dl.__len__
    Dl.__getitem__
