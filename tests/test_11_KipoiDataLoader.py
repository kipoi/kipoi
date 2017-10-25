"""Test the KipoiDataLoader loading
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
    Dl = kipoi.DataLoader_factory(example_dir, source="dir")

    Dl.type
    Dl.defined_as
    Dl.args
    Dl.info
    Dl.schema
    Dl.source
    Dl.__len__
    Dl.__getitem__
