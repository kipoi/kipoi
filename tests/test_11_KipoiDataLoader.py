"""Test the KipoiDataLoader loading
"""
import pytest
import kipoi
import sys
from kipoi.pipeline import install_dataloader_requirements
import config

EXAMPLES_TO_RUN = ["iris_model_template", "pyt"]
# TODO - finish the unit-test

INSTALL_REQ = config.install_req


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_load_model(example):
    example_dir = "example/models/{0}".format(example)

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
    # datalaoder
    Dl.batch_iter
    Dl.load_all

    Dl.print_args()

    kipoi.get_dataloader_descr(example_dir, source="dir").print_kwargs()
