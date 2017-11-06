"""Test the CLI interface
"""
# import keras  # otherwise I get a segfault from keras ?!
import pytest
import sys
import os
import yaml
import kipoi
import kipoi.utils
from kipoi.pipeline import install_model_requirements
import config

# TODO - check if you are on travis or not regarding the --install-req flag
# INSTALL_REQ = True
INSTALL_REQ = config.install_req

EXAMPLES_TO_RUN = ["rbp", "extended_coda", "iris_model_template"]


def read_json_yaml(filepath):
    with open(filepath) as ifh:
        return yaml.load(ifh)


def get_dataloader_cfg(model_dir):
    return read_json_yaml(os.path.join(model_dir, 'dataloader.yaml'))


def get_test_kwargs(model_dir):
    return read_json_yaml(os.path.join(model_dir, 'test_files/test.json'))


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_dataloader_model(example):
    """Test dataloader
    """
    if example in {"rbp", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")

    example_dir = "examples/{0}".format(example)

    # install the dependencies
    # - TODO maybe put it implicitly in load_dataloader?
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)

    Dl = kipoi.get_dataloader_factory(example_dir, source="dir")

    test_kwargs = get_test_kwargs(example_dir)

    # get dataloader

    # get model
    model = kipoi.get_model(example_dir, source="dir")

    with kipoi.utils.cd(example_dir + "/test_files"):
        # initialize the dataloader
        dataloader = Dl(**test_kwargs)

        # sample a batch of data
        it = dataloader.batch_iter()
        batch = next(it)
        # predict with a model
        model.predict_on_batch(batch["inputs"])
