"""Test the CLI interface
"""
# import keras  # otherwise I get a segfault from keras ?!
import pytest
import sys
import os
import yaml
import kipoi
import kipoi_utils.utils
from kipoi.pipeline import install_model_requirements
import config
import numpy as np 

# TODO - check if you are on travis or not regarding the --install-req flag
# INSTALL_REQ = True
INSTALL_REQ = config.install_req

EXAMPLES_TO_RUN = ["pyt"]


def read_json_yaml(filepath):
    with open(filepath) as ifh:
        return yaml.safe_load(ifh)


def get_dataloader_cfg(model_dir):
    return read_json_yaml(os.path.join(model_dir, 'dataloader.yaml'))


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_dataloader_model(example):
    """Test dataloader
    """
    if example in {"rbp", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")

    example_dir = "example/models/{0}".format(example)

    # install the dependencies
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)

    Dl = kipoi.get_dataloader_factory(example_dir, source="dir")

    test_kwargs = Dl.example_kwargs
    # get dataloader

    # get model
    model = kipoi.get_model(example_dir, source="dir")

    with kipoi_utils.utils.cd(example_dir):
        # initialize the dataloader
        dataloader = Dl(**test_kwargs)

        # sample a batch of data
        it = dataloader.batch_iter()
        batch = next(it)
        # predict with a model        
        if isinstance(batch["inputs"], np.ndarray):
            model.predict_on_batch(batch["inputs"].astype(np.float32))
        else:
            model.predict_on_batch(batch["inputs"])
