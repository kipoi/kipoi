"""Test the CLI interface
"""
import torch  # need it before keras otherwise I get the following error: ImportError: dlopen: cannot load any more object with static TLS
from torch.utils.data import DataLoader
import keras  # otherwise I get a segfault from keras ?!
import pytest
import sys
import os
import yaml
from contextlib import contextmanager
import kipoi
from kipoi.data import numpy_collate
from kipoi.utils import cd
from kipoi.pipeline import install_model_requirements


# TODO - check if you are on travis or not regarding the --install-req flag
INSTALL_REQ = True
INSTALL_REQ = False

EXAMPLES_TO_RUN = ["rbp", "extended_coda"]


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
    if example == "rbp" and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    example_dir = "examples/{0}".format(example)

    Dl = kipoi.DataLoader_factory(example_dir, source="dir")

    test_kwargs = get_test_kwargs(example_dir)

    # install the dependencies
    # - TODO maybe put it implicitly in load_dataloader?
    if INSTALL_REQ:
        install_model_requirements(example_dir)
    # get dataloader

    # get model
    model = kipoi.Model(example_dir, source="dir")

    with cd(example_dir + "/test_files"):
        # initialize the dataloader
        dataloader = Dl(**test_kwargs)
        # get first sample
        dataloader[0]
        len(dataloader)

        # sample a batch of data
        dl = DataLoader(dataloader, collate_fn=numpy_collate)
        it = iter(dl)
        batch = next(it)
        # predict with a model
        model.predict_on_batch(batch["inputs"])
