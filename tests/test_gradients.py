import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader
import pytest
import sys
import os
import yaml
from contextlib import contextmanager
import kipoi
from kipoi.data import numpy_collate
from kipoi.pipeline import install_model_requirements
import tensorflow as tf



# TODO: Test on Theano model


# TODO - check if you are on travis or not regarding the --install-req flag
INSTALL_REQ = True
# INSTALL_REQ = False

EXAMPLES_TO_RUN = ["rbp", "extended_coda"]


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def read_json_yaml(filepath):
    with open(filepath) as ifh:
        return yaml.load(ifh)


def get_extractor_cfg(model_dir):
    return read_json_yaml(os.path.join(model_dir, 'extractor.yaml'))


def get_test_kwargs(model_dir):
    return read_json_yaml(os.path.join(model_dir, 'test_files/test.json'))

class Slice_conv:
    def __getitem__(self, key): return key



@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_extractor_model(example):
    """Test extractor
    """
    if example == "rbp" and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")
    #
    example_dir = "examples/{0}".format(example)
    cfg = get_extractor_cfg(example_dir)
    #
    kipoi.data.validate_extractor_spec(cfg["extractor"])
    test_kwargs = get_test_kwargs(example_dir)
    #
    # install the dependencies
    # - TODO maybe put it implicitly in load_extractor?
    if INSTALL_REQ:
        install_model_requirements(example_dir)
    # get extractor
    Extractor = kipoi.load_extractor(example_dir, source="dir")
    #
    # get model
    model = kipoi.load_model(example_dir, source="dir")
    #
    with cd(example_dir + "/test_files"):
        # initialize the extractor
        extractor = Extractor(**test_kwargs)
        # get first sample
        extractor[0]
        len(extractor)
        kipoi.data.validate_extractor(extractor)
        #
        # sample a batch of data
        dl = DataLoader(extractor, collate_fn=numpy_collate)
        it = iter(dl)
        batch = next(it)
        # predict with a model
        model.predict_on_batch(batch["inputs"])
        if example == "rbp":
            model.input_grad(batch["inputs"], -1, Slice_conv()[:, 0])
        elif example == "extended_coda":
            model.input_grad(batch["inputs"], -1, filter_func=tf.reduce_max, filter_func_kwargs={"axis": 1})


