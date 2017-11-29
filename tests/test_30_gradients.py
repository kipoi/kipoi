import warnings
warnings.filterwarnings('ignore')
import pytest
import sys
import os
import yaml
from contextlib import contextmanager
import kipoi
from kipoi.pipeline import install_model_requirements
from kipoi.utils import Slice_conv




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
    return read_json_yaml(os.path.join(model_dir, 'example_files/test.json'))



@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_extractor_model(example):
    """Test extractor
    """
    if example == "rbp" and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")
    #
    example_dir = "examples/{0}".format(example)
    # install the dependencies
    # - TODO maybe put it implicitly in load_dataloader?
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)
    #
    Dl = kipoi.get_dataloader_factory(example_dir, source="dir")
    #
    test_kwargs = get_test_kwargs(example_dir)
    #
    # install the dependencies
    # - TODO maybe put it implicitly in load_extractor?
    if INSTALL_REQ:
        install_model_requirements(example_dir, source="dir")
    # get extractor
    #Extractor = kipoi.load_extractor(example_dir, source="dir")
    #
    # get model
    model = kipoi.get_model(example_dir, source="dir")
    #
    with cd(example_dir + "/example_files"):
        # initialize the dataloader
        dataloader = Dl(**test_kwargs)
        #
        # sample a batch of data
        it = dataloader.batch_iter()
        batch = next(it)
        # predict with a model
        model.predict_on_batch(batch["inputs"])
        model.pred_grad(batch["inputs"], Slice_conv()[:, 0])
        #if example == "rbp":
        #    model._input_grad(batch["inputs"], -1, Slice_conv()[:, 0])
        #elif example == "extended_coda":
        #    model._input_grad(batch["inputs"], -1, filter_func=tf.reduce_max, filter_func_kwargs={"axis": 1})


