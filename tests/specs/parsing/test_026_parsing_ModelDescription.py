"""Test the parsing utilities for ArraySchema
"""
import pytest
from pytest import raises
from kipoi.specs import ModelDescription
from related import from_yaml
import six

# Class to test
CLS = ModelDescription

# common header
inp_targ = """
"""

GOOD_EXAMPLES = ["""
type: keras
args:
    arch: model/model.json
    weights: model/weights.h5
    custom_objects: model/custom_keras_objects.py
default_dataloader: dataloader.yaml # shall we call it just dataloader?
info:
    authors:
        - name: Ziga Avsec
    name: rbp_eclip
    version: 0.1
    doc: RBP binding prediction
schema:
    inputs:
        seq:
            shape: (4, 101)
            special_type: DNASeq
            doc: One-hot encoded RNA sequence
        dist_polya_st:
            shape: (None, 1, 10)
            doc: Distance to poly-a site transformed with B-splines
    targets:
        binding_site:
            shape: (1, )
            doc: Predicted binding strength
"""]

BAD_EXAMPLES = ["""
type: keras
args:
    arch: model/model.json
    weights: model/weights.h5
    custom_objects: model/custom_keras_objects.py
default_dataloader: dataloader.yaml # shall we call it just dataloader?
# info is missing
schema:
    inputs:
        seq:
            shape: (4, 101)
            special_type: DNASeq
            doc: One-hot encoded RNA sequence
        dist_polya_st:
            shape: (None, 1, 10)
            doc: Distance to poly-a site transformed with B-splines
    targets:
        binding_site:
            shape: (1, )
            doc: Predicted binding strength
"""]


@pytest.mark.parametrize("info_str", GOOD_EXAMPLES)
def test_parse_correct_info(info_str):
    info_str = inp_targ + info_str  # add the input: targets headers
    # loading works
    info = CLS.from_config(from_yaml(info_str))

    # cfg works
    cfg = info.get_config()
    info2 = CLS.from_config(cfg)
    assert str(info) == str(info2)


@pytest.mark.parametrize("info_str", BAD_EXAMPLES)
def test_parse_bad_info(info_str):
    info_str = inp_targ + info_str  # add the input: targets headers
    bim = from_yaml(info_str)

    with raises(Exception):
        CLS.from_config(bim)


# --------------------------------------------
# load example yaml files


KERAS_EXAMPLES_TO_LOAD = ["rbp", "extended_coda"]


@pytest.mark.parametrize("example", KERAS_EXAMPLES_TO_LOAD)
def test_model_loading_on_examples(example):
    """Test extractor
    """
    model_file = "example/models/{0}/model.yaml".format(example)

    md = ModelDescription.load(model_file)

    # check all the fields exists
    md.type == "keras"

    md.args["weights"]
    # md.args["custom_objects"]  # doesn't have to be defined

    md.default_dataloader

    md.info
    md.info.authors
    md.info.name
    md.info.version
    md.info.tags
    md.info.doc

    md.schema
    md.schema.inputs

    inp_elem = six.next(six.itervalues(md.schema.inputs))
    inp_elem.shape
    inp_elem.special_type
    inp_elem.associated_metadata

    md.schema.targets
