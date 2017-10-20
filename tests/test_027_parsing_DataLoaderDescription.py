"""Test the parsing utilities for ArraySchema
"""
import pytest
import six
from pytest import raises
from kipoi.components import DataLoaderDescription
from related import from_yaml

# Class to test
CLS = DataLoaderDescription

# common header
inp_targ = """
"""

GOOD_EXAMPLES = ["""
type: Dataset
defined_as: dataloader.py::SeqDistDataset
args:
    intervals_file:
        descr: tsv file with `chrom start end id score strand`
        type: str
    fasta_file:
        descr: Reference genome sequence
    gtf_file:
        descr: file path; Genome annotation GTF file pickled using pandas.
    preproc_transformer:
        descr: path to the serialized tranformer used for pre-processing.
    target_file:
        descr: path to the targets (txt) file
        optional: True # use the same semantics as for the CLI interface?
info:
    author: Ziga Avsec
    name: rbp_eclip
    version: 0.1
    descr: RBP binding prediction
schema:
    inputs:
        seq:
            shape: (4, 101)
            special_type: DNASeq
            descr: One-hot encoded RNA sequence
            associated_metadata: ranges
        dist_polya_st:
            shape: (1, 10)
            descr: Distance to poly-a site transformed with B-splines
    targets:
        binding_site:
            shape: (1, )
            descr: Measured binding strength
    metadata:
        ranges:
            chr:
                type: str
                descr: Chromosome
            start:
                type: int
                descr: Start position
            end:
                type: int
                descr: End position
            id:
                type: str
                descr: Id of the sequence
            strand:
                type: str
                descr: Sequence strand
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
            descr: One-hot encoded RNA sequence
        dist_polya_st:
            shape: (None, 1, 10)
            descr: Distance to poly-a site transformed with B-splines
    targets:
        binding_site:
            shape: (1, )
            descr: Predicted binding strength
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
    model_file = "examples/{0}/dataloader.yaml".format(example)

    dl = DataLoaderDescription.load(model_file)

    # check all the fields exists
    dl.type == "Dataset"

    dl.defined_as
    dl.args
    arg_elem = six.next(six.itervalues(dl.args))
    arg_elem.descr
    arg_elem.type
    arg_elem.optional

    dl.info
    dl.info.author
    dl.info.name
    dl.info.version
    dl.info.tags
    dl.info.descr

    dl.schema
    dl.schema.inputs
    inp_elem = six.next(six.itervalues(dl.schema.inputs))
    inp_elem.shape
    inp_elem.special_type
    inp_elem.associated_metadata

    dl.schema.targets

    dl.schema.metadata
