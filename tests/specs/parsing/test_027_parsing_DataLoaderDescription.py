"""Test the parsing utilities for ArraySchema
"""
import pytest
import six
from pytest import raises
from kipoi.specs import DataLoaderDescription, example_kwargs, RemoteFile
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
        doc: tsv file with `chrom start end id score strand`
        example:
          url: intervals.tsv
          md5: dummy-md5
        type: str
    fasta_file:
        doc: Reference genome sequence
        example: genome.fa
    gtf_file:
        doc: file path; Genome annotation GTF file pickled using pandas.
        example: gtf.gtf
    preproc_transformer:
        doc: path to the serialized tranformer used for pre-processing.
        example: path.transformer
    target_file:
        doc: path to the targets (txt) file
        optional: True # use the same semantics as for the CLI interface?
info:
    authors:
        - name: Ziga Avsec
    name: rbp_eclip
    version: 0.1
    doc: RBP binding prediction
output_schema:
    inputs:
        seq:
            shape: (4, 101)
            special_type: DNASeq
            doc: One-hot encoded RNA sequence
            associated_metadata: ranges
        dist_polya_st:
            shape: (1, 10)
            doc: Distance to poly-a site transformed with B-splines
    targets:
        binding_site:
            shape: (1, )
            doc: Measured binding strength
    metadata:
        ranges:
            chr:
                type: str
                doc: Chromosome
            start:
                type: int
                doc: Start position
            end:
                type: int
                doc: End position
            id:
                type: str
                doc: Id of the sequence
            strand:
                type: str
                doc: Sequence strand
"""]

BAD_EXAMPLES = ["""
type: keras
args:
    arch: model/model.json
    weights: model/weights.h5
    custom_objects: model/custom_keras_objects.py
default_dataloader: dataloader.yaml # shall we call it just dataloader?
# info is missing
output_schema:
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

    assert isinstance(example_kwargs(info.args), dict)

    assert isinstance(info.args["intervals_file"].example, RemoteFile)
    assert isinstance(info.args["fasta_file"].example, str)

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
    model_file = "example/models/{0}/dataloader.yaml".format(example)

    dl = DataLoaderDescription.load(model_file)

    # check all the fields exists
    dl.type == "Dataset"

    dl.defined_as
    dl.args
    arg_elem = six.next(six.itervalues(dl.args))
    arg_elem.doc
    arg_elem.type
    arg_elem.optional

    dl.info
    dl.info.authors
    dl.info.name
    dl.info.version
    dl.info.tags
    dl.info.doc

    dl.output_schema
    dl.output_schema.inputs
    inp_elem = six.next(six.itervalues(dl.output_schema.inputs))
    inp_elem.shape
    inp_elem.special_type
    inp_elem.associated_metadata

    dl.output_schema.targets

    dl.output_schema.metadata
