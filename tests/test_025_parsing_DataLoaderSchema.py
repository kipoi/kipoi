"""Test the parsing utilities for ArraySchema
"""
import pytest
from pytest import raises
from kipoi.components import DataLoaderSchema
from related import from_yaml

# Class to test
CLS = DataLoaderSchema

# common header
inp_targ = """
inputs:
    seq:
        shape: (4, 100)
        doc: One-hot encoded DNA sequence
        special_type: bigwig
targets:
    binding_site:
        shape: (1, )
        doc: Binding strength
        special_type: bigwig
"""

GOOD_EXAMPLES = ["""
metadata:
    ranges:
        type: GenomicRanges
        doc: One-hot encoded RNA sequence
    dist_polya_st:
        doc: Array of something
""", """
metadata:
    ranges:
        type: GenomicRanges
        doc: One-hot encoded RNA sequence
    dist_polya_st:
        - doc: Array of something
        - doc: Array of something else
        - doc: Array 3
""", """
metadata:
    ranges:
        type: GenomicRanges
        doc: One-hot encoded RNA sequence
    dist_polya_st:
        nested_structure:
            doc: array
        nested_structure:
            - subnested1:
                  doc: array
              subnested3:
                  doc: array
            - doc: this is another array here
              type: GenomicRanges
"""]

BAD_EXAMPLES = ["""
metadata:
    ranges:
        type: GenomicRanges
        # doc missing
    dist_polya_st:
        doc: Array of something
""", """
metadata:
    ranges:
        type: asd # unsupported special_type
        doc: One-hot encoded RNA sequence
    dist_polya_st:
        - doc: Array of something
        - doc: Array of something else
        - doc: Array 3
""", """
metadata:
    ranges:
        type: GenomicRanges
        doc: One-hot encoded RNA sequence
    dist_polya_st:
        nested_structure:
            doc: array
        nested_structure:
            - subnested1:
                  doc: array
              subnested3:
                  doc: array
            - this # element with no real leaf
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
