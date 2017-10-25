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
        descr: One-hot encoded DNA sequence
        special_type: bigwig
targets:
    binding_site:
        shape: (1, )
        descr: Binding strength
        special_type: bigwig
"""

GOOD_EXAMPLES = ["""
metadata:
    ranges:
        type: Ranges
        descr: One-hot encoded RNA sequence
    dist_polya_st:
        descr: Array of something
""", """
metadata:
    ranges:
        type: Ranges
        descr: One-hot encoded RNA sequence
    dist_polya_st:
        - descr: Array of something
        - descr: Array of something else
        - descr: Array 3
""", """
metadata:
    ranges:
        type: Ranges
        descr: One-hot encoded RNA sequence
    dist_polya_st:
        nested_structure:
            descr: array
        nested_structure:
            - subnested1:
                  descr: array
              subnested3:
                  descr: array
            - descr: this is another array here
              type: Ranges
"""]


BAD_EXAMPLES = ["""
metadata:
    ranges:
        type: Ranges
        # descr missing
    dist_polya_st:
        descr: Array of something
""", """
metadata:
    ranges:
        type: asd # unsupported special_type
        descr: One-hot encoded RNA sequence
    dist_polya_st:
        - descr: Array of something
        - descr: Array of something else
        - descr: Array 3
""", """
metadata:
    ranges:
        type: Ranges
        descr: One-hot encoded RNA sequence
    dist_polya_st:
        nested_structure:
            descr: array
        nested_structure:
            - subnested1:
                  descr: array
              subnested3:
                  descr: array
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
