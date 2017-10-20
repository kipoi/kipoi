"""Test the parsing utilities for ArraySchema
"""
import pytest
from pytest import raises
from kipoi.components import DataLoaderSchema
from related import from_yaml

# common header
inp_targ = """
inputs:
    seq:
        shape: (4, 100)
        description: One-hot encoded DNA sequence
        special_type: bigwig
targets:
    binding_site:
        shape: (1, )
        description: Binding strength
        special_type: bigwig
"""

GOOD_SCHEMAS = ["""
metadata:
    ranges:
        special_type: ranges
        description: One-hot encoded RNA sequence
    dist_polya_st:
        description: Array of something
""", """
metadata:
    ranges:
        special_type: ranges
        description: One-hot encoded RNA sequence
    dist_polya_st:
        - description: Array of something
        - description: Array of something else
        - description: Array 3
""", """
metadata:
    ranges:
        special_type: ranges
        description: One-hot encoded RNA sequence
    dist_polya_st:
        nested_structure:
            description: array
        nested_structure:
            - subnested1:
                  description: array
              subnested3:
                  description: array
            - description: this is another array here
              special_type: ranges
"""]


BAD_SCHEMAS = ["""
metadata:
    ranges:
        special_type: ranges
        # description missing
    dist_polya_st:
        description: Array of something
""", """
metadata:
    ranges:
        special_type: asd # unsupported special_type
        description: One-hot encoded RNA sequence
    dist_polya_st:
        - description: Array of something
        - description: Array of something else
        - description: Array 3
""", """
metadata:
    ranges:
        special_type: ranges
        description: One-hot encoded RNA sequence
    dist_polya_st:
        nested_structure:
            description: array
        nested_structure:
            - subnested1:
                  description: array
              subnested3:
                  description: array
            - this # element with no real leaf
"""]


# Class to test
CLS = DataLoaderSchema


@pytest.mark.parametrize("info_str", GOOD_SCHEMAS)
def test_parse_correct_info(info_str):
    info_str = inp_targ + info_str  # add the input: targets headers
    # loading works
    info = CLS.from_config(from_yaml(info_str))

    # cfg works
    cfg = info.get_config()
    info2 = CLS.from_config(cfg)
    assert str(info) == str(info2)


@pytest.mark.parametrize("info_str", BAD_SCHEMAS)
def test_parse_bad_info(info_str):
    info_str = inp_targ + info_str  # add the input: targets headers
    bim = from_yaml(info_str)

    with raises(Exception):
        CLS.from_config(bim)
