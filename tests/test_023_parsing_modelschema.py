"""Test the parsing utilities for ArraySchema
"""
import pytest
from pytest import raises
from kipoi.components import ModelSchema
from related import from_yaml

GOOD_SCHEMAS = ["""
inputs:
    seq:
        shape: (4, None)
        description: One-hot encoded RNA sequence
        special_type: DNASeq
        associated_metadata: ranges
    dist_polya_st:
        shape: (1, 10)
        description: Distance to poly-a site transformed with B-splines
targets:
    binding_site:
        shape: (1, )
        description: Binding strength
""", """
inputs:
    seq:
        shape: (4, 100)
        description: One-hot encoded RNA sequence
        special_type: DNASeq
        associated_metadata:
          - ranges
targets:
    binding_site:
        shape: (1, )
        description: Binding strength
""", """
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
"""]

BAD_SCHEMAS = ["""
inputs:
    seq:
        description: One-hot encoded RNA sequence
        special_type: DNASeq
        associated_metadata: ranges
    dist_polya_st:
        shape: (1, 10)
        description: Distance to poly-a site transformed with B-splines
targets:
    binding_site:
        shape: (1, )
        description: Binding strength
""", """
# inputs missing
targets:
    binding_site:
        shape: (1, )
        description: Binding strength
""", """
# targets missing
inputs:
    seq:
        shape: (4, 100)
        description: One-hot encoded RNA sequence
""", """
# no output
inputs:
    seq:
        shape: (4, 100)
        description: One-hot encoded RNA sequence
targets:
"""]


@pytest.mark.parametrize("info_str", GOOD_SCHEMAS)
def test_parse_correct_info(info_str):
    # loading works
    info = ModelSchema.from_config(from_yaml(info_str))

    # cfg works
    cfg = info.get_config()
    info2 = ModelSchema.from_config(cfg)
    assert str(info) == str(info2)


@pytest.mark.parametrize("info_str", BAD_SCHEMAS)
def test_parse_bad_info(info_str):
    bim = from_yaml(info_str)

    with raises(Exception):
        ModelSchema.from_config(bim)
