"""Test the parsing utilities for ArraySchema
"""
import pytest
from pytest import raises
from kipoi.components import ModelSchema
from related import from_yaml

CLS = ModelSchema

GOOD_EXAMPLES = ["""
inputs:
    seq:
        shape: (4, None)
        descr: One-hot encoded RNA sequence
        special_type: DNASeq
        associated_metadata: ranges
    dist_polya_st:
        shape: (1, 10)
        descr: Distance to poly-a site transformed with B-splines
targets:
    binding_site:
        shape: (1, )
        descr: Binding strength
""", """
inputs:
    seq:
        shape: (4, 100)
        descr: One-hot encoded RNA sequence
        special_type: DNASeq
        associated_metadata:
          - ranges
targets:
    binding_site:
        shape: (1, )
        descr: Binding strength
""", """
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
"""]

BAD_EXAMPLES = ["""
inputs:
    seq:
        descr: One-hot encoded RNA sequence
        special_type: DNASeq
        associated_metadata: ranges
    dist_polya_st:
        shape: (1, 10)
        descr: Distance to poly-a site transformed with B-splines
targets:
    binding_site:
        shape: (1, )
        descr: Binding strength
""", """
# inputs missing
targets:
    binding_site:
        shape: (1, )
        descr: Binding strength
""", """
# targets missing
inputs:
    seq:
        shape: (4, 100)
        descr: One-hot encoded RNA sequence
""", """
# no output
inputs:
    seq:
        shape: (4, 100)
        descr: One-hot encoded RNA sequence
targets:
"""]


@pytest.mark.parametrize("info_str", GOOD_EXAMPLES)
def test_parse_correct_info(info_str):
    # loading works
    info = CLS.from_config(from_yaml(info_str))

    # cfg works
    cfg = info.get_config()
    info2 = CLS.from_config(cfg)
    assert str(info) == str(info2)


@pytest.mark.parametrize("info_str", BAD_EXAMPLES)
def test_parse_bad_info(info_str):
    bim = from_yaml(info_str)

    with raises(Exception):
        CLS.from_config(bim)
