"""Test the parsing utilities for ArraySchema
"""
import pytest
from pytest import raises
from kipoi.components import ArraySchema
from related import from_yaml

CLS = ArraySchema

GOOD_EXAMPLES = ["""
shape: (100, )
doc: some input
""", """
shape: (None, 100)
doc: some input
""", """
shape: (None, )
doc: some input
""", """
shape: (100, )
doc: some input
special_type: DNASeq
""", """
shape: (100, )
doc: some input
associated_metadata:  # as a list
  - ranges
""", """
shape: (100, )
doc: some input
associated_metadata: ranges  # as a single element
"""]

BAD_EXAMPLES = ["""
shape: (100, )
# doc missing
""", """
# shape missing
doc: some input
""", """
shape: 100  # not a tuple
doc: some input
""", """
shape: (100, )
doc: some input
special_type: something # type not supported
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


def test_correct_shape():
    correct_shapes = [(100, ),
                      (None, 100),
                      (None, ),
                      (100, ),
                      (100, ),
                      (100, )]

    for i, info_str in enumerate(GOOD_EXAMPLES):
        correct_shape = correct_shapes[i]
        info = CLS.from_config(from_yaml(info_str))
        assert info.shape == correct_shape
