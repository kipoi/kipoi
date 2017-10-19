"""Test the parsing utilities for ArraySchema
"""
import pytest
from pytest import raises
from kipoi.components import ArraySchema
from related import from_yaml

GOOD_SCHEMAS = ["""
shape: (100, )
description: some input
""", """
shape: (None, 100)
description: some input
""", """
shape: (None, )
description: some input
""", """
shape: (100, )
description: some input
special_type: DNASeq
""", """
shape: (100, )
description: some input
associated_metadata:  # as a list
  - ranges
""", """
shape: (100, )
description: some input
associated_metadata: ranges  # as a single element
"""]

BAD_SCHEMAS = ["""
shape: (100, )
# description missing
""", """
# shape missing
description: some input
""", """
shape: 100  # not a tuple
description: some input
""", """
shape: (100, )
description: some input
special_type: something # type not supported
"""]


@pytest.mark.parametrize("info_str", GOOD_SCHEMAS)
def test_parse_correct_info(info_str):
    # loading works
    info = ArraySchema.from_config(from_yaml(info_str))

    # cfg works
    cfg = info.get_config()
    info2 = ArraySchema.from_config(cfg)
    assert str(info) == str(info2)


@pytest.mark.parametrize("info_str", BAD_SCHEMAS)
def test_parse_bad_info(info_str):
    bim = from_yaml(info_str)

    with raises(Exception):
        ArraySchema.from_config(bim)
