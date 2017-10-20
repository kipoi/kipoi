"""Test the parsing utilities for ArraySchema
"""
import pytest
from pytest import raises
from kipoi.components import DataLoaderArgument
from related import from_yaml

# Class to test
CLS = DataLoaderArgument


# common header
inp_targ = """
"""

GOOD_EXAMPLES = ["""
descr: some description
type: str
""", """
descr: some description
""", """
descr: some description
optional: True
name: specified name
""", """
descr: some description
tags:
  - tag1
  - tag2
""", """
descr: some description
# only one tag
tags: tag1
"""]


BAD_EXAMPLES = ["""
# missing description
type: str
""", """
descr: some description
optional: maybe # not bool
""", """
descr: some description
# not a list
tags:
  asd: dsa
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
