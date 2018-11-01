"""Test the parsing utilities
"""

import pytest
from kipoi.specs import Info, Author
from related import from_yaml

CLS = Info

GOOD_EXAMPLES = ["""
authors:
    - name: Ziga Avsec
doc: RBP binding prediction
""", """
authors:
    - name: Ziga Avsec
name: rbp_eclip
version: 0.1
# doc: RBP binding prediction
tags: ['var_interpretation']
extra_field: asd
"""]

BAD_EXAMPLES = ["""
authors:
    - nam: Ziga Avsec
version: 0.1
doc: RBP binding prediction
"""]


@pytest.mark.parametrize("info_str", GOOD_EXAMPLES)
def test_parse_correct_info(info_str):
    # loading works
    info = CLS.from_config(from_yaml(info_str))

    # repr works
    info2 = eval(info.__repr__())
    assert str(info) == str(info2)

    # cfg works
    cfg = info.get_config()
    info = CLS.from_config(cfg)
    assert str(info) == str(info2)


@pytest.mark.parametrize("info_str", BAD_EXAMPLES)
def test_parse_bad_info(info_str):
    bim = from_yaml(info_str)

    with pytest.raises(Exception):
        CLS.from_config(bim)
