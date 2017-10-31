"""Test the parsing utilities
"""

import pytest
from kipoi.components import Info
from related import from_yaml

CLS = Info

GOOD_EXAMPLES = ["""
author: Ziga Avsec
name: rbp_eclip
version: 0.1
descr: RBP binding prediction
""", """
author: Ziga Avsec
name: rbp_eclip
version: 0.1
descr: RBP binding prediction
tags: ['var_interpretation']
"""]


BAD_EXAMPLES = ["""
author: Ziga Avsec
name: rbp_eclip
version: 0.1
descr: RBP binding prediction
extra_field: asd
""", """
author: Ziga Avsec
name: rbp_eclip
version: 0.1
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
