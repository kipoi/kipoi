"""Test the parsing utilities
"""

import pytest
from kipoi.component import Info
import yaml

GOOD_INFO = """
author: Ziga Avsec
name: rbp_eclip
version: 0.1
description: RBP binding prediction
"""

GOOD_INFO2 = """
author: Ziga Avsec
name: rbp_eclip
version: 0.1
description: RBP binding prediction
tags: ['var_interpretation']
"""

BAD_INFO_EXTRA = """
author: Ziga Avsec
name: rbp_eclip
version: 0.1
description: RBP binding prediction
extra_field: asd
"""

BAD_INFO_MISSING = """
author: Ziga Avsec
name: rbp_eclip
version: 0.1
"""


@pytest.mark.parametrize("info_str", [GOOD_INFO, GOOD_INFO2])
def test_parse_correct_info(info_str):
    gi = yaml.load(info_str)

    # loading works
    info = Info.from_config(gi)

    # repr works
    info2 = eval(info.__repr__())
    assert str(info) == str(info2)

    # cfg works
    cfg = Info.from_config(gi).get_config()
    info = Info.from_config(cfg)
    assert str(info) == str(info2)


@pytest.mark.parametrize("info_str", [BAD_INFO_EXTRA, BAD_INFO_MISSING])
def test_parse_bad_info(info_str):
    bim = yaml.load(info_str)

    with pytest.raises(Exception):
        Info.from_config(bim)
