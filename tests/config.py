import sys

import pytest

install_req = False


pythonversion = pytest.mark.skipif(
    sys.version_info.major == 3 and sys.version_info.minor < 8, \
         reason="These tests are either repeated in tests/legacy/test_cli_examples.py \
             or not supported >=3.8"
)
