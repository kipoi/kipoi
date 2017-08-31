"""Run the example scripts
"""

import pytest
import subprocess
import sys

EXAMPLES_TO_RUN = ["rbp", "extended_coda"]


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_example_dir(example):
    if example == "rbp" and sys.version_info[0] == 2:
        print("rbp example not supported on python 2 ")
        return None

    example_dir = "examples/{0}".format(example)

    # TODO - check if you are on travis or not...
    returncode = subprocess.call(args=["python", "./modelzoo/__main__.py", "test",
                                       "--batch_size=4",
                                       "--install-req",
                                       example_dir])
    assert returncode == 0
