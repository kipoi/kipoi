"""Run the example scripts
"""

import pytest
import subprocess


EXAMPLES_TO_RUN = ["rbp", "extended_coda"]


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_example_dir(example):
    example_dir = "examples/{0}".format(example)

    # TODO - check if you are on travis or not...
    returncode = subprocess.call(args=["python", "./modelzoo/__main__.py", "test",
                                       "--install-req", example_dir])
    assert returncode == 0
