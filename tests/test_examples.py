"""Run the example scripts
"""

import pytest
import subprocess
import sys
import os
import deepdish
import modelzoo
import yaml

EXAMPLES_TO_RUN = ["rbp", "extended_coda"]


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_example_dir(example):
    if example == "rbp" and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    example_dir = "examples/{0}".format(example)

    # TODO - check if you are on travis or not regarding the --install-req flag
    returncode = subprocess.call(args=["python", "./modelzoo/__main__.py", "test",
                                       "--batch_size=4",
                                       "--install-req",
                                       example_dir])
    assert returncode == 0


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_example_to_hdf5(example, tmpdir):
    if example == "rbp" and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    example_dir = "examples/{0}".format(example)

    print(example)
    print("tmpdir: {0}".format(tmpdir))
    tmpfile = str(tmpdir.mkdir("example").join("out.h5"))

    # run the
    returncode = subprocess.call(args=["python", os.path.abspath("./modelzoo/__main__.py"), "preproc",
                                       "../",  # directory
                                       "--batch_size=4",
                                       "--extractor_args=test.json",
                                       "--install-req",
                                       "--output", tmpfile],
                                 cwd=os.path.realpath(example_dir + "/test_files"))
    assert returncode == 0

    assert os.path.exists(tmpfile)

    data = deepdish.io.load(tmpfile)

    with open(example_dir + "/extractor.yaml", "r") as f:
        ex_descr = yaml.load(f)

    assert data["inputs"].keys() == ex_descr["extractor"]["output"]["inputs"].keys()

    # TODO - add size unit-tests?
