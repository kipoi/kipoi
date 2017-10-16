"""Run the example scripts
"""
import pytest
import subprocess
import sys
import os
import deepdish
import yaml
import pandas as pd

# TODO - check if you are on travis or not regarding the --install-req flag
INSTALL_FLAG = "--install-req"
# INSTALL_FLAG = ""

EXAMPLES_TO_RUN = ["rbp", "extended_coda"]


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_test_example(example):
    """kipoi test ...
    """
    if example == "rbp" and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    example_dir = "examples/{0}".format(example)

    args = ["python", "./kipoi/__main__.py", "test",
            "--batch_size=4",
            example_dir]
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)
    returncode = subprocess.call(args=args)
    assert returncode == 0


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_preproc_example(example, tmpdir):
    """kipoi preproc ...
    """
    if example == "rbp" and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    example_dir = "examples/{0}".format(example)

    tmpfile = str(tmpdir.mkdir("example").join("out.h5"))

    # run the
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "preproc",
            "../",  # directory
            "--source=dir",
            "--batch_size=4",
            "--extractor_args=test.json",
            "--output", tmpfile]
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)
    returncode = subprocess.call(args=args,
                                 cwd=os.path.realpath(example_dir + "/test_files"))

    assert returncode == 0

    assert os.path.exists(tmpfile)

    data = deepdish.io.load(tmpfile)

    with open(example_dir + "/extractor.yaml", "r") as f:
        ex_descr = yaml.load(f)

    assert data["inputs"].keys() == ex_descr["extractor"]["output"]["inputs"].keys()


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_predict_example(example, tmpdir):
    """kipoi predict ...
    """
    if example == "rbp" and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    example_dir = "examples/{0}".format(example)

    if example == "rbp":
        file_format = "tsv"
    else:
        file_format = "hdf5"

    print(example)
    print("tmpdir: {0}".format(tmpdir))
    tmpfile = str(tmpdir.mkdir("example").join("out.{0}".format(file_format)))

    # run the
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "predict",
            "../",  # directory
            "--source=dir",
            "--batch_size=4",
            "--extractor_args=test.json",
            "--file_format", file_format,
            "--output", tmpfile]
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)
    returncode = subprocess.call(args=args,
                                 cwd=os.path.realpath(example_dir + "/test_files"))
    assert returncode == 0

    assert os.path.exists(tmpfile)

    if file_format == "hdf5":
        data = deepdish.io.load(tmpfile)
        assert {'metadata', 'predictions'} <= set(data.keys())
    else:
        data = pd.read_csv(tmpfile, sep="\t")
        assert list(data.columns[:6]) == ['chr', 'start', 'end', 'name', 'score', 'strand']
        assert data.columns[6].startswith("y")


def test_pull_kipoi():
    """Test that pull indeed pulls the right model
    """
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "pull",
            "rbp"]
    returncode = subprocess.call(args=args)
    assert returncode == 0
    assert os.path.exists(os.path.expanduser('~/.kipoi/models/rbp/model.yaml'))
    assert os.path.exists(os.path.expanduser('~/.kipoi/models/rbp/model_files/weights.h5'))
