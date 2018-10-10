"""Run the example scripts
"""
import pytest
import subprocess
import sys
import os
import yaml
import pandas as pd
import config
# import filecmp
import kipoi
from kipoi.readers import HDF5Reader
import numpy as np
from utils import cp_tmpdir

if config.install_req:
    INSTALL_FLAG = "--install_req"
else:
    INSTALL_FLAG = ""

EXAMPLES_TO_RUN = ["rbp", "extended_coda", "sklearn_iris", "iris_model_template",
                   "non_bedinput_model", "pyt", "iris_tensorflow", "kipoi_dataloader_decorator"]

predict_activation_layers = {
    "rbp": "concatenate_6",
    "pyt": "3"  # two before the last layer
}
ACTIVATION_EXAMPLES = ['rbp', 'pyt']


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_test_example(example, tmpdir):
    """kipoi test ...
    """
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} \
            and sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")

    example_dir = cp_tmpdir("example/models/{0}".format(example), tmpdir)

    args = ["python", "./kipoi/__main__.py", "test",
            "--batch_size=4",
            example_dir]
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)
    returncode = subprocess.call(args=args)
    assert returncode == 0

    if example == 'pyt':
        kipoi.cli.main.cli_test("test", args[3:])


def test_postproc_cli_fail():
    """kipoi test ...
    """
    # This command should fail
    args = ["python", "./kipoi/__main__.py", "postproc", "score_variants"]
    returncode = subprocess.call(args=args)
    assert returncode > 0

    args = ["python", "./kipoi/__main__.py", "other"]
    returncode = subprocess.call(args=args)
    assert returncode > 0


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_preproc_example(example, tmpdir):
    """kipoi preproc ...
    """
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")
    if example in {"extended_coda", "kipoi_dataloader_decorator"}:
        # extended_coda will anyway be tested in models
        pytest.skip("randomly failing on circleci without any reason. Skipping this test.")

    example_dir = cp_tmpdir("example/models/{0}".format(example), tmpdir)
    # example_dir = "example/models/{0}".format(example)

    tmpfile = str(tmpdir.mkdir("output", ).join("out.h5"))

    # run the
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "preproc",
            "../",  # directory
            "--source=dir",
            "--batch_size=4",
            "--num_workers=2",
            "--dataloader_args=test.json",
            "--output", tmpfile]
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)
    returncode = subprocess.call(args=args,
                                 cwd=os.path.realpath(example_dir + "/example_files"))

    assert returncode == 0

    assert os.path.exists(tmpfile)

    data = HDF5Reader.load(tmpfile)

    with open(example_dir + "/dataloader.yaml", "r") as f:
        ex_descr = yaml.load(f)

    if example not in {"pyt", "sklearn_iris"}:
        assert data["inputs"].keys() == ex_descr["output_schema"]["inputs"].keys()

    if example == 'pyt':
        args[-1] = tmpfile + "2.h5"
        with kipoi.utils.cd(os.path.join(example_dir, "example_files")):
            kipoi.cli.main.cli_preproc("preproc", args[3:])


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_predict_example(example, tmpdir):
    """kipoi predict ...
    """
    # TODO - test -out
    # Traceback (most recent call last):
    #   File "/home/avsec/projects-work/kipoi/kipoi/__main__.py", line 60, in <module>
    #     main()
    #   File "/home/avsec/projects-work/kipoi/kipoi/__main__.py", line 56, in main
    #     command_fn(args.command, sys.argv[2:])
    #   File "/home/avsec/bin/anaconda3/lib/python3.6/site-packages/kipoi/pipeline.py", line 273, in cli_predict
    #     pred_batch = model.predict_on_batch(batch['inputs'])
    #   File "/home/avsec/bin/anaconda3/lib/python3.6/site-packages/kipoi/model.py", line 22, in predict_on_batch
    #     raise NotImplementedError
    # NotImplementedError
    # _________________________
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    if example in {'kipoi_dataloader_decorator'}:
        pytest.skip("Automatically-dowloaded input files skipped for prediction")

    example_dir = cp_tmpdir("example/models/{0}".format(example), tmpdir)
    # example_dir = "example/models/{0}".format(example)

    if example == "rbp":
        file_format = "tsv"
    else:
        file_format = "hdf5"

    print(example)
    print("tmpdir: {0}".format(tmpdir))
    tmpfile = str(tmpdir.mkdir("output").join("out.{0}".format(file_format)))

    # run the
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "predict",
            "../",  # directory
            "--source=dir",
            "--batch_size=4",
            "--num_workers=2",
            "--dataloader_args=test.json",
            "--output", tmpfile]
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)
    returncode = subprocess.call(args=args,
                                 cwd=os.path.realpath(example_dir + "/example_files"))
    assert returncode == 0

    assert os.path.exists(tmpfile)

    if file_format == "hdf5":
        data = HDF5Reader.load(tmpfile)
        assert {'metadata', 'preds'} <= set(data.keys())
    else:
        data = pd.read_csv(tmpfile, sep="\t")
        assert list(data.columns) == ['metadata/ranges/chr',
                                      'metadata/ranges/end',
                                      'metadata/ranges/id',
                                      'metadata/ranges/start',
                                      'metadata/ranges/strand',
                                      'preds/0']
    if example == 'pyt':
        args[-1] = tmpfile + "out2.{0}".format(file_format)
        with kipoi.utils.cd(os.path.join(example_dir, "example_files")):
            kipoi.cli.main.cli_predict("predict", args[3:])


@pytest.mark.parametrize("example", ACTIVATION_EXAMPLES)
def test_predict_activation_example(example, tmpdir):
    """Kipoi predict --layer=x with a specific output layer specified
    """
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")
    if example in {'kipoi_dataloader_decorator'}:
        pytest.skip("Automatically-dowloaded input files skipped for prediction")

    example_dir = cp_tmpdir("example/models/{0}".format(example), tmpdir)
    # example_dir = "example/models/{0}".format(example)

    print(example)
    print("tmpdir: {0}".format(tmpdir))
    tmpfile = str(tmpdir.mkdir("output").join("out.h5"))

    # run the
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "predict",
            "../",  # directory
            "--source=dir",
            "--layer", predict_activation_layers[example],
            "--batch_size=4",
            "--num_workers=2",
            "--dataloader_args=test.json",
            "--output", tmpfile]
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)
    returncode = subprocess.call(args=args,
                                 cwd=os.path.realpath(example_dir + "/example_files"))
    assert returncode == 0

    assert os.path.exists(tmpfile)

    data = HDF5Reader.load(tmpfile)
    assert {'metadata', 'preds'} <= set(data.keys())
    if example == 'pyt':
        args[-1] = tmpfile + "2.h5"
        with kipoi.utils.cd(os.path.join(example_dir, "example_files")):
            kipoi.cli.main.cli_predict("predict", args[3:])


def test_kipoi_pull():
    """Test that pull indeed pulls the right model
    """
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "pull",
            "rbp_eclip/AARS"]
    returncode = subprocess.call(args=args)
    assert returncode == 0
    assert os.path.exists(os.path.expanduser('~/.kipoi/models/rbp_eclip/AARS/model.yaml'))
    assert os.path.exists(os.path.expanduser('~/.kipoi/models/rbp_eclip/AARS/model_files/model.h5'))

    kipoi.cli.main.cli_pull("pull", ["rbp_eclip/AARS"])


def test_kipoi_info():
    """Test that pull indeed pulls the right model
    """
    if sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "info",
            "rbp_eclip/AARS"]
    returncode = subprocess.call(args=args)
    assert returncode == 0
