"""Run the example scripts
"""
import os
import subprocess
import sys

import pandas as pd
import pytest
import yaml

import config
# import filecmp
import kipoi
import kipoi_conda
import kipoi_utils
from kipoi.env_db import EnvDbEntry
from kipoi.readers import HDF5Reader
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


def test_cli_get_example(tmpdir):
    """kipoi test ..., add also output file writing
    """
    example = "kipoi_dataloader_decorator"
    example_dir = "example/models/{0}".format(example)

    outdir = os.path.join(str(tmpdir), example)
    args = ["python", "./kipoi/__main__.py", "get-example",
            example_dir,
            "--source", 'dir',
            "-o", outdir]
    kipoi.cli.main.cli_get_example("get-example", args[3:])
    assert os.path.exists(os.path.join(outdir, "targets_file"))


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_test_example(example, tmpdir):
    """kipoi test ..., add also output file writing
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
        # python interface, write also the output file
        output_file = os.path.join(example_dir, 'preds.h5')
        kipoi.cli.main.cli_test("test", args[3:] + ["-o", output_file])

        assert os.path.exists(output_file)
        preds = HDF5Reader.load(output_file)
        assert 'inputs' in preds
        assert 'metadata' in preds
        assert 'preds' in preds


def test_cli_test_expect(tmpdir):
    """kipoi test - check that the expected predictions also match
    """
    example = 'pyt'
    example_dir = cp_tmpdir("example/models/{0}".format(example), tmpdir)

    # fail the test
    args = ["python", "./kipoi/__main__.py", "test",
            "--batch_size=4",
            "-e", os.path.join(example_dir, "wrong.pred.h5"),
            example_dir]
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)
    returncode = subprocess.call(args=args)
    assert returncode == 1

    # succeed
    kipoi.cli.main.cli_test("test", ["--batch_size=4",
                                     "-e", os.path.join(example_dir, "expected.pred.h5"),
                                     example_dir])


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

@pytest.mark.parametrize("new_dataloader_kwargs_format", [False, True])
@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_preproc_example(example, new_dataloader_kwargs_format, tmpdir):
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

    if example in {"rbp"} and new_dataloader_kwargs_format:
        if example == "rbp":
            dataloader_args = [
                "intervals_file=intervals.tsv",
                "fasta_file=hg38_chr22.fa",
                "preproc_transformer=../dataloader_files/encodeSplines.pkl",
                "gtf_file=gencode_v25_chr22.gtf.pkl.gz",
                "tarOget_file=targets.tsv"
            ]
        elif example == "extended_coda":
            dataloader_args = [
                "intervals_file=intervals.tsv",
                "input_data_sources={'H3K27AC_subsampled':'H3K27AC_subsampled.bw'}",
                "batch_size=4"
            ]
        # run the
        args = ["python", os.path.abspath("./kipoi/__main__.py"), "preproc",
                "../",  # directory
                "--source=dir",
                "--batch_size=4",
                "--num_workers=2",
                "--dataloader_args"] + dataloader_args + ["--output", tmpfile]

    else:
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
        with kipoi_utils.utils.cd(os.path.join(example_dir, "example_files")):
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
        with kipoi_utils.utils.cd(os.path.join(example_dir, "example_files")):
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
        with kipoi_utils.utils.cd(os.path.join(example_dir, "example_files")):
            kipoi.cli.main.cli_predict("predict", args[3:])


def test_kipoi_pull():
    """Test that pull indeed pulls the right model
    """
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "pull",
            "rbp_eclip/AARS"]
    returncode = subprocess.call(args=args)
    assert returncode == 0
    # obsolete - not using the git-lfs source anymore
    # assert (os.path.exists(os.path.expanduser('~/.kipoi/models/rbp_eclip/downloaded/AARS/model_files/model.h5')) or
    #         os.path.exists(os.path.expanduser('~/.kipoi/models/rbp_eclip/AARS/model_files/model.h5')))

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


def assert_rec(a, b):
    if isinstance(a, dict):
        assert set(a.keys()) == set(b.keys())
        for k in a:
            assert_rec(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for a_el, b_el in zip(a, b):
            assert_rec(a_el, b_el)
    else:
        assert a == b


def process_args(args):
    raw_args = args[4:]
    cmd = " ".join(args)
    return cmd, raw_args


class PseudoConda:
    def __init__(self, tmpdir):
        self.existing_envs = {}
        self.tmpdir = tmpdir

    @staticmethod
    def strip_yaml_suffix(env):
        env = env.split("/")[-1]
        if env.endswith(".yaml"):
            return env[:-len(".yaml")]
        else:
            return env

    def add_env(self, env, **kwargs):
        env = self.strip_yaml_suffix(env)
        if env in self.existing_envs:
            return 1

        kipoi_cli_path = os.path.join(str(self.tmpdir), "kipoi_cli_" + env)
        with open(kipoi_cli_path, "w") as ofh:
            ofh.write("kipoi")
        self.existing_envs[env] = kipoi_cli_path
        return 0

    def get_cli(self, env):
        env = self.strip_yaml_suffix(env)
        if env not in self.existing_envs:
            return None
        return self.existing_envs[env]

    def delete_env(self, env):
        env = self.strip_yaml_suffix(env)
        if env in self.existing_envs:
            self.existing_envs.pop(env)
            return 0
        else:
            raise Exception("Failed")


def test_kipoi_env_create_cleanup_remove(tmpdir, monkeypatch):
    from kipoi.cli.env import cli_create, cli_cleanup, cli_remove, cli_get, cli_get_cli, cli_list
    tempfile = os.path.join(str(tmpdir), "envs.json")

    # Define things necessary for monkeypatching

    def get_assert_env(equals):
        def assert_to(val):
            assert len(val) == len(equals)
            assert all([v.create_args.env == e for v, e in zip(val, equals)])

        return assert_to

    def get_assert_env_cli(equals):
        def assert_to(val):
            assert len(val) == len(equals)
            assert all([v.cli_path == e for v, e in zip(val, equals)])

        return assert_to

    # pseudo kipoi CLI executable
    conda = PseudoConda(tmpdir)

    if os.path.exists(tempfile):
        os.unlink(tempfile)

    test_model = "example/models/pyt"
    test_env_name = "kipoi-testenv"
    source_path = kipoi.get_source("dir").local_path

    # monkeypatch:
    old_env_db_path = kipoi.config._env_db_path
    monkeypatch.setattr(kipoi.config, '_env_db_path', tempfile)
    monkeypatch.setattr(kipoi_conda, 'create_env_from_file', conda.add_env)
    monkeypatch.setattr(kipoi_conda, 'remove_env', conda.delete_env)
    monkeypatch.setattr(kipoi_conda, 'get_kipoi_bin', conda.get_cli)
    monkeypatch.setattr(kipoi.cli.env, 'print_env_names', get_assert_env([test_env_name]))
    # load the db from the new path
    kipoi.env_db.reload_model_env_db()

    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "create", "--source", "dir", "--env",
            test_env_name, test_model]

    # pretend to run the CLI
    cli_create(*process_args(args))

    # make sure the successful flag is set and the kipoi-cli exists
    kipoi.env_db.reload_model_env_db()
    db = kipoi.env_db.get_model_env_db()

    entry = db.get_entry_by_model(os.path.join(source_path, test_model))
    assert entry.successful
    assert os.path.exists(entry.cli_path)

    # add a new entry that does not exist:
    cfg = entry.get_config()
    cfg["create_args"]["env"] += "____AAAAAA_____"
    cfg["cli_path"] += "____AAAAAA_____"
    db.append(EnvDbEntry.from_config(cfg))

    # now test the get environment name and the get_kipoi_bin
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "get", "--source", "dir", test_model]
    cli_get(*process_args(args))

    monkeypatch.setattr(kipoi.cli.env, 'print_env_cli_paths', get_assert_env_cli([conda.get_cli(test_env_name)]))
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "get_bin", "--source", "dir", test_model]
    cli_get_cli(*process_args(args))

    # list environments:
    monkeypatch.setattr(kipoi.cli.env, 'print_valid_env_names', get_assert_env([test_env_name]))
    monkeypatch.setattr(kipoi.cli.env, 'print_invalid_env_names', get_assert_env([test_env_name + "____AAAAAA_____"]))
    monkeypatch.setattr(subprocess, 'call', lambda *args, **kwargs: None)
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "list"]
    cli_list(*process_args(args))

    # pretend also the first installation didn't work
    entry.successful = False
    first_config = entry.get_config()
    db.save()

    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "cleanup", "--all", '--yes']
    print(conda.existing_envs)
    print(db.entries)
    # pretend to run the CLI
    cli_cleanup(*process_args(args))

    # now
    kipoi.env_db.reload_model_env_db()
    db = kipoi.env_db.get_model_env_db()
    assert len(db.entries) == 1
    assert_rec(db.entries[0].get_config(), cfg)

    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "cleanup", "--all", "--db", '--yes']
    # pretend to run the CLI
    cli_cleanup(*process_args(args))

    kipoi.env_db.reload_model_env_db()
    db = kipoi.env_db.get_model_env_db()
    assert len(db.entries) == 0
    assert len(conda.existing_envs) == 0

    # now final test of creating and removing an environment:

    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "create", "--source", "dir", "--env",
            test_env_name, test_model]
    # pretend to run the CLI
    cli_create(*process_args(args))
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "remove", "--source", "dir", test_model, '--yes']
    cli_remove(*process_args(args))

    kipoi.env_db.reload_model_env_db()
    db = kipoi.env_db.get_model_env_db()
    assert len(db.entries) == 0
    assert len(conda.existing_envs) == 0

    # just make sure this resets after the test.
    kipoi.config._env_db_path = old_env_db_path
    kipoi.env_db.reload_model_env_db()


def test_kipoi_env_create_all(tmpdir, monkeypatch):
    from kipoi.cli.env import cli_create
    conda = PseudoConda(tmpdir)
    monkeypatch.setattr(kipoi_conda, 'create_env_from_file', conda.add_env)
    monkeypatch.setattr(kipoi_conda, 'remove_env', conda.delete_env)
    monkeypatch.setattr(kipoi_conda, 'get_kipoi_bin', conda.get_cli)

    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "create", "all"]
    # pretend to run the CLI
    cli_create(*process_args(args))


def test_kipoi_env_create_all_dry_run():
    from kipoi.cli.env import cli_create
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "env", "create", "all", "--dry-run"]
    # pretend to run the CLI
    cli_create(*process_args(args))
