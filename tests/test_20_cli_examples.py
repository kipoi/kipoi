"""Run the example scripts
"""
import pytest
import subprocess
import sys
import os
import yaml
import pandas as pd
import config
import h5py
import filecmp
from utils import compare_vcfs
from kipoi.readers import HDF5Reader

# TODO - check if you are on travis or not regarding the --install_req flag
if config.install_req:
    INSTALL_FLAG = "--install_req"
else:
    INSTALL_FLAG = ""

EXAMPLES_TO_RUN = ["rbp", "extended_coda", "iris_model_template",
                   "non_bedinput_model", "pyt", "iris_tensorflow"]

predict_activation_layers = {
    "rbp": "concatenate_6",
    "pyt": "3"  # two before the last layer
}


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_test_example(example):
    """kipoi test ...
    """
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} \
       and sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")

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
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")
    if example in {"extended_coda"}:
        # extended_coda will anyway be tested in models
        pytest.skip("randomly failing on circleci without any reason. Skipping this test.")

    example_dir = "examples/{0}".format(example)

    tmpfile = str(tmpdir.mkdir("example").join("out.h5"))

    # run the
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "preproc",
            "../",  # directory
            "--source=dir",
            "--batch_size=4",
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

    if example != "pyt":
        assert data["inputs"].keys() == ex_descr["output_schema"]["inputs"].keys()


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


@pytest.mark.parametrize("example", list(predict_activation_layers))
def test_predict_activation_example(example, tmpdir):
    """Kipoi predict --layer=x with a specific output layer specified
    """
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    example_dir = "examples/{0}".format(example)

    print(example)
    print("tmpdir: {0}".format(tmpdir))
    tmpfile = str(tmpdir.mkdir("example").join("out.h5"))

    # run the
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "predict",
            "../",  # directory
            "--source=dir",
            "--layer", predict_activation_layers[example],
            "--batch_size=4",
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


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_predict_variants_example(example, tmpdir):
    """kipoi predict ...
    """
    if (example not in {"rbp", "non_bedinput_model"}) or (sys.version_info[0] == 2):
        pytest.skip("Only rbp example testable at the moment, which only runs on py3")

    example_dir = "examples/{0}".format(example)

    tmpdir_here = tmpdir.mkdir("example")

    for restricted_bed in [True, False]:
        # non_bedinput_model is not compatible with restricted bed files as
        # alterations in region generation have no influence on that model
        if restricted_bed and (example != "rbp"):
            continue
        for file_format in ["tsv", "hdf5"]:
            print(example)
            print("tmpdir: {0}".format(tmpdir))
            tmpfile = str(tmpdir_here.join("out.{0}".format(file_format)))
            vcf_tmpfile = str(tmpdir_here.join("out.{0}".format("vcf")))

            dataloader_kwargs = {"fasta_file": example_dir + "/example_files/hg38_chr22.fa",
                                 "preproc_transformer": example_dir + "/dataloader_files/encodeSplines.pkl",
                                 "gtf_file": example_dir + "/example_files/gencode_v25_chr22.gtf.pkl.gz",
                                 "intervals_file": example_dir + "/example_files/variant_intervals.tsv"}
            import json
            dataloader_kwargs_str = json.dumps(dataloader_kwargs)

            args = ["python", os.path.abspath("./kipoi/__main__.py"),
                    "postproc",
                    "score_variants",
                    #"./",  # directory
                    example_dir,
                    "--source=dir",
                    "--batch_size=4",
                    "--dataloader_args='%s'" % dataloader_kwargs_str,
                    "--vcf_path", example_dir + "/" + "example_files/variants.vcf",
                    # this one was now gone in the master?!
                    "--out_vcf_fpath", vcf_tmpfile,
                    "--output", tmpfile]
            # run the
            if INSTALL_FLAG:
                args.append(INSTALL_FLAG)

            if restricted_bed:
                args += ["--restriction_bed", example_dir + "/" + "example_files/restricted_regions.bed"]

            returncode = subprocess.call(args=args,
                                         cwd=os.path.realpath(example_dir) + "/../../")
            assert returncode == 0

            assert os.path.exists(tmpfile)
            assert os.path.exists(vcf_tmpfile)

            if restricted_bed:
                # assert filecmp.cmp(example_dir + "/example_files/variants_ref_out2.vcf", vcf_tmpfile)
                compare_vcfs(example_dir + "/example_files/variants_ref_out2.vcf", vcf_tmpfile)
            else:
                # assert filecmp.cmp(example_dir + "/example_files/variants_ref_out.vcf", vcf_tmpfile)
                compare_vcfs(example_dir + "/example_files/variants_ref_out.vcf", vcf_tmpfile)

            if file_format == "hdf5":
                data = HDF5Reader.load(tmpfile)
            else:
                table_labels = []
                table_starts = []
                table_ends = []
                tables = {}
                head_line_id = "KPVEP_"
                with open(tmpfile, "r") as ifh:
                    for i, l in enumerate(ifh):
                        if head_line_id in l:
                            if (len(table_starts) > 0):
                                table_ends.append(i - 1)
                            table_labels.append(l.rstrip()[len(head_line_id):])
                            table_starts.append(i + 1)
                    table_ends.append(i)
                for label, start, end in zip(table_labels, table_starts, table_ends):
                    tables[label] = pd.read_csv(tmpfile, sep="\t", skiprows=start, nrows=end - start, index_col=0)


def test_pull_kipoi():
    """Test that pull indeed pulls the right model
    """
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "pull",
            "rbp_eclip/AARS"]
    returncode = subprocess.call(args=args)
    assert returncode == 0
    assert os.path.exists(os.path.expanduser('~/.kipoi/models/rbp_eclip/AARS/model.yaml'))
    assert os.path.exists(os.path.expanduser('~/.kipoi/models/rbp_eclip/AARS/model_files/model.h5'))
