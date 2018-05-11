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
from utils import compare_vcfs
from kipoi.readers import HDF5Reader
import numpy as np

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

grad_inputs = {
    "rbp": "seq",
    "pyt": None
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


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
@pytest.mark.parametrize("restricted_bed", [True, False])
@pytest.mark.parametrize("file_format", ["tsv", "hdf5"])
def test_predict_variants_example(example, restricted_bed, file_format, tmpdir):
    """kipoi predict ...
    """
    if (example not in {"rbp", "non_bedinput_model"}) or (sys.version_info[0] == 2):
        pytest.skip("Only rbp example testable at the moment, which only runs on py3")

    example_dir = "examples/{0}".format(example)

    tmpdir_here = tmpdir.mkdir("example")

    # non_bedinput_model is not compatible with restricted bed files as
    # alterations in region generation have no influence on that model
    if restricted_bed and (example != "rbp"):
        pytest.skip("Resticted_bed only available for rbp_eclip")
    print(example)
    print("tmpdir: {0}".format(tmpdir))
    tmpfile = str(tmpdir_here.join("out.{0}".format(file_format)))
    vcf_tmpfile = str(tmpdir_here.join("out.{0}".format("vcf")))

    dataloader_kwargs = {"fasta_file": "example_files/hg38_chr22.fa",
                         "preproc_transformer": "dataloader_files/encodeSplines.pkl",
                         "gtf_file": "example_files/gencode_v25_chr22.gtf.pkl.gz",
                         "intervals_file": "example_files/variant_intervals.tsv"}
    import json
    dataloader_kwargs_str = json.dumps(dataloader_kwargs)

    args = ["python", os.path.abspath("./kipoi/__main__.py"),
            "postproc",
            "score_variants",
            # "./",  # directory
            example_dir,
            "--source=dir",
            "--batch_size=4",
            "--dataloader_args='%s'" % dataloader_kwargs_str,
            "--vcf_path", "example_files/variants.vcf",
            # this one was now gone in the master?!
            "--out_vcf_fpath", vcf_tmpfile,
            "--output", tmpfile]
    # run the
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)

    if restricted_bed:
        args += ["--restriction_bed", "example_files/restricted_regions.bed"]

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


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_generate_mutation_maps_example(example, tmpdir):
    """kipoi predict ...
    """
    if (example not in {"rbp"}) or (sys.version_info[0] == 2):
        pytest.skip("Only rbp example testable at the moment, which only runs on py3")

    example_dir = "examples/{0}".format(example)

    tmpdir_here = tmpdir.mkdir("example")

    # restricted_bed = False
    print(example)
    print("tmpdir: {0}".format(tmpdir))
    mm_tmpfile = str(tmpdir_here.join("out_mm.hdf5"))
    plt_tmpfile = str(tmpdir_here.join("plot.png"))

    dataloader_kwargs = {"fasta_file": "example_files/hg38_chr22.fa", "preproc_transformer": "dataloader_files/encodeSplines.pkl", "gtf_file": "example_files/gencode_v25_chr22.gtf.pkl.gz", "intervals_file": "example_files/variant_intervals.tsv"}
    import json
    dataloader_kwargs_str = json.dumps(dataloader_kwargs)

    args = ["python", os.path.abspath("./kipoi/__main__.py"),
            "postproc",
            "create_mutation_map",
            # "./",  # directory
            ".",
            "--source=dir",
            "--batch_size=4",
            "--dataloader_args='%s'" % dataloader_kwargs_str,
            "--regions_file", "example_files/first_variant.vcf",
            "--output", mm_tmpfile]
    # run the
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)

    returncode = subprocess.call(args=args,
                                 cwd=os.path.realpath(example_dir))
    assert returncode == 0

    assert os.path.exists(mm_tmpfile)

    # make the plot
    args = ["python", os.path.abspath("./kipoi/__main__.py"),
            "postproc",
            "plot_mutation_map",
            # "./",  # directory
            ".",
            "--input_file=" + mm_tmpfile,
            "--input_line=0",
            "--model_seq_input=seq",
            "--scoring_key=diff",
            "--model_output=rbp_prb",
            "--output", plt_tmpfile]

    returncode = subprocess.call(args=args,
                                 cwd=os.path.realpath(example_dir))
    assert returncode == 0

    assert os.path.exists(plt_tmpfile)

    os.unlink(mm_tmpfile)
    os.unlink(plt_tmpfile)


def test_pull_kipoi():
    """Test that pull indeed pulls the right model
    """
    args = ["python", os.path.abspath("./kipoi/__main__.py"), "pull",
            "rbp_eclip/AARS"]
    returncode = subprocess.call(args=args)
    assert returncode == 0
    assert os.path.exists(os.path.expanduser('~/.kipoi/models/rbp_eclip/AARS/model.yaml'))
    assert os.path.exists(os.path.expanduser('~/.kipoi/models/rbp_eclip/AARS/model_files/model.h5'))


def test_parse_filter_slice():
    from  kipoi.cli.postproc import parse_filter_slice
    class DummySlice():
        def __getitem__(self, key):
            return key

    assert DummySlice()[1] == parse_filter_slice("[1]")
    assert DummySlice()[::-1, ...] == parse_filter_slice("[::-1,...]")
    assert DummySlice()[..., 1:3, :7, 1:, ...] == parse_filter_slice("[..., 1:3, :7, 1:, ...]")
    assert DummySlice()[..., 1:3, :7, 1:, ...] == parse_filter_slice("(..., 1:3, :7, 1:, ...)")
    assert DummySlice()[1] == parse_filter_slice("1")
    with pytest.raises(Exception):
        parse_filter_slice("[:::2]")

@pytest.mark.parametrize("example", list(predict_activation_layers))
def test_grad_predict_example(example):
    """kipoi postproc grad ...
    """
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    example_dir = "examples/{0}".format(example)

    for file_format in ["tsv", "hdf5"]:
        print(example)
        tmpfile = os.path.realpath(str("./grad_outputs.{0}".format(file_format)))
        bedgraph_temp_file = os.path.realpath(str("./grad_x_input.bed"))

        # run the
        args = ["python", os.path.abspath("./kipoi/__main__.py"), "postproc", "grad",
                "../",  # directory
                "--source=dir",
                "--batch_size=4",
                "--dataloader_args=test.json",
                "--output", tmpfile]
        layer_args = ["--layer", predict_activation_layers[example],]
        final_layer_arg = ["--final_layer"]

        if INSTALL_FLAG:
            args.append(INSTALL_FLAG)

        for la in [layer_args, final_layer_arg]:
            returncode = subprocess.call(args=args + la, cwd=os.path.realpath(example_dir + "/example_files"))
            assert returncode == 0

            assert os.path.exists(tmpfile)

            if file_format == "hdf5":
                data = HDF5Reader.load(tmpfile)
                assert {'metadata', 'preds', 'inputs'} <= set(data.keys())
                # Here we can attempt to write a bedgraph file:
                bg_args = ["python", os.path.abspath("./kipoi/__main__.py"), "postproc", "gr_inp_to_file",
                        "../",  # directory
                        "--source=dir",
                        '--output', bedgraph_temp_file,
                        "--input_file", tmpfile]
                if grad_inputs[example] is not None:
                    bg_args += ["--model_input", grad_inputs[example]]
                returncode = subprocess.call(args=bg_args,
                                             cwd=os.path.realpath(example_dir + "/example_files"))

                assert returncode == 0
                assert os.path.exists(bedgraph_temp_file)
                os.unlink(bedgraph_temp_file)

            else:
                data = pd.read_csv(tmpfile, sep="\t")
                inputs_columns = data.columns.str.contains("inputs/")
                preds_columns = data.columns.str.contains("preds/")
                assert np.all(np.in1d(data.columns.values[preds_columns],
                                      data.columns.str.replace("inputs/", "preds/").values[inputs_columns]))
                other_cols = data.columns.values[~(preds_columns | inputs_columns)]
                expected = ['metadata/ranges/chr',
                          'metadata/ranges/end',
                          'metadata/ranges/id',
                          'metadata/ranges/start',
                          'metadata/ranges/strand']
                assert np.all(np.in1d(expected, other_cols))

            os.unlink(tmpfile)
