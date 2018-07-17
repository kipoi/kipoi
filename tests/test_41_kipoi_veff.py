"""Test kipoi_veff plugin
"""
import pytest
import pandas as pd
import sys
import os
import subprocess
import config
from kipoi.readers import HDF5Reader
from utils import compare_vcfs

if config.install_req:
    INSTALL_FLAG = "--install_req"
else:
    INSTALL_FLAG = ""


@pytest.mark.parametrize("file_format", ["hdf5"])
def test_predict_variants_example_multimodel(file_format, tmpdir):
    """kipoi predict ...
    """
    import json
    if sys.version_info[0] == 2:
        pytest.skip("Only rbp example testable at the moment, which only runs on py3")

    example_dir = "example/models/non_bedinput_model/"

    tmpdir_here = tmpdir.mkdir("example")

    # non_bedinput_model is not compatible with restricted bed files as
    # alterations in region generation have no influence on that model
    tmpfile = str(tmpdir_here.join("out.{0}".format(file_format)))
    vcf_tmpfile = str(tmpdir_here.join("out.{0}".format("vcf")))

    dataloader_kwargs = {"fasta_file": "example_files/hg38_chr22.fa",
                         "preproc_transformer": "dataloader_files/encodeSplines.pkl",
                         "gtf_file": "example_files/gencode_v25_chr22.gtf.pkl.gz",
                         "intervals_file": "example_files/variant_intervals.tsv"}
    dataloader_kwargs = {k: example_dir + v for k, v in dataloader_kwargs.items()}
    dataloader_kwargs_str = json.dumps(dataloader_kwargs)

    args = ["python", os.path.abspath("./kipoi/__main__.py"), "veff",
            "score_variants",
            # "./",  # directory
            example_dir,
            "--source=dir",
            "--batch_size=4",
            "--dataloader_args='%s'" % dataloader_kwargs_str,
            "--input_vcf", example_dir + "/example_files/variants.vcf",
            # this one was now gone in the master?!
            "--output_vcf", vcf_tmpfile,
            "--std_var_id",
            "--extra_output", tmpfile]
    # run the
    if INSTALL_FLAG:
        args.append(INSTALL_FLAG)

    returncode = subprocess.call(args=args, cwd=".")
    assert returncode == 0

    assert os.path.exists(tmpfile)

    # assert filecmp.cmp(example_dir + "/example_files/variants_ref_out.vcf", vcf_tmpfile)
    # import pdb
    # pdb.set_trace()

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
