"""Test kipoi_veff plugin
"""
import pytest
import pandas as pd
import sys
import os
import subprocess
from kipoi.readers import HDF5Reader

@pytest.mark.parametrize("file_format", ["hdf5"])
def test_predict_variants_example_multimodel(file_format, tmpdir):
    """kipoi predict ...
    """
    import json

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
   
    # kipoi-veff is no longer used for variant effect prediction. Please use https://github.com/kipoi/kipoi-veff2 directly

    returncode = subprocess.call(args=args)
    assert returncode > 0
    