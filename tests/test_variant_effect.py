
import kipoi
from kipoi.postprocessing.variant_effects import predict_snvs
import numpy as np
import pytest
import sys
from kipoi.pipeline import install_model_requirements
import warnings
import filecmp
import config
import os
from kipoi.utils import cd

warnings.filterwarnings('ignore')

# TODO: We still need a way to get the model output annotation from somewhere...
# TODO: which other arguments should we use for variant effect predictions?
# Only viable model at the moment is rbp, so not offering to test anything else
# INSTALL_REQ = True
INSTALL_REQ = config.install_req

# TODO - check if you are on travis or not regarding the --install-req flag
INSTALL_REQ = True
# INSTALL_REQ = False


def test_var_eff_pred():
    if sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")
    # Take the rbp model
    model_dir = "examples/rbp/"
    if INSTALL_REQ:
        install_model_requirements(model_dir, "dir", and_dataloaders=True)

    model = kipoi.get_model(model_dir, source="dir")
    # The preprocessor
    Dataloader = kipoi.get_dataloader_factory(model_dir, source="dir")

    dataloader_arguments = {
        "fasta_file": "example_files/hg38_chr22.fa",
        "preproc_transformer": "dataloader_files/encodeSplines.pkl",
        "gtf_file": "example_files/gencode_v25_chr22.gtf.pkl.gz",
    }

    # Run the actual predictions
    vcf_path = "example_files/variants.vcf"
    out_vcf_fpath = "example_files/variants_generated.vcf"
    ref_out_vcf_fpath = "example_files/variants_ref_out.vcf"
    with cd(model.source_dir):
        res = predict_snvs(model, vcf_path, dataloader_args=dataloader_arguments,
                           dataloader=Dataloader, batch_size=32,
                           evaluation_function_kwargs={"diff_type": "diff"},
                           out_vcf_fpath=out_vcf_fpath)
        assert filecmp.cmp(out_vcf_fpath, ref_out_vcf_fpath)
        os.unlink(out_vcf_fpath)
