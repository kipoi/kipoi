
import kipoi
from kipoi.postprocessing.variant_effects import predict_variants
import numpy as np
import pytest
import sys
from kipoi.pipeline import install_model_requirements
import warnings
import filecmp
import config
import os

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

    # Hacky: take the example arguments
    import yaml
    with open(model_dir + "test_files/test.json", "r") as f:
        exec_files_path = yaml.load(f)

    for k in exec_files_path:
        exec_files_path[k] = model_dir + "test_files/" + exec_files_path[k]

    exec_files_path_here = {}
    for k in exec_files_path:
        if k != "target_file":
            exec_files_path_here[k] = exec_files_path[k]

    # Run the actual predictions
    vcf_path = model_dir + "test_files/variants.vcf"
    out_vcf_fpath = model_dir + "test_files/variants_generated.vcf"
    ref_out_vcf_fpath = model_dir + "test_files/variants_ref_out.vcf"
    res = predict_variants(model, vcf_path, exec_files_path=exec_files_path_here,
                           dataloader=Dataloader, batch_size=32,
                           evaluation_function_kwargs={"diff_type": "diff"},
                           out_vcf_fpath=out_vcf_fpath)

    assert filecmp.cmp(out_vcf_fpath, ref_out_vcf_fpath)
    os.unlink(out_vcf_fpath)
