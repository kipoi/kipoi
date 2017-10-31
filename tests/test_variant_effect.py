import sys
import warnings

import numpy as np
import pytest

import config
import kipoi
from kipoi.pipeline import install_model_requirements
from kipoi.post_processing.variant_effects import predict_variants

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

    # Derive a list model output labels
    if isinstance(model.schema.targets, dict):
        model_out_annotation = np.array(list(model.schema.targets.keys()))
    elif isinstance(model.schema.targets, list):
        model_out_annotation = np.array([x.name for x in model.schema.targets])
    else:
        # TODO - all targets need to have the keys defined
        model_out_annotation = np.array([model.schema.targets.name])

    # Run the actual predictions
    vcf_path = model_dir + "test_files/variants.vcf"
    res = predict_variants(model, vcf_path, seq_length=101, exec_files_path=exec_files_path_here,
                           dataloader_function=Dataloader, batch_size=32,
                           model_out_annotation=model_out_annotation,
                           evaluation_function_kwargs={"diff_type": "diff"})
