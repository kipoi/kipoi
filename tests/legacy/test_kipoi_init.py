"""Test the kipoi init examples

It setus up a model for all major combinations of `kipoi init` arguments
and tests the model validity with `kipoi test .`
"""
import pytest
import logging
from kipoi.cli.main import cli_init, cli_test

# Build up all configs
test_all = [(m, inp, inp, dl)
            for m in ["custom", "keras"]
            for inp in ["np.array", "list of np.arrays", "dict of np.arrays"]
            for dl in ["Dataset", "PreloadedDataset", "BatchDataset", "SampleIterator",
                       "SampleGenerator", "BatchIterator", "BatchGenerator"]] + \
           [("sklearn", "np.array", "np.array", "Dataset")]


@pytest.mark.parametrize("model_type,model_input_type,model_output_type,dataloader_type", test_all)
def test_all_args(model_type,
                  model_input_type,
                  model_output_type,
                  dataloader_type,
                  tmpdir,
                  caplog):
    caplog.set_level(logging.INFO)
    
    # initialize the repo
    cli_init("init", "",
             no_input=True,
             extra_context={"model_name": "test_model",
                            "model_type": model_type,
                            "model_input_type": model_input_type,
                            "model_output_type": model_output_type,
                            "dataloader_type": dataloader_type},
             output_dir=str(tmpdir)
             )
    # Test that everything worked
    cli_test("test", [str(tmpdir) + "/test_model"])

    # check that there were no warnings in the logs
    for record in caplog.records:
        assert record.levelname not in ['ERROR', 'CRITICAL']
