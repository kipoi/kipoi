"""Some small tests for kipoi-interpret
"""
import kipoi
import pytest
from pytest import raises
import numpy as np
from kipoi.components import ModelSchema
from related import from_yaml
import config
import sys


from kipoi.pipeline import install_model_requirements
import json
from tqdm import tqdm
from kipoi import writers, readers
from kipoi.cli.main import prepare_batch
import os
import kipoi_utils
from kipoi_utils.data_utils import numpy_collate
import numpy as np
import copy
from kipoi_interpret.importance_scores.gradient import Gradient

INSTALL_REQ = config.install_req
predict_activation_layers = {
    "tal1_model": "dense_1"
}


def test_score():
    example = "tal1_model"
    layer = predict_activation_layers[example]
    example_dir = "example/models/{0}".format(example)
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)

    model = kipoi.get_model(example_dir, source="dir")
    # The preprocessor
    Dataloader = kipoi.get_dataloader_factory(example_dir, source="dir")
    #
    with open(example_dir + "/example_files/test.json", "r") as ifh:
        dataloader_arguments = json.load(ifh)

    for k in dataloader_arguments:
        dataloader_arguments[k] = "example_files/" + dataloader_arguments[k]

    g = Gradient(model, None, layer=layer, avg_func="sum")

    if os.path.exists(model.source_dir + "/example_files/grads_pred.hdf5"):
        os.unlink(model.source_dir + "/example_files/grads_pred.hdf5")

    writer = writers.HDF5BatchWriter(file_path=model.source_dir + "/example_files/grads_pred.hdf5")

    with kipoi_utils.utils.cd(model.source_dir):
        dl = Dataloader(**dataloader_arguments)
        it = dl.batch_iter(batch_size=32, num_workers=0)
        # Loop through the data, make predictions, save the output
        for i, batch in enumerate(tqdm(it)):
            # make the prediction
            pred_batch = g.score(batch['inputs'])
            output_batch = batch
            output_batch["grads"] = pred_batch
            writer.batch_write(output_batch)
        writer.close()

    obj1 = readers.HDF5Reader.load(model.source_dir + "/example_files/grads_pred.hdf5")
    obj2 = readers.HDF5Reader.load(model.source_dir + "/example_files/grads.hdf5")
    kipoi_utils.utils.compare_numpy_dict(obj1, obj2)

    if os.path.exists(model.source_dir + "/example_files/grads_pred.hdf5"):
        os.unlink(model.source_dir + "/example_files/grads_pred.hdf5")
