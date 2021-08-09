"""Test the KipoiModel loading
"""
import pytest
import kipoi
import sys
import config
import six
import kipoi
from kipoi.specs import Dependencies
from kipoi.pipeline import install_model_requirements

# HACK - prevents ImportError: dlopen: cannot load any more object with static TLS
import torch
import keras

EXAMPLES_TO_RUN = ["sklearn_iris", "iris_model_template", "pyt"]
# TODO - finish the unit-test
INSTALL_REQ = config.install_req


@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_load_model(example):
    example_dir = "example/models/{0}".format(example)

    if example in {"rbp", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")

    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir")
    m = kipoi.get_model(example_dir, source="dir")

    if isinstance(m, kipoi.model.KerasModel):
        m.arch
        m.weights
    m.info
    m.schema
    m.schema.inputs
    m.source
    m.default_dataloader
    m.model
    m.predict_on_batch


DEPENDENCY_MODEL = [("no-dep", kipoi.model.BaseModel),
                    ("keras", kipoi.model.KerasModel),
                    ("pytorch", kipoi.model.PyTorchModel),
                    ("scikit-learn", kipoi.model.SklearnModel),
                    ("tensorflow", kipoi.model.TensorFlowModel)]


@pytest.mark.parametrize("dependency,Model", DEPENDENCY_MODEL)
def test_deps(dependency, Model):
    contains = [Dependencies(pip=["bar", dependency]),
                Dependencies(conda=[dependency, "foo"]),
                Dependencies(conda=["asd::" + dependency])]

    doesnt_contain = [Dependencies(pip=["bar"]),
                      Dependencies(conda=["bar"])]

    for deps in contains:
        assert Model._sufficient_deps(deps)

    if dependency != "no-dep":
        for deps in doesnt_contain:
            assert not Model._sufficient_deps(deps)
