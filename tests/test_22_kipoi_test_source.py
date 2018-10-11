"""test: kipoi test-source
"""
import numpy as np
import pytest
import sys
import subprocess as sp
from kipoi.cli.source_test import modified_files
from kipoi.sources import list_softlink_dependencies, LocalSource
import kipoi
import os


def test_list_softlink_dependencies():
    """Test if finding model dependencies works
    """
    component_dir = kipoi.get_source("kipoi").local_path
    assert list_softlink_dependencies(os.path.join(component_dir, 'rbp_eclip/UPF1'),
                                      component_dir) == {'rbp_eclip/template'}
    assert list_softlink_dependencies(os.path.join(component_dir, 'HAL'),
                                      component_dir) == {'MaxEntScan/template',
                                                         'MaxEntScan/template/example_files',
                                                         'labranchor/example_files'}
    assert list_softlink_dependencies(os.path.join(component_dir, 'deepTarget'),
                                      component_dir) == set()


def dont_test_diff():
    git_range = ["master", "HEAD"]
    local_path = "/home/avsec/.kipoi/models"
    modified_files(["master", "HEAD"], "/home/avsec/.kipoi/models", relative=True)

    sp.call(['git', 'diff', '--relative=/home/avsec/.kipoi/models',
             '--name-only', 'master...HEAD',
             '--', '/home/avsec/.kipoi/models/*', '/home/avsec/.kipoi/models/*/*'])


def test_single_model_dry():
    # Dry run
    returncode = sp.call(["python", os.path.abspath("./kipoi/__main__.py"),
                          "test-source",
                          "kipoi",
                          "--git-range", "master", "HEAD",
                          "-n"])

    assert returncode == 0


def test_single_model():
    MODEL = "HAL"
    returncode = sp.call(["python", os.path.abspath("./kipoi/__main__.py"),
                          "test-source",
                          "kipoi",
                          "--git-range", "master", "HEAD",
                          "--all",
                          "-x",  # Exit immediately
                          "-c",  # clean environment
                          "-k", MODEL])
    assert returncode == 0


@pytest.fixture
def source():
    source_dir = 'example/models'
    return LocalSource(source_dir)


source = source()  # TODO - remove

# MODEL = 'multiple_models'

def test_list_components(source):
    # 1. list
    # 2. get each model
    # 3. check that the arguments were set correctly (doc as well as the resize len)
    # 4. check that the made prediction is correct

    # 1. instantiate the source
    l = source._list_components("model")

    # standard models
    assert 'pyt' in l

    # group models
    assert 'multiple_models/model1' in l
    assert 'multiple_models/submodel/model2' in l

    # dataloader
    l = source._list_components("dataloader")

    # standard models
    assert 'pyt' in l

    # group models
    assert 'multiple_models/model1' in l
    assert 'multiple_models/submodel/model2' in l


def test_is_component(source):
    # _is_component
    assert source._is_component("pyt", 'model')
    assert source._is_component("pyt", 'dataloader')

    assert not source._is_component("multiple_models", 'model')
    assert not source._is_component("multiple_models", 'dataloader')

    assert source._is_component("multiple_models/model1", 'model')
    assert source._is_component("multiple_models/model1", 'dataloader')

    assert source._is_component("multiple_models/submodel/model2", 'model')
    assert source._is_component("multiple_models/submodel/model2", 'dataloader')


def test_pull_component(source):
    assert source._pull_component("pyt", 'model') == os.path.join(source.local_path, "pyt/model.yaml")
    assert source._pull_component("pyt", 'dataloader') == os.path.join(source.local_path, "pyt/dataloader.yaml")

    # group component
    assert source._pull_component("multiple_models/model1", 'model') is None
    assert source._pull_component("multiple_models/model1", 'dataloader') is None

    assert source._pull_component("multiple_models/submodel/model2", 'model') is None
    assert source._pull_component("multiple_models/submodel/model2", 'dataloader') is None
    # TODO - make sure you can download anything that's there


def test_get_component_descr(source):
    assert source._get_component_descr("pyt", 'model').info.doc  # model has some description
    assert source._get_component_descr("pyt", 'dataloader').info.doc  # dataloader has some description

    # test overriding
    assert source._get_component_descr("multiple_models/model1", 'model').info.doc == "model returning one"
    assert source._get_component_descr("multiple_models/submodels/model2", 'model').info.doc == "model returning two"

    # test placeholders
    assert source._get_component_descr("multiple_models/model1", 'model').schema.inputs.doc == "sequence one"
    assert source._get_component_descr("multiple_models/submodels/model2", 'model').schema.inputs.doc == "sequence two"


def test_get_model(source):
    # model correctly instentiated
    assert kipoi.get_model("pyt", source).info.doc

    assert kipoi.get_model("multiple_models/model1", source).dummy == 1
    assert kipoi.get_model("multiple_models/submodels/model2", source).dummy == 2

    # model examples correctly performed
    m = kipoi.get_model("multiple_models/model1", source)
    assert np.all(m.pipeline.predict_example()['targets'] == 1)

    m = kipoi.get_model("multiple_models/submodels/model2", source)
    assert np.all(m.init_example().load_all()['targets'] == 2)


def test_list_models(source):
    df = source.list_models()
    assert df.model.str.contains("pyt").any()

    assert not df.model.str.contains("multiple_models").any()
    assert df.model.str.contains("multiple_models/model1").any()
    assert df.model.str.contains("multiple_models/submodels/model2").any()
