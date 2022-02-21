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

def test_singularity_non_kipoi_src_fail():
    returncode = sp.call(["python", os.path.abspath("./kipoi/__main__.py"),
                          "test-source",
                          "dir",
                          "--all",
                          "-x",
                          "--singularity"]
                          )

    assert returncode == 1

def test_singularity_commonenv_together_fail():
    returncode = sp.call(["python", os.path.abspath("./kipoi/__main__.py"),
                          "test-source",
                          "kipoi",
                          "--all",
                          "-x",
                          "--singularity",
                          "--common_env"]
                          )

    assert returncode == 1


def test_list_softlink_dependencies():
    """Test if finding model dependencies works
    """
    component_dir = kipoi.get_source("kipoi").local_path
    deps = list_softlink_dependencies(os.path.join(component_dir, 'HAL'),
                                      component_dir)
    # one of these two, depending on the model source
    assert (deps == {'MaxEntScan'}) or (deps == {'MaxEntScan/template',
                                                 'MaxEntScan/template/example_files',
                                                 'labranchor/example_files'})
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
    try:
        proc = sp.Popen(["python", os.path.abspath("./kipoi/__main__.py"), "test-source",
                          "kipoi",
                          "--git-range", "master", "HEAD",
                          "--all",
                          "-x",
                          "-c",
                          f"-k {MODEL}"], stdout=sp.PIPE, stderr=sp.PIPE)
        proc.wait()
        stdout, stderr = proc.communicate()
    except sp.CalledProcessError as err:
        print(f"Error: {err.stderr}")

def test_single_model_singularity():
    MODEL = "epidermal_basset"
    try:
        proc = sp.Popen(["python", os.path.abspath("./kipoi/__main__.py"), "test-source",
                          "kipoi",
                          "--all",
                          "-x",
                          "--singularity",
                          f"-k {MODEL}"], stdout=sp.PIPE, stderr=sp.PIPE)
        proc.wait()
        stdout, stderr = proc.communicate()
    except sp.CalledProcessError as err:
        print(f"Error: {err.stderr}")


@pytest.fixture
def source():
    source_dir = 'example/models'
    return LocalSource(source_dir)


# source = source()  # TODO - remove

# MODEL = 'multiple_models'


def test_list_components(source):
    # 1. list
    # 2. get each model
    # 3. check that the arguments were set correctly (doc as well as the resize len)
    # 4. check that the made prediction is correct

    # 1. instantiate the source
    ls = source._list_components("model")

    # standard models
    assert 'pyt' in ls

    # group models
    assert 'multiple_models/model1' in ls
    assert 'multiple_models/submodel/model2' in ls

    # dataloader
    ls = source._list_components("dataloader")

    # standard models
    assert 'pyt' in ls

    # group dataloader - not present
    assert 'multiple_models/model1' not in ls
    assert 'multiple_models/submodel/model2' not in ls


def test_is_component(source):
    # _is_component
    assert source._is_component("pyt", 'model')
    assert source._is_component("pyt", 'dataloader')

    assert not source._is_component("multiple_models", 'model') 
    assert not source._is_component("multiple_models", 'dataloader')

    assert source._is_component("multiple_models/model1", 'model')
    assert not source._is_component("multiple_models/model1", 'dataloader')

    assert not source._is_component("multiple_models", 'model') 
    assert not source._is_component("multiple_models", 'dataloader')

    assert source._is_component("multiple_models/submodel/model2", 'model')
    assert not source._is_component("multiple_models/submodel/model2", 'dataloader')


def test_pull_component(source):
    assert source._get_component_dir("pyt", 'model') == os.path.join(source.local_path, "pyt")
    assert source._get_component_dir("pyt", 'dataloader') == os.path.join(source.local_path, "pyt")

    # group component
    assert source._get_component_dir("multiple_models/model1", 'model') == os.path.join(source.local_path,
                                                                                        "multiple_models")
    with pytest.raises(ValueError):
        source._get_component_dir("multiple_models/model1", 'dataloader') is None

    assert source._get_component_dir("multiple_models/submodel/model2", 'model') == \
           os.path.join(source.local_path, "multiple_models")


def test_get_component_descr(source):
    assert source._get_component_descr("pyt", 'model').info.doc  # model has some description
    assert source._get_component_descr("pyt", 'dataloader').info.doc  # dataloader has some description

    # test overriding
    assert source._get_component_descr("multiple_models/model1", 'model').info.doc == "model returning one"
    assert source._get_component_descr("multiple_models/submodel/model2", 'model').info.doc == "model returning two"

    # test placeholders
    assert source._get_component_descr("multiple_models/model1", 'model').schema.inputs.doc == "sequence one"
    assert source._get_component_descr("multiple_models/submodel/model2", 'model').schema.inputs.doc == "sequence two"


def test_get_model(source):
    # model correctly instentiated
    assert kipoi.get_dataloader_factory("pyt", source).info.doc
    assert kipoi.get_model("pyt", source).info.doc

    assert kipoi.get_model("multiple_models/model1", source).dummy_add == 1
    assert kipoi.get_model("multiple_models/submodel/model2", source).dummy_add == 2

    # model examples correctly performed
    m = kipoi.get_model("multiple_models/model1", source)
    assert np.all(m.pipeline.predict_example() == 1)

    m = kipoi.get_model("multiple_models/submodel/model2", source)
    assert np.all(m.pipeline.predict_example() == 2)


def test_list_models(source):
    df = source.list_models()
    assert "pyt" in list(df.model)
    assert "multiple_models" not in list(df.model)
    assert "multiple_models/model1" in list(df.model)
    assert "multiple_models/submodel/model2" in list(df.model)


def test_loading_target(source):
    # tests that the column names
    # were loaded correctly
    md = kipoi.get_model_descr("Basset")
    assert len(md.schema.targets.column_labels) > 1
