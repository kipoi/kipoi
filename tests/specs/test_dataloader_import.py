"""
"""
import os
import shutil
import sys
import pytest
from kipoi.specs import DataLoaderImport
import kipoi
from kipoi_utils.utils import inherits_from
from uuid import uuid4


def cp_tmpdir(example, tmpdir):
    tdir = os.path.join(str(tmpdir), example, str(uuid4()))
    shutil.copytree(example, tdir)
    return tdir


def dont_test_DataLoaderImport():
    imp = DataLoaderImport(defined_as='kipoi.data.Dataset')
    a = imp.get()
    assert a == kipoi.data.Dataset
    assert inherits_from(a, kipoi.data.BaseDataLoader)


def test_parameter_overriding(tmpdir):
    if sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")

    model = "example/models/kipoi_dataloader_decorator"
    m = kipoi.get_model(cp_tmpdir(model, tmpdir), source='dir')
    dl = m.default_dataloader.init_example()
    assert dl.dummy == 10
