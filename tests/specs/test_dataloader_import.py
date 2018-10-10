"""
"""
import os
import shutil
import sys
import pytest
from kipoi.specs import DataLoaderImport
import kipoi
from kipoi.utils import inherits_from
from config import is_master


def dont_test_DataLoaderImport():
    imp = DataLoaderImport(defined_as='kipoi.data.Dataset')
    a = imp.get()
    assert a == kipoi.data.Dataset
    assert inherits_from(a, kipoi.data.BaseDataLoader)


def test_parameter_overriding(tmpdir):
    if sys.version_info[0] == 2:
        pytest.skip("example not supported on python 2 ")

    model = "example/models/kipoi_dataloader_decorator"
    shutil.copytree(model, str(tmpdir))
    new_model = os.path.join(str(tmpdir), model)

    m = kipoi.get_model(new_model, source='dir')
    dl = m.default_dataloader.init_example()
    assert dl.dummy == 10
