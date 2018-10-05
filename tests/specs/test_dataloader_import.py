"""
"""
from kipoi.specs import DataLoaderImport
import kipoi
from kipoi.utils import inherits_from


def test_DataLoaderImport():
    imp = DataLoaderImport(defined_as='kipoi.data.Dataset')
    a = imp.get()
    assert a == kipoi.data.Dataset
    assert inherits_from(a, kipoi.data.BaseDataLoader)
