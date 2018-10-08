"""
"""
from kipoi.specs import DataLoaderImport
import kipoi
from kipoi.utils import inherits_from


def dont_test_DataLoaderImport():
    imp = DataLoaderImport(defined_as='kipoi.data.Dataset')
    a = imp.get()
    assert a == kipoi.data.Dataset
    assert inherits_from(a, kipoi.data.BaseDataLoader)


def test_parameter_overriding():

    m = kipoi.get_model("example/models/kipoi_dataloader_decorator", source='dir')
    dl = m.default_dataloader.init_example()
    assert dl.dummy == 10
