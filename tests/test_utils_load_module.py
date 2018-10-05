"""Test load module
"""
import kipoi
from kipoi.utils import load_obj


def test_import_module_fn():
    fn = load_obj("kipoi.get_model")
    assert fn == kipoi.get_model


def test_import_module_cls():
    cls = load_obj("kipoi.model.BaseModel")
    assert cls == kipoi.model.BaseModel
