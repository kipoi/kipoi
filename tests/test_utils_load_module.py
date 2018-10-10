"""Test load module
"""
import kipoi
from kipoi.utils import load_obj, inherits_from, override_default_kwargs, infer_parent_class
from kipoi.data import BaseDataLoader, Dataset, AVAILABLE_DATALOADERS
import pytest


def test_import_module_fn():
    fn = load_obj("kipoi.get_model")
    assert fn == kipoi.get_model


def test_import_module_cls():
    cls = load_obj("kipoi.model.BaseModel")
    assert cls == kipoi.model.BaseModel


def test_inherits_from():
    class A(Dataset):
        pass

    class B(object):
        pass

    assert inherits_from(A, BaseDataLoader)
    assert inherits_from(A, Dataset)
    assert inherits_from(Dataset, BaseDataLoader)
    assert not inherits_from(B, BaseDataLoader)
    assert not inherits_from(B, Dataset)


def test_infer_parent_class():
    class A(Dataset):
        pass

    class B(object):
        pass

    assert 'Dataset' == infer_parent_class(A, AVAILABLE_DATALOADERS)
    assert infer_parent_class(B, AVAILABLE_DATALOADERS) is None


def test_override_default_args():
    def fn(a, b=2):
        return a, b
    assert fn(1) == (1, 2)
    override_default_kwargs(fn, {})
    assert fn(1) == (1, 2)

    override_default_kwargs(fn, dict(b=4))
    assert fn(1) == (1, 4)
    assert fn(1, 3) == (1, 3)

    class A(object):
        def __init__(self, a, b=2):
            self.a = a
            self.b = b

        def get_values(self):
            return self.a, self.b

    assert A(1).get_values() == (1, 2)
    override_default_kwargs(A, dict(b=4))
    assert A(1).get_values() == (1, 4)
    assert A(1, 3).get_values() == (1, 3)

    with pytest.raises(ValueError):
        override_default_kwargs(A, dict(c=4))


def test_sequential_model_loading():
    m = kipoi.get_model("example/models/kipoi_dataloader_decorator", source='dir')
    m = kipoi.get_model("example/models/extended_coda", source='dir')
