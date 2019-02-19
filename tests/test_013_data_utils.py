"""Test data_utils
"""

import pytest
from kipoi.data import PreloadedDataset
import numpy as np


def test_preloaded_dataset(data):
    def data_fn():
        return data

    d = PreloadedDataset.from_fn(data_fn)()

    assert d.load_all() == data
    assert len(d) == 3
    assert d[1] == {"a": [1], "b": {"d": 1}, "c": np.array([1])}
    assert list(d.batch_iter(2))[1] == {'a': [np.array([2])], 'b': {'d': np.array([2])}, 'c': np.array([[2]])}
