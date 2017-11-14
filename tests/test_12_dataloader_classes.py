"""Test the different dataloader wrappers
"""
from __future__ import print_function, division
import numpy as np
from pytest import fixture
from kipoi.data import (PreloadedDataset, Dataset, BatchDataset,
                        SampleIterator, SampleGenerator,
                        BatchIterator, BatchGenerator)
from kipoi.data_utils import get_dataset_item


@fixture
def data():
    return {
        "inputs": {
            "i1": np.arange(10),
            "i2": np.arange(10)[::-1],
        },
        "targets": np.arange(10),
        "metadata": [
            np.arange(10),
            np.arange(10)[::-1]]
    }


def compare_arrays(a, b):
    assert np.allclose(a["inputs"]["i1"], b["inputs"]["i1"])
    assert np.allclose(a["inputs"]["i2"], b["inputs"]["i2"])
    assert np.allclose(a["targets"], b["targets"])
    assert np.allclose(a["metadata"][0], b["metadata"][0])
    assert np.allclose(a["metadata"][1], b["metadata"][1])

# --------------------------------------------


def test_PreloadedDataset(data):
    # PreloadedDataset example:
    def data_fn():
        return data
    # ------------------------

    d = PreloadedDataset.from_fn(data_fn)()

    compare_arrays(d.load_all(), data)
    it = d.batch_iter(3)
    compare_arrays(next(it), get_dataset_item(data, np.arange(3)))


def test_Dataset(data):
    # Dataset example:
    class MyDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return self.data["targets"].shape[0]

        def __getitem__(self, idx):
            return get_dataset_item(self.data, idx)
    # ------------------------

    d = MyDataset(data)

    compare_arrays(d.load_all(), data)
    it = d.batch_iter(3)
    compare_arrays(next(it), get_dataset_item(data, np.arange(3)))


def test_BatchDataset(data):
    # BatchDataset example:
    class MyBatchDataset(BatchDataset):
        def __init__(self, data, batch_size=3):
            self.data = data
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(self.data["targets"].shape[0] / self.batch_size))

        def __getitem__(self, idx):
            start = idx * self.batch_size
            end = min((idx + 1) * self.batch_size, self.data["targets"].shape[0])
            return get_dataset_item(self.data, np.arange(start, end))
    # ------------------------
    d = MyBatchDataset(data)

    compare_arrays(d.load_all(), data)
    it = d.batch_iter()
    compare_arrays(next(it), get_dataset_item(data, np.arange(3)))


def test_SampleIterator(data):
    # SampleIterator example:
    class MySampleIterator(SampleIterator):
        def __init__(self, data):
            self.data = data
            self.idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.idx >= self.data["targets"].shape[0]:
                raise StopIteration
            ret = get_dataset_item(self.data, self.idx)
            self.idx += 1
            return ret
        next = __next__
    # ------------------------

    d = MySampleIterator(data)

    compare_arrays(d.load_all(), data)
    d = MySampleIterator(data)
    it = d.batch_iter(batch_size=3)
    compare_arrays(next(it), get_dataset_item(data, np.arange(3)))


def test_BatchIterator(data):
    # BatchIterator example:
    class MyBatchIterator(BatchIterator):
        def __init__(self, data, batch_size):
            self.data = data
            self.batch_size = batch_size
            self.idx = 0

        def __iter__(self):
            return self

        def __next__(self):
            idx = self.idx
            start = idx * self.batch_size
            if start >= self.data["targets"].shape[0]:
                raise StopIteration
            end = min((idx + 1) * self.batch_size, self.data["targets"].shape[0])
            self.idx += 1
            return get_dataset_item(self.data, np.arange(start, end))
        next = __next__
    # ------------------------

    d = MyBatchIterator(data, 3)

    compare_arrays(d.load_all(), data)
    d = MyBatchIterator(data, 3)
    it = d.batch_iter()
    compare_arrays(next(it), get_dataset_item(data, np.arange(3)))


def test_SampleGenerator(data):
    # SampleGenerator example:
    def generator_fn(data):
        for idx in range(data["targets"].shape[0]):
            yield get_dataset_item(data, idx)
    # ------------------------

    d = SampleGenerator.from_fn(generator_fn)(data)

    compare_arrays(d.load_all(), data)
    d = SampleGenerator.from_fn(generator_fn)(data)

    it = d.batch_iter(batch_size=3)
    compare_arrays(next(it), get_dataset_item(data, np.arange(3)))


def test_BatchGenerator(data):
    # BatchGenerator example:
    def generator_fn(data, batch_size):
        for idx in range(int(np.ceil(data["targets"].shape[0] / batch_size))):
            start = idx * batch_size
            end = min((idx + 1) * batch_size, data["targets"].shape[0])
            yield get_dataset_item(data, np.arange(start, end))
    # ------------------------

    d = BatchGenerator.from_fn(generator_fn)(data, 3)

    compare_arrays(d.load_all(), data)
    d = BatchGenerator.from_fn(generator_fn)(data, 3)

    it = d.batch_iter()
    compare_arrays(next(it), get_dataset_item(data, np.arange(3)))
