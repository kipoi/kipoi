import numpy as np
from kipoi.external.flatten_json import unflatten_list
from abc import abstractmethod


class Reader(object):

    @abstractmethod
    def batch_iter(self, batch_size=4, **kwargs):
        pass

    @abstractmethod
    def load_all(self):
        pass

# --------------------------------------------


def _h5py_dataset_iterator(g, prefix=''):
    import h5py
    for key in g:
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset):  # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group):  # test for group (go down)
            for x in _h5py_dataset_iterator(item, path):
                yield x


class HDF5Reader(Reader):

    def __init__(self, file_path):
        import h5py
        self.file_path = file_path

        # self.f = h5py.File(self.file_path, "r")
        self.f = None

    def ls(self):
        return list(_h5py_dataset_iterator(self.f))

    def load_all(self, unflatten=True):
        d = dict()
        for k, v in self.ls():
            d[k] = v[:]
        if unflatten:
            return unflatten_list(d, "/")
        else:
            return d

    def batch_iter(self, batch_size=16, **kwargs):

        datasets = self.ls()
        size = datasets[0][1].shape[0]
        n_batches = int(np.ceil(size / batch_size))
        for i in range(n_batches):
            d = dict()
            for k, v in datasets:
                if i == n_batches - 1:
                    # last batch
                    d[k] = v[(i * batch_size):]
                else:
                    d[k] = v[(i * batch_size):((i + 1) * batch_size)]
            yield unflatten_list(d, "/")

    def __enter__(self):
        import h5py
        self.f = h5py.File(self.file_path, "r")
        return self

    def __exit__(self, *args):
        self.f.close()

    def open(self):
        self.__enter__()

    def close(self):
        self.__exit__()

    @classmethod
    def load(cls, file_path, unflatten=True):
        with cls(file_path) as f:
            return f.load_all(unflatten=unflatten)
