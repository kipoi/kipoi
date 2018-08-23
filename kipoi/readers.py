"""Readers useful for creating new dataloaders

- HDF5Reader
"""
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
    """Read the HDF5 file. Convenience wrapper around h5py.File

    # Arguments
        file_path: File path to an HDF5 file
    """

    def __init__(self, file_path):
        import h5py
        self.file_path = file_path

        # self.f = h5py.File(self.file_path, "r")
        self.f = None

    def ls(self):
        """Recursively list the arrays
        """
        self._file_open()
        return list(_h5py_dataset_iterator(self.f))

    def _file_open(self):
        if self.f is None:
            raise ValueError("File not opened. Please run self.open() or use the context manager" +
                             ": with HDF5Reader('file') as f: ...")

    def load_all(self, unflatten=True):
        """Load the whole file

        # Arguments
            unflatten: if True, nest/unflatten the keys.
              e.g. an entry `f['/foo/bar']` would need to be accessed
              using two nested `get` call: `f['foo']['bar']`
        """
        d = dict()
        for k, v in self.ls():
            d[k] = v[:]
        if unflatten:
            return unflatten_list(d, "/")
        else:
            return d

    def batch_iter(self, batch_size=16, **kwargs):
        """Create a batch iterator over the whole file

        # Arguments
            batch_size: batch size
            **kwargs: ignored argument. Used for consistency with other dataloaders
        """
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
        """Open the file
        """
        self.__enter__()

    def close(self):
        """Close the file
        """
        self.__exit__()

    @classmethod
    def load(cls, file_path, unflatten=True):
        """Load the data all at once (classmethod).

        # Arguments
            file_path: HDF5 file path
            unflatten: see `load_all`
        """
        with cls(file_path) as f:
            return f.load_all(unflatten=unflatten)
