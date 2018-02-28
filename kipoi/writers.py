"""Writers used in `kipoi predict`

- TsvWriter
- HDF5Writer
"""
from __future__ import absolute_import
from __future__ import print_function
from abc import abstractmethod
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from kipoi.utils import flatten_nested, map_nested
from kipoi.data_utils import flatten_batch


class BatchWriter(object):

    @abstractmethod
    def batch_write(self, batch):
        """Write a single batch of data

        Args:
          batch is one batch of data (nested numpy arrays with the same axis 0 shape)
        """
        pass

    @abstractmethod
    def close(self):
        """Close the file
        """
        pass

# --------------------------------------------

# helper functions


class TsvBatchWriter(BatchWriter):

    def __init__(self,
                 file_path,
                 nested_sep="/"):
        """

        Args:
          file_path (str): File path of the output tsv file
          nested_sep: What separator to use for flattening the nested dictionary structure
            into a single key
        """
        self.file_path = file_path
        self.nested_sep = nested_sep
        self.first_pass = True

    def batch_write(self, batch):
        df = pd.DataFrame(flatten_batch(batch, nested_sep=self.nested_sep))
        if self.first_pass:
            df.to_csv(self.file_path, sep="\t", index=False)
            self.first_pass = False
        else:
            df.to_csv(self.file_path, sep="\t", index=False, header=None, mode="a")

    def close(self):
        # nothing to do
        pass


class HDF5BatchWriter(BatchWriter):

    def __init__(self, file_path,
                 chunk_size=1000):
        """

        Args:
          file_path (str): File path of the output tsv file
          chunk_size (str): Which chunk size to use for storing the files
          nested_sep: What separator to use for flattening the nested dictionary structure
            into a single key
        """
        pass

    def batch_write(self, batch):
        pass

    def close(self):
        pass
