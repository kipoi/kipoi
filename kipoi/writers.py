"""Writers used in `kipoi predict`

- TsvWriter
- HDF5Writer
"""
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
from abc import abstractmethod
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from kipoi.utils import map_nested
from kipoi.data_utils import flatten_batch, numpy_collate_concat
from kipoi.external.flatten_json import flatten
from kipoi.components import MetadataType


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
        df.sort_index(axis=1, inplace=True)
        if self.first_pass:
            df.to_csv(self.file_path, sep="\t", index=False)
            self.first_pass = False
        else:
            df.to_csv(self.file_path, sep="\t", index=False, header=None, mode="a")

    def close(self):
        # nothing to do
        pass


class BedBatchWriter(BatchWriter):

    def __init__(self,
                 file_path,
                 metadata_schema,
                 header=True):
        """

        Args:
          file_path (str): File path of the output tsv file
          dataloader_schema: Schema of the dataloader. Used to find the ranges object
          nested_sep: What separator to use for flattening the nested dictionary structure
            into a single key
        """
        self.file_path = file_path
        self.header = header
        self.first_pass = True

        f_dl_schema = flatten(metadata_schema)
        range_keys = ["metadata/" + k for k in f_dl_schema if f_dl_schema[k].type == MetadataType.GENOMIC_RANGES]
        if len(range_keys) > 1:
            raise ValueError("Found multiple genomic ranges in metadata: {0}. For writing to the " +
                             "bed file exactly one genomic range has to exist".format(range_keys))
        elif len(range_keys) == 0:
            raise ValueError("Found no genomic ranges in metadata. For writing to the " +
                             "bed file exactly one genomic range has to exist")
        self.ranges_key = range_keys[0]

    def batch_write(self, batch):
        fbatch = flatten_batch(batch, nested_sep="/")

        # since 'score' is not defined in GenomicRanges, use "."
        if os.path.join(self.ranges_key, "score") not in fbatch:
            fbatch[os.path.join(self.ranges_key, "score")] = "."

        bed_cols = ["chr", "start", "end", "id", "score", "strand"]
        cols = [os.path.join(self.ranges_key, x) for x in bed_cols] + \
            sorted([x for x in fbatch if x.startswith("preds/")])
        df = pd.DataFrame(fbatch)[cols]
        df.rename(columns={os.path.join(self.ranges_key, bc): bc for bc in bed_cols}, inplace=True)
        df.rename(columns={"id": "name"}, inplace=True)
        if self.first_pass and self.header:
            df.to_csv(self.file_path, sep="\t", index=False)
            self.first_pass = False
        else:
            df.to_csv(self.file_path, sep="\t", index=False, header=None, mode="a")

    def close(self):
        # nothing to do
        pass


class HDF5BatchWriter(BatchWriter):

    def __init__(self, file_path,
                 chunk_size=10000,
                 compression='default'):
        """
        Args:
          file_path (str): File path of the output tsv file
          chunk_size (str): Chunk size for storing the files
          nested_sep: What separator to use for flattening the nested dictionary structure
            into a single key
        """
        import h5py
        if sys.version_info[0] == 2:
            self.string_type = h5py.special_dtype(vlen=unicode)
        else:
            self.string_type = h5py.special_dtype(vlen=str)

        self.file_path = file_path
        self.chunk_size = chunk_size
        self.compression = compression
        self.write_buffer = None
        self.write_buffer_size = 0
        self.first_pass = True
        self.file_handle = None
        # open the file
        self.f = h5py.File(self.file_path, 'a')  # Create file

    def batch_write(self, batch):
        fbatch = flatten(batch, separator="/")

        batch_sizes = [fbatch[k].shape[0] for k in fbatch]
        # assert all shapes are the same
        assert len(pd.Series(batch_sizes).unique()) == 1
        batch_size = batch_sizes[0]

        if self.first_pass:
            # have a dictionary holding
            for k in fbatch:
                if fbatch[k].dtype.type in [np.string_, np.str_, np.unicode_]:
                    dtype = self.string_type
                else:
                    dtype = fbatch[k].dtype

                self.f.create_dataset(k,
                                      shape=(0, ) + fbatch[k].shape[1:],
                                      dtype=dtype,
                                      maxshape=(None, ) + fbatch[k].shape[1:],
                                      chunks=(self.chunk_size, ) + fbatch[k].shape[1:])
            self.first_pass = False
        # add data to the buffer
        if self.write_buffer is None:
            self.write_buffer = fbatch
            self.write_buffer_size = batch_size
        else:
            self.write_buffer = numpy_collate_concat([self.write_buffer, fbatch])
            self.write_buffer_size += batch_size

        if self.write_buffer is not None and self.write_buffer_size >= self.chunk_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffer
        """
        for k in self.write_buffer:
            dset = self.f[k]
            clen = dset.shape[0]
            # resize
            dset.resize(clen + self.write_buffer_size, axis=0)
            # write
            dset[clen:] = self.write_buffer[k]
        self.f.flush()
        self.write_buffer = None
        self.write_buffer_size = 0

    def close(self):
        if self.write_buffer is not None:
            self._flush_buffer()
        self.f.close()

    @classmethod
    def dump(cls, file_path, batch):
        cls(file_path=file_path).batch_write(batch).close()

# Nice-to-have writers:
# - parquet
# - zarr, bcolz <-> xarray


FILE_SUFFIX_MAP = {"h5": HDF5BatchWriter,
                   "hdf5": HDF5BatchWriter,
                   "tsv": TsvBatchWriter,
                   "bed": BedBatchWriter}
