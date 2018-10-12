"""Writers used in `kipoi predict`

- TsvBatchWriter
- BedBatchWriter
- HDF5BatchWriter
- RegionWriter
- BedGraphWriter
- BigWigWriter
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
from kipoi.specs import MetadataType


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
    """Tab-separated file writer

    # Arguments
      file_path (str): File path of the output tsv file
      nested_sep: What separator to use for flattening the nested dictionary structure
        into a single key
    """

    def __init__(self,
                 file_path,
                 nested_sep="/"):
        self.file_path = file_path
        self.nested_sep = nested_sep
        self.first_pass = True

    def batch_write(self, batch):
        """Write a batch of data

        # Arguments
            batch: batch of data. Either a single `np.array` or a list/dict thereof.
        """
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
    """Bed-file writer

    # Arguments
      file_path (str): File path of the output tsv file
      dataloader_schema: Schema of the dataloader. Used to find the ranges object
      nested_sep: What separator to use for flattening the nested dictionary structure
        into a single key
    """

    def __init__(self,
                 file_path,
                 metadata_schema,
                 header=True):
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
        """Write a batch of data to bed file

        # Arguments
            batch: batch of data. Either a single `np.array` or a list/dict thereof.
        """
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
    """HDF5 file writer

    # Arguments
      file_path (str): File path of the output tsv file
      chunk_size (str): Chunk size for storing the files
      nested_sep: What separator to use for flattening the nested dictionary structure
        into a single key
      compression (str): default compression to use for the hdf5 datasets.
         see also: <http://docs.h5py.org/en/latest/high/dataset.html#dataset-compression>
    """

    def __init__(self, file_path,
                 chunk_size=10000,
                 compression='gzip'):
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
        """Write a batch of data to bed file

        # Arguments
            batch: batch of data. Either a single `np.array` or a list/dict thereof.
        """
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
                                      shape=(0,) + fbatch[k].shape[1:],
                                      dtype=dtype,
                                      maxshape=(None,) + fbatch[k].shape[1:],
                                      compression=self.compression,
                                      chunks=(self.chunk_size,) + fbatch[k].shape[1:])
            self.first_pass = False
        # add data to the buffer
        if self.write_buffer is None:
            self.write_buffer = [fbatch]
            self.write_buffer_size = batch_size
        else:
            self.write_buffer.append(fbatch)
            self.write_buffer_size += batch_size

        if self.write_buffer is not None and self.write_buffer_size >= self.chunk_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffer
        """
        wb = numpy_collate_concat(self.write_buffer)
        for k in wb:
            dset = self.f[k]
            clen = dset.shape[0]
            # resize
            dset.resize(clen + self.write_buffer_size, axis=0)
            # write
            dset[clen:] = wb[k]
        self.f.flush()
        self.write_buffer = None
        self.write_buffer_size = 0

    def close(self):
        """Close the file handle
        """
        if self.write_buffer is not None:
            self._flush_buffer()
        self.f.close()

    @classmethod
    def dump(cls, file_path, batch):
        """In a single shot write the batch/data to a file and
        close the file.

        # Arguments
            file_path: file path
            batch: batch of data. Either a single `np.array` or a list/dict thereof.
        """
        obj = cls(file_path=file_path)
        obj.batch_write(batch)
        obj.close()


# Nice-to-have writers:
# - parquet
# - zarr, bcolz <-> xarray


FILE_SUFFIX_MAP = {"h5": HDF5BatchWriter,
                   "hdf5": HDF5BatchWriter,
                   "tsv": TsvBatchWriter,
                   "bed": BedBatchWriter}


class RegionWriter(object):
    @abstractmethod
    def region_write(self, region, data):
        """Write a single batch of data

        # Arguments
          region: a `kipoi.metadata.GenomicRanges` object or a dictionary with at least keys: "chr", "start", "end" and list-values 
            of length 1
          data: a 1D-array of values to be written - where the 0th entry is at 0-based "start"
        """
        pass

    @abstractmethod
    def close(self):
        """Close the file
        """
        pass


class BedGraphWriter(RegionWriter):
    """
    # Arguments
      file_path (str): File path of the output bedgraph file
    """

    def __init__(self,
                 file_path):
        self.file_path = file_path
        self.file = open(file_path, "w")

    def region_write(self, region, data):
        """Write region to file.

        # Arguments
            region: Defines the region that will be written position by position. Example: `{"chr":"chr1", "start":0, "end":4}`.
            data: a 1D or 2D numpy array vector that has length "end" - "start". if 2D array is passed then
                `data.sum(axis=1)` is performed on it first.
        """

        def get_el(obj):
            if isinstance(obj, np.ndarray):
                assert len(data.shape) == 1
            if isinstance(obj, list) or isinstance(obj, np.ndarray):
                assert len(obj) == 1
                return obj[0]
            return obj

        chr = get_el(region["chr"])
        start = int(get_el(region["start"]))
        end = int(get_el(region["end"]))
        assert data.shape[0] == end - start
        if len(data.shape) == 2:
            data = data.sum(axis=1)
        assert len(data.shape) == 1
        for zero_pos, value in zip(range(start, end), data):
            self.write_entry(chr, zero_pos, zero_pos + 1, value)

    def write_entry(self, chr, start, end, value):
        """Write region to file.

        # Arguments
            region: Defines the region that will be written position by position. Example: `{"chr":"chr1", "start":0, "end":4}`.
            data: a 1D or 2D numpy array vector that has length "end" - "start". if 2D array is passed then
                `data.sum(axis=1)` is performed on it first.
        """
        tokens = [chr, start, end, value]
        self.file.write("\t".join([str(el) for el in tokens]) + "\n")

    def close(self):
        """Close the file
        """
        self.file.close()


class BigWigWriter(RegionWriter):
    """BigWig entries have to be sorted so the generated values are cached in a bedgraph file.

    # Arguments
      file_path (str): File path of the output tsv file
    """

    def __init__(self,
                 file_path):
        import tempfile
        self.temp_bedgraph_path = tempfile.mkstemp()[1]
        self.file_path = file_path
        self.bgw = BedGraphWriter(file_path=self.temp_bedgraph_path)
        raise Exception("BigWigWriter is not functional due to a Segmentation fault when trying to write to a file.")

    def region_write(self, region, data):
        self.bgw.region_write(region, data)

    def write_entry(self, chr, start, end, value):
        """Write region to file.

        # Arguments
            region: Defines the region that will be written position by position. Example: `{"chr":"chr1", "start":0, "end":4}`.
            data: a 1D or 2D numpy array vector that has length "end" - "start". if 2D array is passed then
                `data.sum(axis=1)` is performed on it first.
        """
        self.bgw.write_entry(chr, start, end, value)

    def close(self):
        """Close the file
        """
        from pybedtools import BedTool
        import pyBigWig
        # close the temp file
        self.bgw.close()
        # sort the tempfile and get the path of the sorted file
        sorted_fn = BedTool(self.temp_bedgraph_path).sort().fn
        # write the bigwig file
        bw = pyBigWig.open(self.file_path, "w")

        with open(sorted_fn, "r") as ifh:
            for l in ifh:
                chr, start, end, val = l.rstrip().split("\t")
                bw.addEntries([chr], [int(start)], ends=[int(end)], values=[float(val)])

        bw.close()
