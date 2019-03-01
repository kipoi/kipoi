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
import time
import warnings
from abc import abstractmethod
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from kipoi_utils.utils import map_nested, get_subsuffix
from kipoi_utils.data_utils import flatten_batch, numpy_collate_concat
from kipoi_utils.external.flatten_json import flatten
from kipoi.specs import MetadataType
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


class MultipleBatchWriter(BatchWriter):
    """A container holding multiple batch writers

    # Arguments
      batch_writers: a list of BatchWriters
    """

    def __init__(self, batch_writers):
        assert isinstance(batch_writers, list) or isinstance(batch_writers, set)
        for bw in batch_writers:
            if not isinstance(bw, BatchWriter):
                raise ValueError("{} doesn't inherit from kipoi.writers.BatchWriter".format(bw))
        self.batch_writers = batch_writers

    def batch_write(self, batch):
        """Write a batch of data to multiple files

        # Arguments
            batch: batch of data. Either a single `np.array` or a list/dict thereof.
        """
        for bw in self.batch_writers:
            bw.batch_write(batch)

    def close(self):
        """Close the batch writers
        """
        for bw in self.batch_writers:
            bw.close()





class AsyncBatchWriter(BatchWriter):

    def __init__(self, batch_writer, max_queue_size=10, close_wait=None):
        """
        Args:
          batch_writer: BatchWriter object
          max_queue_size: maximal queue size. If it gets
            larger then batch_write needs to wait
             till it can write to the queue again.
          close_wait: deprecated, we wait unit job is done
        """
        from multiprocessing import JoinableQueue, Process
        self.batch_writer = batch_writer
        if close_wait is not None:
            warnings.warn("AsyncBatchWriter usage of kw close_wait is deprecated", 
                DeprecationWarning) 


        def _write_worker(q, batch_writer):
            """Writer loop

            Args:
              q: multiprocessing.JoinableQueue
              batch_writer.
            """
            while True:
                batch = q.get() # <- this blocks
                batch_writer.batch_write(batch)
                q.task_done()

        # instantiate the queue and start the process
        self.queue = JoinableQueue(max_queue_size)
        self.process = Process(target=_write_worker,
                               args=(self.queue, self.batch_writer))
        self.process.start()

    def batch_write(self, batch):
        """Write a single batch of data

        Args:
          batch is one batch of data (nested numpy arrays with the same axis 0 shape)
        """
        self.queue.put(batch)

    def close(self):
        """Close the file
        """
        # stop the process,
        # make sure the queue is empty
        # close the file
        self.queue.join()  # wait one second to close it
        self.batch_writer.close()
        self.process.terminate()

    def __del__(self):
        self.close()
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


class ParquetBatchWriter(BatchWriter):
    """
    Args:
      it: iterator yielding dictionaries
      fname: chunksize - number of elements to batch together
      compression: str, dict compression to apply to each column,
         e.g. GZIP or SNAPPY or {col1: "SNAPPY", col2: None} to specify per column.
      file_scheme: 'simple'|'hive'
        If simple: all goes in a single file If hive: each row group is in a separate file,
        and a separate file (called "_metadata") contains the metadata
      partition_on: list of column names
        Passed to groupby in order to split data within each row-group, producing a structured
        directory tree. Note: as with pandas, null values will be dropped.
        Ignored if file_scheme is simple.
      create_dirs: if True, recursively create required folders
      **kwargs: additional arguments to pd.DataFrame constructor


    # Install: conda install -c conda-forge fastparquet
    """

    def __init__(self,
                 file_path,
                 nested_sep="/", **kwargs):
        try:
            import fastparquet as fp
        except:
            raise ValueError("fastparquet not installed. "
                             "Please install it to use ParquetBatchWriter")
        self.kwargs = kwargs
        self.file_path = file_path
        self.nested_sep = nested_sep
        self.first_pass = True
        self.fp = fp

    def batch_write(self, batch):
        """Write a batch of data

        # Arguments
            batch: batch of data. Either a single `np.array` or a list/dict thereof.
        """
        df = pd.DataFrame(flatten_batch(batch, nested_sep=self.nested_sep))
        df.sort_index(axis=1, inplace=True)
        if self.first_pass:
            self.fp.write(self.file_path, df, append=not self.first_pass, write_index=False, **self.kwargs)
            self.first_pass = False
        else:
            self.fp.write(self.file_path, df, append=not self.first_pass, write_index=False, **self.kwargs)

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
      file_path (str): File path of the output .h5 file
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


def get_zarr_store(file_path):
    """Get the storage type
    """
    import zarr
    ZARR_STORE_MAP = {"lmdb": zarr.LMDBStore,
                      "zip": zarr.ZipStore,
                      "dbm": zarr.DBMStore,
                      "default": zarr.DirectoryStore}

    suffix, subsuffix = get_subsuffix(file_path)
    if suffix != 'zarr' or (subsuffix is not None and subsuffix not in ZARR_STORE_MAP):
        return ZARR_STORE_MAP['default'](file_path)
    else:
        return ZARR_STORE_MAP[subsuffix](file_path)


class ZarrBatchWriter(BatchWriter):
    """Zarr file writer

    # Arguments
      file_path (str): File path of the output zarr file
      chunk_size (str): Chunk size for storing the files
      store: zarr.storage. If not specified, it's inferred from the file-name.
        For example: *.lmdb.zarr uses LMDB, *.zip.zarr uses Zip, and no special suffix
        uses DirectoryStore      
      compressor (str): Zarr compressor from numcodecs. Example:
        from numcodecs import Blosc
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
      string_dtype: how to encode the string. If None, variable length is used
    """

    def __init__(self, file_path,
                 chunk_size=10000,
                 store=None,
                 string_dtype=None,
                 compressor=None):
        import zarr

        self.file_path = file_path
        self.chunk_size = chunk_size

        if string_dtype is None:
            if sys.version_info[0] == 2:
                self.string_dtype = unicode
            else:
                self.string_dtype = str
        else:
            self.string_dtype = string_dtype

        if compressor is None:
            # use blosc compressor by default
            from numcodecs import Blosc
            self.compressor = Blosc()
            # self.compressor = None
        else:
            self.compressor = compressor
        self.write_buffer = None
        self.write_buffer_size = 0
        self.first_pass = True

        if store is None:
            self.store = get_zarr_store(self.file_path)
        else:
            self.store = store

        # setup the group
        self.root = zarr.group(self.store, overwrite=True)

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
                    dtype = self.string_dtype
                else:
                    dtype = fbatch[k].dtype

                self.root.empty(k,
                                shape=(0,) + fbatch[k].shape[1:],
                                dtype=dtype,
                                compressor=self.compressor,
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
        wb = numpy_collate_concat(self.write_buffer)  # merge the buffer
        for k in wb:
            if sys.version_info[0] == 2 and wb[k].dtype.type in [np.string_, np.str_, np.unicode_]:
                self.root[k].append(wb[k].astype(unicode))
            else:
                self.root[k].append(wb[k])
        self.write_buffer = None
        self.write_buffer_size = 0

    def close(self):
        """Close the file handle
        """
        if self.write_buffer is not None:
            self._flush_buffer()
        if hasattr(self.store, 'close'):
            self.store.close()

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
                   "pq": ParquetBatchWriter,
                   "parquet": ParquetBatchWriter,
                   "zarr": ZarrBatchWriter,
                   "pqt": ParquetBatchWriter,
                   "tsv": TsvBatchWriter,
                   "bed": BedBatchWriter}


def get_writer(output_file, metadata_schema=None, **kwargs):
    """Given the output file suffix, get the appropriate writer

    # Arguments
      output_file: file path string
      dl: (optinal) dataloader metadata_schema
      **kwargs: additional kwargs passed to the batch_writer

    # Returns
      BatchWriter object
    """
    ending = output_file.split('.')[-1]
    W = FILE_SUFFIX_MAP.get(ending, None)
    if ending == "tsv":
        assert W == TsvBatchWriter
        return TsvBatchWriter(file_path=output_file, nested_sep="/", **kwargs)
    elif ending == "bed":
        assert W == BedBatchWriter
        if metadata_schema is None:
            raise ValueError("metadata_schema needs to be specified for BedBatchWriter")
        return BedBatchWriter(file_path=output_file,
                              dataloader_schema=metadata_schema,
                              header=True, **kwargs)
    elif ending in ["hdf5", "h5"]:
        return HDF5BatchWriter(file_path=output_file, **kwargs)
    else:
        return None


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
    """

    # Arguments
      file_path (str): File path of the output tsv file
      genome_file: genome file containing chromosome sizes. Can
        be None. Can be overriden by `chrom_sizes`.
      chrom_sizes: a list of tuples containing chromosome sizes.
        If not None, it overrided `genome_file`.
      is_sorted: if True, the provided entries need to be sorted beforehand

    Note: One of `genome_file` or `chrom_sizes` shouldn't be None.
    """

    def __init__(self, file_path, genome_file=None, chrom_sizes=None, is_sorted=True):
        import pandas as pd
        import pyBigWig
        self.file_path = file_path
        self.genome_file = genome_file
        # read the genome file
        if chrom_sizes is not None:
            self.chrom_sizes = chrom_sizes
        else:
            if genome_file is None:
                raise ValueError("One of `chrom_sizes` or `genome_file` should not be None")
            self.chrom_sizes = pd.read_csv(self.genome_file, header=None, sep='\t').values.tolist()
        self.bw = pyBigWig.open(self.file_path, "w")
        self.bw.addHeader(self.chrom_sizes)
        self.is_sorted = is_sorted
        if not self.is_sorted:
            import tempfile
            self.bgw = BedGraphWriter(tempfile.mkstemp()[1])
        else:
            self.bgw = None

    def region_write(self, region, data):
        """Write region to file. Note: the written regions need to be sorted beforehand.

        # Arguments
          region: a `kipoi.metadata.GenomicRanges`,  `pybedtools.Interval` or a dictionary with at least keys:
            "chr", "start", "end" and list-values. Example: `{"chr":"chr1", "start":0, "end":4}`.
          data: a 1D-array of values to be written - where the 0th entry is at 0-based "start"
        """
        if not self.is_sorted:
            self.bgw.region_write(region, data)
            return None

        def get_el(obj):
            if isinstance(obj, np.ndarray):
                assert len(data.shape) == 1
            if isinstance(obj, list) or isinstance(obj, np.ndarray):
                assert len(obj) == 1
                return obj[0]
            return obj

        if isinstance(region, dict):
            if 'chr' in region:
                chr = get_el(region["chr"])
            elif 'chrom' in region:
                chr = get_el(region["chr"])
            else:
                raise ValueError("'chr' or 'chrom' not in `region`")
            start = int(get_el(region["start"]))

            if 'end' in region:
                end = int(get_el(region["end"]))
            elif 'stop' in region:
                end = int(get_el(region["end"]))
            else:
                raise ValueError("'end' or 'stop' not in `region`")
        else:
            # works also with pybedtools.Interval
            chr = region['chrom']
            start = region['start']
            end = region['end']

        if end - start != len(data):
            raise ValueError("end - start ({end - start})!= len(data) ({len(data)})".
                             format(start=start, end=end, data=data))
        # if len(data.shape) == 2:
        #     data = data.sum(axis=1)
        assert len(data.shape) == 1

        self.bw.addEntries(chr, int(start), values=data.astype(float), span=1, step=1, validate=True)

    def close(self):
        """Close the file
        """
        if self.is_sorted:
            self.bw.close()
        else:
            # convert bedGraph to bigwig
            from pybedtools import BedTool
            # close the temp file
            self.bgw.close()
            # sort the tempfile and get the path of the sorted file
            sorted_fn = BedTool(self.bgw.file_path).sort().fn
            # write the bigwig file
            with open(sorted_fn) as ifh:
                for l in ifh:
                    chr, start, end, val = l.rstrip().split("\t")
                    self.bw.addEntries([chr], [int(start)], ends=[int(end)], values=[float(val)])
            self.bw.close()
