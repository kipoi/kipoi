import six
import json
from typing import Any

import numpy as np
from pysam import FastaFile


# Reference: https://github.com/kundajelab/genomelake/blob/3f53f490c202fcbca83d6e4a9f1e5f2c68066133/genomelake/extractors.py#L15
NUM_SEQ_CHARS = 4

# Reference: https://github.com/deepmind/deepmind-research/blob/fa8c9be4bb0cfd0b8492203eb2a9f31ef995633c/enformer/enformer.py#L306-L318
def one_hot_encode_sequence(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
  """One-hot encode sequence."""
  def to_uint8(string):
    return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
  hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
  hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
  hash_table[to_uint8(neutral_alphabet)] = neutral_value
  hash_table = hash_table.astype(dtype)
  return hash_table[to_uint8(sequence)]
  
# Reference: https://github.com/kundajelab/genomelake/blob/3f53f490c202fcbca83d6e4a9f1e5f2c68066133/genomelake/backend.py#L101-L138
def load_directory(base_dir, in_memory=False):
    with open(os.path.join(base_dir, "metadata.json"), "r") as fp:
        metadata = json.load(fp)

    if metadata["type"] == "array_numpy":
        mmap_mode = None if in_memory else "r"
        data = {
            chrom: np.load(
                "{}.npy".format(os.path.join(base_dir, chrom)), mmap_mode=mmap_mode
            )
            for chrom in metadata["file_shapes"]
        }

    elif metadata["type"] == "array_bcolz":
        data = {
            chrom: bcolz.open(os.path.join(base_dir, chrom), mode="r")
            for chrom in metadata["file_shapes"]
        }
        if in_memory:
            data = {k: data[k].copy() for k in data.keys()}

    elif metadata["type"] == "array_tiledb":
        data = {
            chrom: load_tiledb(os.path.join(base_dir, chrom))
            for chrom in metadata["file_shapes"]
        }

    else:
        raise ValueError("Can only extract from array_bcolz and array_numpy")

    for chrom, shape in six.iteritems(metadata["file_shapes"]):
        if data[chrom].shape != tuple(shape):
            raise ValueError(
                "Inconsistent shape found in metadata file: "
                "{} - {} vs {}".format(chrom, shape, data[chrom].shape)
            )

    return data


# Reference: https://github.com/kundajelab/genomelake/blob/3f53f490c202fcbca83d6e4a9f1e5f2c68066133/genomelake/extractors.py#L18-L51
class BaseExtractor(object):
    dtype = np.float32

    def __init__(self, datafile, **kwargs):
        self._datafile = datafile

    def __call__(self, intervals, out=None, **kwargs):
        data = self._check_or_create_output_array(intervals, out)
        self._extract(intervals, data, **kwargs)
        return data

    def _check_or_create_output_array(self, intervals, out):
        width = intervals[0].stop - intervals[0].start
        output_shape = self._get_output_shape(len(intervals), width)

        if out is None:
            out = np.zeros(output_shape, dtype=self.dtype)
        else:
            if out.shape != output_shape:
                raise ValueError('out array has incorrect shape: {} '
                                 '(need {})'.format(out.shape, output_shape))
            if out.dtype != self.dtype:
                raise ValueError('out array has incorrect dtype: {} '
                                 '(need {})'.format(out.dtype, self.dtype))
        return out

    def _extract(self, intervals, out, **kwargs):
        'Subclassses should implement this and return the data'
        raise NotImplementedError

    @staticmethod
    def _get_output_shape(num_intervals, width):
        'Subclasses should implement this and return the shape of output'
        raise NotImplementedError

# Reference: https://github.com/kundajelab/genomelake/blob/3f53f490c202fcbca83d6e4a9f1e5f2c68066133/genomelake/extractors.py#L83-L115
class FastaExtractor(BaseExtractor):

    def __init__(self, datafile, use_strand=False, **kwargs):
        """Fasta file extractor
        
        NOTE: The extractor is not thread-save.
        If you with to use it with multiprocessing,
        create a new extractor object in each process.
        
        Args:
          datafile (str): path to the bigwig file
          use_strand (bool): if True, the extracted sequence
            is reverse complemented in case interval.strand == "-"
        """
        super(FastaExtractor, self).__init__(datafile, **kwargs)
        self.use_strand = use_strand
        self.fasta = FastaFile(self._datafile)

    def _extract(self, intervals, out, **kwargs):    
        for index, interval in enumerate(intervals):
            seq = self.fasta.fetch(str(interval.chrom), interval.start,
                                       interval.stop)
            one_hot_encode_sequence(seq, out[index, :, :])

            # reverse-complement seq the negative strand
            if self.use_strand and interval.strand == "-":
                out[index, :, :] = out[index, ::-1, ::-1]

        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        return (num_intervals, width, NUM_SEQ_CHARS)

# Reference: https://github.com/kundajelab/genomelake/blob/3f53f490c202fcbca83d6e4a9f1e5f2c68066133/genomelake/extractors.py#L54-L80
class ArrayExtractor(BaseExtractor):

    def __init__(self, datafile, in_memory=False, **kwargs):
        super(ArrayExtractor, self).__init__(datafile, **kwargs)
        self._data = load_directory(datafile, in_memory=in_memory)
        self.multiprocessing_safe = in_memory

        arr = next(iter(self._data.values()))
        def _mm_extract(self, intervals, out, **kwargs):
            mm_data = self._data
            for index, interval in enumerate(intervals):
                out[index] = mm_data[interval.chrom][interval.start:interval.stop]

        # output shape method
        shape = arr.shape
        if len(shape) == 1:
            def _get_output_shape(num_intervals, width):
                return (num_intervals, width)
        elif len(shape) == 2:
            def _get_output_shape(num_intervals, width):
                return (num_intervals, width, shape[1])
        else:
            raise ValueError('Can only extract from 1D/2D arrays')

        self._mm_extract = _mm_extract.__get__(self)
        self._extract = self._mm_extract
        self._get_output_shape = staticmethod(_get_output_shape).__get__(self)

