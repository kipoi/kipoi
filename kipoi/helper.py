import six
import json
from typing import Any
import os
import numpy as np
from pysam import FastaFile
import bcolz

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


