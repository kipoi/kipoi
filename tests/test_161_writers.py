"""Test kipoi.writers
"""
from pytest import fixture
from kipoi.metadata import GenomicRanges
from kipoi.writers import TsvBatchWriter
import numpy as np
import pandas as pd


@fixture
def dl_batch():
    return {"inputs": np.arange(3),
            "metadata": {
                "ranges": GenomicRanges(chr=np.array(["chr1", "chr1", "chr1"]),
                                        start=np.arange(3) + 1,
                                        end=np.arange(3) + 5,
                                        id=np.arange(3).astype(str)
                                        ),
                "gene_id": np.arange(3).astype(str)
    }}


@fixture
def pred_batch_array():
    return np.arange(9).reshape((3, 3))


@fixture
def pred_batch_list():
    return [np.arange(9).reshape((3, 3)),
            np.arange(9).reshape((3, 3))]


@fixture
def pred_batch_dict():
    return {"first": np.arange(9).reshape((3, 3)),
            "second": np.arange(9).reshape((3, 3))}


pred_batch_array = pred_batch_array()
dl_batch = dl_batch()
tmpfile = "/tmp/kipoi/test.tsv"
# --------------------------------------------


def test_TsvBatchWriter_array(dl_batch, pred_batch_array, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.tsv"))
    writer = TsvBatchWriter(tmpfile)
    writer.batch_write(dl_batch, pred_batch_array)
    writer.batch_write(dl_batch, pred_batch_array)
    writer.close()
    # TODO - read in the produced file
    df = pd.read_csv(tmpfile, sep="\t")

    pass
