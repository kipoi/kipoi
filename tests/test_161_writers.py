"""Test kipoi.writers
"""
import os
import pytest
from pytest import fixture
from kipoi.metadata import GenomicRanges
from kipoi.writers import (
    AsyncBatchWriter,
    BedBatchWriter,
    TsvBatchWriter,
    ZarrBatchWriter,
    get_zarr_store,
    HDF5BatchWriter,
    BedGraphWriter,
    MultipleBatchWriter,
    ParquetBatchWriter,
    ParquetFileBatchWriter,
    ParquetDirBatchWriter,
)
from kipoi.readers import HDF5Reader, ZarrReader
from kipoi.cli.main import prepare_batch
import numpy as np
import pandas as pd
from kipoi.specs import DataLoaderSchema, ArraySchema, MetadataStruct, MetadataType
from collections import OrderedDict

from kipoi_utils.utils import get_subsuffix
import zarr


def on_circle_ci():
    if os.environ.get('CI') is not None:
        return True
    elif os.environ.get('CIRCLECI') is not None:
        return True
    elif os.environ.get('CIRCLE_BRANCH') is not None:
        return True
    else:
        return False


@fixture
def metadata_schema():
    return OrderedDict([("ranges", MetadataStruct(type=MetadataType.GENOMIC_RANGES, doc="ranges")),
                        ("gene_id", MetadataStruct(type=MetadataType.STR, doc="gene id"))])


@fixture
def dl_batch():
    return {
        "inputs": np.arange(3),
        "metadata": {
            "ranges": GenomicRanges(
                chr=np.array(["chr1", "chr1", "chr1"]),
                start=np.arange(3) + 1,
                end=np.arange(3) + 5,
                id=np.arange(3).astype(str),
                strand=np.array(["*"] * 3)
            ),
            "gene_id": np.arange(3).astype(str)
        }
    }


@fixture
def pred_batch_array():
    return np.arange(9).reshape((3, 3))


@fixture
def pred_batch_list():
    return [np.arange(9).reshape((3, 3)),
            np.arange(9).reshape((3, 3))]


@fixture
def pred_batch_vlen():
    return [
        np.arange(9).reshape((3, 3)),
        np.asarray([np.arange(i + 3) for i in range(3)], dtype=object),
    ]


@fixture
def pred_batch_dict():
    return {
        "first": np.arange(9).reshape((3, 3)),
        "second": np.arange(9).reshape((3, 3))
    }


# pred_batch_array = pred_batch_array()
# pred_batch_list = pred_batch_list()
# pred_batch_vlen = pred_batch_vlen()
# pred_batch_dict = pred_batch_dict()
# dl_batch = dl_batch()
# tmpfile = "/tmp/kipoi/test.tsv"
# metadata_schema = metadata_schema()


# --------------------------------------------


def test_TsvBatchWriter_array(dl_batch, pred_batch_array, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.tsv"))
    writer = TsvBatchWriter(tmpfile)
    batch = prepare_batch(dl_batch, pred_batch_array, keep_metadata=True)
    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    df = pd.read_csv(tmpfile, sep="\t")

    assert set(list(df.columns)) == {'metadata/ranges/id',
                                     'metadata/ranges/strand',
                                     'metadata/ranges/chr',
                                     'metadata/ranges/start',
                                     'metadata/ranges/end',
                                     'metadata/gene_id',
                                     'preds/0',
                                     'preds/1',
                                     'preds/2'}
    assert list(df['metadata/ranges/id']) == [0, 1, 2, 0, 1, 2]


def test_ParquetFileBatchWriter_array(dl_batch, pred_batch_array, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.parquet"))
    writer = ParquetFileBatchWriter(tmpfile)
    batch = prepare_batch(dl_batch, pred_batch_array, keep_metadata=True)
    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    df = pd.read_parquet(tmpfile)

    assert set(list(df.columns)) == {'metadata/ranges/id',
                                     'metadata/ranges/strand',
                                     'metadata/ranges/chr',
                                     'metadata/ranges/start',
                                     'metadata/ranges/end',
                                     'metadata/gene_id',
                                     'preds/0',
                                     'preds/1',
                                     'preds/2'}
    assert list(df['metadata/ranges/id']) == ['0', '1', '2', '0', '1', '2']


def test_ParquetDirBatchWriter_array(dl_batch, pred_batch_array, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.parquet"))
    writer = ParquetDirBatchWriter(tmpfile)
    batch = prepare_batch(dl_batch, pred_batch_array, keep_metadata=True)
    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    df = pd.read_parquet(tmpfile)

    assert set(list(df.columns)) == {'metadata/ranges/id',
                                     'metadata/ranges/strand',
                                     'metadata/ranges/chr',
                                     'metadata/ranges/start',
                                     'metadata/ranges/end',
                                     'metadata/gene_id',
                                     'preds/0',
                                     'preds/1',
                                     'preds/2'}
    assert list(df['metadata/ranges/id']) == ['0', '1', '2', '0', '1', '2']


# For no good reason this test fails when installing
# from conda even tough this work very fine locally
@pytest.mark.skipif("os.environ.get('CI_JOB_PY_YAML') is not None")
def test_AsyncTsvBatchWriter_array(dl_batch, pred_batch_array, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.tsv"))
    writer = AsyncBatchWriter(TsvBatchWriter(tmpfile))
    batch = prepare_batch(dl_batch, pred_batch_array, keep_metadata=True)
    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    df = pd.read_csv(tmpfile, sep="\t")

    assert set(list(df.columns)) == {'metadata/ranges/id',
                                     'metadata/ranges/strand',
                                     'metadata/ranges/chr',
                                     'metadata/ranges/start',
                                     'metadata/ranges/end',
                                     'metadata/gene_id',
                                     'preds/0',
                                     'preds/1',
                                     'preds/2'}
    assert list(df['metadata/ranges/id']) == [0, 1, 2, 0, 1, 2]


def test_HDF5BatchWriter_array(dl_batch, pred_batch_array, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.h5"))
    batch = prepare_batch(dl_batch, pred_batch_array, keep_metadata=True)
    writer = HDF5BatchWriter(tmpfile, chunk_size=4)

    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    with HDF5Reader(tmpfile) as f:
        assert np.all(list(f.batch_iter(2))[0]['metadata']['gene_id'] == dl_batch['metadata']['gene_id'][:2])
        out = f.load_all()
        assert np.all(
            out['metadata']['gene_id'] == np.concatenate([
                dl_batch['metadata']['gene_id'],
                dl_batch['metadata']['gene_id']
            ])
        )
        assert np.all(
            out['metadata']['ranges']["chr"] == np.concatenate([
                dl_batch['metadata']['ranges']['chr'],
                dl_batch['metadata']['ranges']['chr']
            ])
        )
        assert np.all(
            out['metadata']['ranges']["start"] == np.concatenate([
                dl_batch['metadata']['ranges']['start'],
                dl_batch['metadata']['ranges']['start']
            ])
        )
        assert np.all(out['preds'][:3] == pred_batch_array)


def test_HDF5BatchWriter_list(dl_batch, pred_batch_list, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.h5"))
    batch = prepare_batch(dl_batch, pred_batch_list, keep_metadata=True)
    writer = HDF5BatchWriter(tmpfile, chunk_size=4)

    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    with HDF5Reader(tmpfile) as f:
        assert np.all(list(f.batch_iter(2))[0]['metadata']['gene_id'] == dl_batch['metadata']['gene_id'][:2])
        out = f.load_all()
        assert np.all(
            out['metadata']['gene_id'] == np.concatenate([
                dl_batch['metadata']['gene_id'],
                dl_batch['metadata']['gene_id']
            ])
        )
        assert np.all(
            out['metadata']['ranges']["chr"] == np.concatenate([
                dl_batch['metadata']['ranges']['chr'],
                dl_batch['metadata']['ranges']['chr']
            ])
        )
        assert np.all(
            out['metadata']['ranges']["start"] == np.concatenate([
                dl_batch['metadata']['ranges']['start'],
                dl_batch['metadata']['ranges']['start']
            ])
        )
        assert np.all(out['preds'][0][:3] == pred_batch_list[0])


def test_HDF5BatchWriter_vlen(dl_batch, pred_batch_vlen, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.h5"))
    batch = prepare_batch(dl_batch, pred_batch_vlen, keep_metadata=True)
    writer = HDF5BatchWriter(tmpfile, chunk_size=4)

    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    with HDF5Reader(tmpfile) as f:
        assert np.all(list(f.batch_iter(2))[0]['metadata']['gene_id'] == dl_batch['metadata']['gene_id'][:2])
        out = f.load_all()
        assert np.all(
            out['metadata']['gene_id'] == np.concatenate([
                dl_batch['metadata']['gene_id'],
                dl_batch['metadata']['gene_id']
            ])
        )
        assert np.all(
            out['metadata']['ranges']["chr"] == np.concatenate([
                dl_batch['metadata']['ranges']['chr'],
                dl_batch['metadata']['ranges']['chr']
            ])
        )
        assert np.all(
            out['metadata']['ranges']["start"] == np.concatenate([
                dl_batch['metadata']['ranges']['start'],
                dl_batch['metadata']['ranges']['start']
            ])
        )
        assert np.all(
            out['preds'][0][:3] == pred_batch_vlen[0])


# Zarr

def test_get_subsuffix():
    assert get_subsuffix("asds.lmdb.zarr") == ('zarr', 'lmdb')
    assert get_subsuffix("/asdasd.asd/asds.lmdb.zarr") == ('zarr', 'lmdb')
    assert get_subsuffix("asds.zarr") == ('zarr', "")
    assert get_subsuffix("asdszarr") == ('', '')


def test_zarr_store():
    assert isinstance(get_zarr_store('output.zarr'), zarr.storage.DirectoryStore)
    assert isinstance(get_zarr_store('output.za'), zarr.storage.DirectoryStore)
    assert isinstance(get_zarr_store('output'), zarr.storage.DirectoryStore)
    assert isinstance(get_zarr_store('output.zip.zarr'), zarr.storage.ZipStore)


def test_ZarrBatchWriter_array(dl_batch, pred_batch_array, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.zip.zarr"))
    batch = prepare_batch(dl_batch, pred_batch_array, keep_metadata=True)
    writer = ZarrBatchWriter(tmpfile, chunk_size=4)

    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    with ZarrReader(tmpfile) as f:
        assert np.all(list(f.batch_iter(2))[0]['metadata']['gene_id'] == dl_batch['metadata']['gene_id'][:2])
        out = f.load_all()
        assert np.all(out['metadata']['gene_id'] == np.concatenate(
            [dl_batch['metadata']['gene_id'], dl_batch['metadata']['gene_id']]))
        assert np.all(out['metadata']['ranges']["chr"] == np.concatenate([dl_batch['metadata']['ranges']['chr'],
                                                                          dl_batch['metadata']['ranges']['chr']]))
        assert np.all(out['metadata']['ranges']["start"] == np.concatenate([dl_batch['metadata']['ranges']['start'],
                                                                            dl_batch['metadata']['ranges']['start']]))
        assert np.all(out['preds'][:3] == pred_batch_array)


def test_ZarrBatchWriter_list(dl_batch, pred_batch_list, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.zip.zarr"))
    batch = prepare_batch(dl_batch, pred_batch_list, keep_metadata=True)
    writer = ZarrBatchWriter(tmpfile, chunk_size=4)

    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    with ZarrReader(tmpfile) as f:
        assert np.all(list(f.batch_iter(2))[0]['metadata']['gene_id'] == dl_batch['metadata']['gene_id'][:2])
        out = f.load_all()
        assert np.all(out['metadata']['gene_id'] == np.concatenate(
            [dl_batch['metadata']['gene_id'], dl_batch['metadata']['gene_id']]))
        assert np.all(out['metadata']['ranges']["chr"] == np.concatenate([dl_batch['metadata']['ranges']['chr'],
                                                                          dl_batch['metadata']['ranges']['chr']]))
        assert np.all(out['metadata']['ranges']["start"] == np.concatenate([dl_batch['metadata']['ranges']['start'],
                                                                            dl_batch['metadata']['ranges']['start']]))
        assert np.all(out['preds'][0][:3] == pred_batch_list[0])


def test_MultipleBatchWriter(dl_batch, pred_batch_array, tmpdir):
    tmpdir = tmpdir.mkdir("example")
    h5_tmpfile = str(tmpdir.join("out.h5"))
    tsv_tmpfile = str(tmpdir.join("out.tsv"))
    batch = prepare_batch(dl_batch, pred_batch_array, keep_metadata=True)
    writer = MultipleBatchWriter([TsvBatchWriter(tsv_tmpfile), HDF5BatchWriter(h5_tmpfile)])
    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    assert os.path.exists(h5_tmpfile)
    assert os.path.exists(tsv_tmpfile)
    df = pd.read_csv(tsv_tmpfile, sep="\t")
    assert set(list(df.columns)) == {'metadata/ranges/id',
                                     'metadata/ranges/strand',
                                     'metadata/ranges/chr',
                                     'metadata/ranges/start',
                                     'metadata/ranges/end',
                                     'metadata/gene_id',
                                     'preds/0',
                                     'preds/1',
                                     'preds/2'}
    assert list(df['metadata/ranges/id']) == [0, 1, 2, 0, 1, 2]


def test_BedBatchWriter(dl_batch, pred_batch_array, metadata_schema, tmpdir):
    tmpfile = str(tmpdir.mkdir("example").join("out.tsv"))
    writer = BedBatchWriter(tmpfile, metadata_schema=metadata_schema)
    batch = prepare_batch(dl_batch, pred_batch_array, keep_metadata=True)
    writer.batch_write(batch)
    writer.batch_write(batch)
    writer.close()
    df = pd.read_csv(tmpfile, sep="\t")

    assert list(df.columns) == ['chr',
                                'start',
                                'end',
                                'name',
                                'score',
                                'strand',
                                'preds/0',
                                'preds/1',
                                'preds/2']
    assert list(df['name']) == [0, 1, 2, 0, 1, 2]


def test_bigwigwriter(tmpdir):
    from kipoi.writers import BigWigWriter
    import pyBigWig
    import tempfile
    tmpfile = str(tmpdir.mkdir("example").join("out.bw"))
    bww = BigWigWriter(tmpfile, chrom_sizes=[('chr1', 1000), ('chr10', 1000)])
    regions = [
        ({"chr": "chr1", "start": 10, "end": 20}, np.arange(10)),
        ({"chr": "chr1", "start": 30, "end": 40}, np.arange(10)[::-1]),
        ({"chr": "chr10", "start": 10, "end": 20}, np.arange(10))
    ]
    for region, data in regions:
        bww.region_write(region, data)
    bww.close()
    # load the bigwig file and validate the values
    r = pyBigWig.open(tmpfile)

    for region, data in regions:
        # query the values
        assert np.allclose(data, r.values(region['chr'],
                                          region['start'],
                                          region['end'], numpy=True))
    # assert there are no values here
    assert np.isnan(r.values("chr1", 20, 30, numpy=True)).all()
    r.close()


def test_bigwigwriter_not_sorted(tmpdir):
    from kipoi.writers import BigWigWriter
    import pyBigWig
    import tempfile
    tmpfile = str(tmpdir.mkdir("example").join("out.bw"))
    bww = BigWigWriter(tmpfile, chrom_sizes=[('chr1', 1000), ('chr10', 1000)], is_sorted=False)
    regions = [
        ({"chr": "chr1", "start": 30, "end": 40}, np.arange(10)[::-1]),
        ({"chr": "chr1", "start": 10, "end": 20}, np.arange(10)),
        ({"chr": "chr10", "start": 10, "end": 20}, np.arange(10))
    ]
    for region, data in regions:
        bww.region_write(region, data)
    bww.close()
    # load the bigwig file and validate the values
    r = pyBigWig.open(tmpfile)

    for region, data in regions:
        # query the values
        assert np.allclose(data, r.values(region['chr'],
                                          region['start'],
                                          region['end'], numpy=True))
    # assert there are no values here
    assert np.isnan(r.values("chr1", 20, 30, numpy=True)).all()
    r.close()


def test_bedgraphwriter():
    import os
    import tempfile
    temp_path = tempfile.mkstemp()[1]
    bgw = BedGraphWriter(temp_path)
    regions = {"chr": ["chr1", "chr7", "chr2"], "start": [10, 30, 20], "end": [11, 31, 21]}
    values = [3.0, 4.0, 45.4]
    for i, val in enumerate(values):
        reg = {k: v[i] for k, v in regions.items()}
        bgw.region_write(reg, np.array([val]))
    bgw.close()
    with open(temp_path, "r") as ifh:
        for i, l in enumerate(ifh):
            els = l.rstrip().split()
            for j, k in enumerate(["chr", "start", "end"]):
                assert str(regions[k][i]) == els[j]
            assert str(values[i]) == els[-1]
    os.unlink(temp_path)
