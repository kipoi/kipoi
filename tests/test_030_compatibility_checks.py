"""Test the compatibility checks
"""
import pytest
import numpy as np
from kipoi.components import ArraySchema, MetadataStruct, DataLoaderSchema
from kipoi.metadata import Ranges
from related import from_yaml

# numpy arrays

GOOD_ARR_SCHEMA_PAIRS = [
    (ArraySchema(shape=(10,), descr=""), np.arange(20).reshape((2, 10))),
    (ArraySchema(shape=(None,), descr=""), np.arange(20).reshape((2, 10))),
    (ArraySchema(shape=(1, None), descr=""), np.arange(20).reshape((2, 1, 10))),
    (ArraySchema(shape=(None, None,), descr=""), np.arange(20).reshape((2, 1, 10))),
    (ArraySchema(shape=(1, 10,), descr=""), np.arange(20).reshape((2, 1, 10))),
    (ArraySchema(shape=(None, 10), descr=""), np.arange(20).reshape((2, 1, 10))),
    (ArraySchema(shape=(), descr=""), np.arange(2)),
]

BAD_ARR_SCHEMA_PAIRS = [
    # dim missmatch
    (ArraySchema(shape=(11,), descr=""), np.arange(20).reshape((2, 10))),
    (ArraySchema(shape=(2, None), descr=""), np.arange(20).reshape((2, 1, 10))),
    # len missmatch
    (ArraySchema(shape=(1, 10, 12), descr=""), np.arange(20).reshape((2, 1, 10))),
    (ArraySchema(shape=(1, 10, None), descr=""), np.arange(20).reshape((2, 1, 10))),
]


@pytest.mark.parametrize("pair", GOOD_ARR_SCHEMA_PAIRS)
def test_good_array_schemas(pair):
    descr, batch = pair
    assert descr.compatible_with(batch)


@pytest.mark.parametrize("pair", BAD_ARR_SCHEMA_PAIRS)
def test_bad_array_schemas(pair):
    descr, batch = pair
    assert not descr.compatible_with(batch)


# --------------------------------------------
# metadata structs
GOOD_MDATASTRUCT_PAIRS = [
    (MetadataStruct(type="str", descr=""), np.arange(10).astype(str)),
    (MetadataStruct(type="int", descr=""), np.arange(10).astype(int)),
    (MetadataStruct(type="float", descr=""), np.arange(10).astype(float)),
    (MetadataStruct(type="array", descr=""), np.arange(10).reshape((2, 5))),
    (MetadataStruct(type="Ranges", descr=""), Ranges(chr="chr1",
                                                     start=10,
                                                     end=20,
                                                     id="1",
                                                     strand="+")),
    (MetadataStruct(type="Ranges", descr=""), dict(chr="chr1",
                                                   start=10,
                                                   end=20,
                                                   id="1",
                                                   strand="+")),
]

BAD_MDATASTRUCT_PAIRS = [
    # larger array
    (MetadataStruct(type="str", descr=""), np.arange(10).reshape((2, 5)).astype(str)),
    (MetadataStruct(type="int", descr=""), np.arange(10).reshape((2, 5)).astype(int)),
    (MetadataStruct(type="float", descr=""), np.arange(10).reshape((2, 5)).astype(float)),
    # not an array
    (MetadataStruct(type="array", descr=""), 1),
    (MetadataStruct(type="array", descr=""), "3"),
    (MetadataStruct(type="array", descr=""), [1, 2]),

    # ranges: not a ranges or a dict
    (MetadataStruct(type="Ranges", descr=""), np.arange(10)),
    # missing chr field
    (MetadataStruct(type="Ranges", descr=""), dict(start=10,
                                                   end=20,
                                                   id="1",
                                                   strand="+")),
]


@pytest.mark.parametrize("pair", GOOD_MDATASTRUCT_PAIRS)
def test_good_mdata_schemas(pair):
    descr, batch = pair
    assert descr.compatible_with(batch)


@pytest.mark.parametrize("pair", BAD_MDATASTRUCT_PAIRS)
def test_bad_mdata_schemas(pair):
    descr, batch = pair
    assert not descr.compatible_with(batch)


# --------------------------------------------
# Test the complete descriptions
# TODO
GOOD_DLSCHEMA_PAIRS = [
    ("""
inputs:
    seq:
        shape: (2, 10)
        descr: One-hot encoded DNA sequence
        special_type: bigwig
targets:
    shape: (1, )
    descr: "."
metadata:
    ranges:
        type: Ranges
        descr: "."
    dist_polya_st:
        - descr: "."
        - descr: "."
        - descr: "."
    """, {
        "inputs": {"seq": np.arange(20).reshape((1, 2, 10))},
        "targets": np.arange(1).reshape((1, 1)),
        "metadata": {
            "ranges": Ranges(chr="chr1", start=10, end=20, id="1", strand="+"),
            "dist_polya_st": [
                np.arange(1),
                np.arange(2),
                np.arange(3),
            ]
        }
    })
]


BAD_DLSCHEMA_PAIRS = [
    (GOOD_DLSCHEMA_PAIRS[0][0], {
        "inputs": {"seq": np.arange(20).reshape((1, 2, 10))},
        "targets": np.arange(1).reshape((1, 1)),
        "metadata": {
            "ranges": Ranges(chr="chr1", start=10, end=20, id="1", strand="+"),
            "dist_polya_st": [
                np.arange(1),
                np.arange(2),
                # one element missing
            ]
        }
    }),
    (GOOD_DLSCHEMA_PAIRS[0][0], {
        # one key too much
        "inputs": {"seq": np.arange(20).reshape((1, 2, 10)), "asd": np.arange(1)},
        "targets": np.arange(1).reshape((1, 1)),
        "metadata": {
            "ranges": Ranges(chr="chr1", start=10, end=20, id="1", strand="+"),
            "dist_polya_st": [
                np.arange(1),
                np.arange(2),
                np.arange(3),
            ]
        }
    }),
    (GOOD_DLSCHEMA_PAIRS[0][0], {
        "inputs": {"seq": np.arange(20).reshape((1, 2, 10))},
        "targets": np.arange(1).reshape((1, 1)),
        "metadata": {
            # ranges miss-specified
            "ranges": {"not ranges": np.arange(1)},
            "dist_polya_st": [
                np.arange(1),
                np.arange(2),
                np.arange(3),
            ]
        }
    }),
    (GOOD_DLSCHEMA_PAIRS[0][0], {
        # wrong struct
        "inputs": np.arange(20).reshape((1, 2, 10)),
        "targets": np.arange(1).reshape((1, 1)),
        "metadata": {
            # ranges miss-specified
            "ranges": Ranges(chr="chr1", start=10, end=20, id="1", strand="+"),
            "dist_polya_st": [
                np.arange(1),
                np.arange(2),
                np.arange(3),
            ]
        }
    }),
]


@pytest.mark.parametrize("pair", GOOD_DLSCHEMA_PAIRS)
def test_good_dloader_schema_pairs(pair):
    unparsed_descr, batch = pair
    descr = DataLoaderSchema.from_config(from_yaml(unparsed_descr))
    assert descr.compatible_with(batch)


@pytest.mark.parametrize("pair", BAD_DLSCHEMA_PAIRS)
def test_bad_dloader_schema_pairs(pair):
    unparsed_descr, batch = pair
    descr = DataLoaderSchema.from_config(from_yaml(unparsed_descr))
    assert not descr.compatible_with(batch)
