import pytest
from kipoi.external.concise.seqplotting_deps import encodeDNA
from kipoi.external.concise.seqplotting_deps import (pad_sequences, DNA, tokenize,
                                            token2one_hot, encodeSequence)
import numpy as np


def test_encodeSequence():
    assert encodeSequence(["ACG", "TGT"], DNA, "N").shape == (2, 3, 4)

    et = encodeSequence(["ACG", "TGT"], DNA, "N", encode_type="token")
    assert et.shape == (2, 3)
    assert np.array_equal(et, np.array([[1, 2, 3],
                                        [4, 3, 4]]))

    # TODO - should I increase the index by one?
    et = encodeSequence(["ACG", "TGTN"], DNA, "N", encode_type="token")
    assert np.array_equal(et, np.array([[1, 2, 3, 0],
                                        [4, 3, 4, 0]]))

def test_tokenize():
    assert tokenize("ACGTTA", DNA, neutral_vocab="N") == [0, 1, 2, 3, 3, 0]
    assert tokenize("ACGTGATGA", ["ACG", "TGA"], neutral_vocab="NNN") == [0, 1, 1]
    assert tokenize("ACGTGATGA", ["ACG"], neutral_vocab="TGA") == [0, -1, -1]
    with pytest.raises(Exception):
        tokenize("ACGTGATGA", ["ACG"], neutral_vocab="NNN")


def test_encodeDNA():

    seq = "ACGTTTATNT"
    assert len(seq) == 10

    with pytest.raises(ValueError):
        encodeDNA(seq)

    assert encodeDNA([seq]).shape == (1, 10, 4)

    assert encodeDNA([seq], maxlen=20).shape == (1, 20, 4)

    assert encodeDNA([seq], maxlen=5).shape == (1, 5, 4)
    assert np.all(encodeDNA([seq])[0, 0] == np.array([1, 0, 0, 0]))
    assert np.all(encodeDNA([seq])[0, 1] == np.array([0, 1, 0, 0]))
    assert np.all(encodeDNA([seq])[0, 2] == np.array([0, 0, 1, 0]))
    assert np.all(encodeDNA([seq])[0, 3] == np.array([0, 0, 0, 1]))
    assert np.all(encodeDNA([seq])[0, 4] == np.array([0, 0, 0, 1]))
    assert np.all(encodeDNA([seq])[0, -1] == np.array([0, 0, 0, 1]))
    assert np.all(encodeDNA([seq])[0, -2] == np.array([0, 0, 0, 0]))


def test_token2one_hot():
    assert np.array_equal(token2one_hot([0, 1, -1], 2), np.array([[1, 0],
                                                                  [0, 1],
                                                                  [0, 0]]))


def test_pad_sequences():
    sequence_vec = ["ACGTTTATNT"]
    assert len(pad_sequences(sequence_vec, value="N",
                             maxlen=20, align="end")[0]) is 20

    # works with lists
    assert pad_sequences([[1, 2, 3], [2, 2, 3, 4], [31, 3], [4, 2]], value=[0],
                         maxlen=5) == [[0, 0, 1, 2, 3],
                                       [0, 2, 2, 3, 4],
                                       [0, 0, 0, 31, 3],
                                       [0, 0, 0, 4, 2]]

    assert pad_sequences([[1, 2, 3], [2, 2, 3, 4], [31, 3], [4, 2]], value=[0],
                         maxlen=2, align="end") == [[2, 3],
                                                    [3, 4],
                                                    [31, 3],
                                                    [4, 2]]

    assert pad_sequences([[1, 2, 3], [2, 2, 3, 4], [31, 3], [4, 2]], value=[0],
                         maxlen=2, align="start") == [[1, 2],
                                                      [2, 2],
                                                      [31, 3],
                                                      [4, 2]]

    # expect error
    with pytest.raises(ValueError):
        pad_sequences(sequence_vec, value="NNN", align="end")

    sequence_vec = ["ACGTTTATC"]
    assert len(pad_sequences(sequence_vec, value="NNN", align="end")[0]) == 9
    with pytest.raises(ValueError):
        pad_sequences(sequence_vec, value="NNN", maxlen=10, align="end")
    assert len(pad_sequences(sequence_vec, value="NNN", maxlen=12, align="end")[0]) == 12



def test_pad_sequences_2():
    sequence_vec = ['CTTACTCAGA', 'TCTTTA']
    assert pad_sequences(sequence_vec, 10, align="start", value="N") == ['CTTACTCAGA', 'TCTTTANNNN']
    assert pad_sequences(sequence_vec, 10, align="end", value="N") == ['CTTACTCAGA', 'NNNNTCTTTA']
    assert pad_sequences(sequence_vec, 4, align="center", value="N") == ['ACTC', 'CTTT']
