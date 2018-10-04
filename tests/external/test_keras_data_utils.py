"""Tests for functions in data_utils.py.
"""
import os
import sys
import tarfile
import threading
import zipfile
import multiprocessing as mp
import numpy as np
import pytest
import six
from six.moves.urllib.parse import urljoin
from six.moves.urllib.request import pathname2url

from kipoi.external.keras.data_utils import _hash_file, get_file, validate_file
from kipoi.external.keras.generic_utils import Progbar
from pytest import fixture


@fixture
def in_tmpdir(tmpdir):
    """Runs a function in a temporary directory.
    Checks that the directory is empty afterwards.
    """
    with tmpdir.as_cwd():
        yield None
    assert not tmpdir.listdir()


def test_data_utils(in_tmpdir):
    """Tests get_file from a url, plus extraction and validation.
    """
    dirname = 'data_utils'

    with open('test.txt', 'w') as text_file:
        text_file.write('Float like a butterfly, sting like a bee.')

    with tarfile.open('test.tar.gz', 'w:gz') as tar_file:
        tar_file.add('test.txt')

    with zipfile.ZipFile('test.zip', 'w') as zip_file:
        zip_file.write('test.txt')

    origin = urljoin('file://', pathname2url(os.path.abspath('test.tar.gz')))

    path = get_file(dirname, origin, untar=True)
    filepath = path + '.tar.gz'
    hashval_sha256 = _hash_file(filepath)
    hashval_md5 = _hash_file(filepath, algorithm='md5')
    path = get_file(dirname, origin, md5_hash=hashval_md5, untar=True)
    path = get_file(filepath, origin, file_hash=hashval_sha256, extract=True)
    assert os.path.exists(filepath)
    assert validate_file(filepath, hashval_sha256)
    assert validate_file(filepath, hashval_md5)
    os.remove(filepath)
    os.remove('test.tar.gz')

    origin = urljoin('file://', pathname2url(os.path.abspath('test.zip')))

    hashval_sha256 = _hash_file('test.zip')
    hashval_md5 = _hash_file('test.zip', algorithm='md5')
    path = get_file(dirname, origin, md5_hash=hashval_md5, extract=True)
    path = get_file(dirname, origin, file_hash=hashval_sha256, extract=True)
    assert os.path.exists(path)
    assert validate_file(path, hashval_sha256)
    assert validate_file(path, hashval_md5)

    os.remove(path)
    os.remove('test.txt')
    os.remove('test.zip')


def test_progbar():
    values_s = [None,
                [['key1', 1], ['key2', 1e-4]],
                [['key3', 1], ['key2', 1e-4]]]

    for target in (len(values_s) - 1, None):
        for verbose in (0, 1, 2):
            bar = Progbar(target, width=30, verbose=verbose, interval=0.05)
            for current, values in enumerate(values_s):
                bar.update(current, values=values)
