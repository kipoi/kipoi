"""Tests for functions in data_utils.py.


-----------------------
COPYRIGHT

All contributions by François Chollet:
Copyright (c) 2015 - 2018, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015 - 2018, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017 - 2018, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import sys
import tarfile
import zipfile
import multiprocessing as mp
import pytest
import six
from six.moves.urllib.parse import urljoin
from six.moves.urllib.request import pathname2url

from kipoi.external.keras.data_utils import _hash_file
from kipoi.external.keras.data_utils import get_file
from kipoi.external.keras.utils.data_utils import validate_file

if sys.version_info < (3,):
    def next(x):
        return x.next()


def use_spawn(func):
    """Decorator to test both Unix (fork) and Windows (spawn)"""
    @six.wraps(func)
    def wrapper(*args, **kwargs):
        out = func(*args, **kwargs)
        if sys.version_info > (3, 4):
            mp.set_start_method('spawn', force=True)
            func(*args, **kwargs)
            mp.set_start_method('fork', force=True)
        return out
    return wrapper


@pytest.fixture
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
