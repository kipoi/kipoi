"""Test 
"""
import os
from kipoi_utils.utils import list_files_recursively


def test_list_files_recursively():
    a = list_files_recursively("example", 'model')
    b = list_files_recursively("example/", 'model')
    c = list_files_recursively(os.path.abspath("example"), 'model')
    assert a == b
    assert a == c
    assert a[0].startswith("models/")
    assert all([x.endswith("model.yaml") for x in a])
    assert all([os.path.exists(os.path.join("example", x)) for x in a])
