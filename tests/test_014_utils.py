"""Test kipoi.utils
"""
import numpy as np
from kipoi.utils import flatten_nested, take_first_nested, map_nested
from pytest import fixture


@fixture
def nested_dict():
    return {"a": 1,
            "b": {
                "c": 3,
                "d": [1, 2, 3],
                "e": [
                    {"f": 1},
                    {"g": 4}]
            }}


@fixture
def nested_dict2():
    return {"a": np.arange(3),
            "b": {
                "c": 3,
                "d": [1, 2, 3],
                "e": [
                    {"f": 1},
                    {"g": 4}]
    }}


nested_dict = nested_dict()
nested_dict2 = nested_dict2()


def test_flatten_dict(nested_dict):
    fd = flatten_nested(nested_dict)
    assert fd == {'a': 1,
                  'b_c': 3,
                  'b_d_0': 1,
                  'b_d_1': 2,
                  'b_d_2': 3,
                  'b_e_0_f': 1,
                  'b_e_1_g': 4}


def test_flatten_dict_no_list(nested_dict):
    fd = flatten_nested(nested_dict,
                        is_list_fn=lambda x: False)
    assert fd == {'a': 1, 'b_c': 3,
                  'b_d': [1, 2, 3],
                  'b_e': [{'f': 1},
                          {'g': 4}]}


def test_take_first_nested(nested_dict, nested_dict2):
    assert take_first_nested(nested_dict) == 1
    assert np.all(take_first_nested(nested_dict2) == np.arange(3))


def test_map_nested(nested_dict, nested_dict2):
    assert map_nested(nested_dict, str)['b']['c'] == "3"
    assert map_nested(nested_dict2, lambda x: isinstance(x, np.ndarray))['a']
    assert not map_nested(nested_dict2, lambda x: isinstance(x, np.ndarray))['b']['c']
