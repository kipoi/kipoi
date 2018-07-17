"""Test kipoi.utils
"""
import numpy as np
from collections import OrderedDict
import kipoi
from kipoi.utils import take_first_nested, map_nested
from kipoi.external.flatten_json import flatten, flatten_ordered, unflatten_list
from pytest import fixture


@fixture
def nested_dict():
    return OrderedDict([("a", 1),
                        ("b", {
                            "c": 3,
                            "d": [1, 2, 3],
                            "e": [
                                {"f": 1},
                                {"g": 4}]
                        })])


@fixture
def nested_dict2():
    return OrderedDict([("a", np.arange(3)),
                        ("b", {
                            "c": 3,
                            "d": [1, 2, 3],
                            "e": [
                                {"f": 1},
                                {"g": 4}]
                        })])


# nested_dict = nested_dict()
# nested_dict2 = nested_dict2()


def test_flatten_dict(nested_dict):
    fd = flatten(nested_dict)

    assert dict(fd) == {'a': 1,
                        'b_c': 3,
                        'b_d_0': 1,
                        'b_d_1': 2,
                        'b_d_2': 3,
                        'b_e_0_f': 1,
                        'b_e_1_g': 4}
    assert unflatten_list(fd) == dict(nested_dict)


def test_flatten_ordered_dict(nested_dict):
    fd = flatten_ordered(nested_dict)
    assert dict(fd) == {'a': 1,
                        'b_c': 3,
                        'b_d_0': 1,
                        'b_d_1': 2,
                        'b_d_2': 3,
                        'b_e_0_f': 1,
                        'b_e_1_g': 4}
    assert unflatten_list(fd) == dict(nested_dict)


# def test_flatten_dict_no_list(nested_dict):
#     fd = flatten(nested_dict)
#     assert fd == {'a': 1, 'b_c': 3,
#                   'b_d': [1, 2, 3],
#                   'b_e': [{'f': 1},
#                           {'g': 4}]}


def test_take_first_nested(nested_dict, nested_dict2):
    assert take_first_nested(nested_dict) == 1

    assert np.all(take_first_nested(nested_dict2) == np.arange(3))


def test_map_nested(nested_dict, nested_dict2):
    assert map_nested(nested_dict, str)['b']['c'] == "3"
    assert map_nested(nested_dict2, lambda x: isinstance(x, np.ndarray))['a']
    assert not map_nested(nested_dict2, lambda x: isinstance(x, np.ndarray))['b']['c']


def test_compare_numpy_dict():
    obj1 = {"a": np.arange(3),
            "b": [np.arange(4)]}
    obj2 = {"a": np.arange(3),
            "b": [np.arange(4)]}
    obj3 = {"d": np.arange(3),
            "b": [np.arange(4)]}
    assert kipoi.utils.compare_numpy_dict(obj1, obj2)
    assert not kipoi.utils.compare_numpy_dict(obj1, obj3)
