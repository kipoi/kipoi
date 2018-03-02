# coding=utf-8
"""Tests modified from https://github.com/amirziai/flatten/blob/master/test_flatten.py

MIT License

Copyright (c) 2016 Amir Ziai

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
import json

try:
    # python2
    from StringIO import StringIO
except ImportError:
    # python3
    from io import StringIO

from kipoi.external.flatten_json import flatten, unflatten, unflatten_list, check_if_numbers_are_consecutive


def test_numbers_consecutive():
    """Checks if all numbers in a list are consecutive integers"""
    list_ = [1, 2, 3, 4, 5]
    assert check_if_numbers_are_consecutive(list_)

    list_ = [0, 1, 5]
    assert not check_if_numbers_are_consecutive(list_)

    list_ = [1.0, 2.0, 3.0]
    assert check_if_numbers_are_consecutive(list_)

    list_ = range(10)
    assert check_if_numbers_are_consecutive(list_)

    list_ = range(10, 0, -1)
    assert not check_if_numbers_are_consecutive(list_)


def test_no_flatten():
    dic = {'a': '1', 'b': '2', 'c': 3}
    expected = dic
    actual = flatten(dic)
    assert actual == expected


def test_one_flatten():
    dic = {'a': '1',
           'b': '2',
           'c': {'c1': '3', 'c2': '4'}
           }
    expected = {'a': '1', 'b': '2', 'c_c1': '3', 'c_c2': '4'}
    actual = flatten(dic)
    assert actual == expected


def test_one_flatten_utf8():
    dic = {'a': '1',
           u'ñ': u'áéö',
           'c': {u'c1': '3', 'c2': '4'}
           }
    expected = {'a': '1', u'ñ': u'áéö', 'c_c1': '3', 'c_c2': '4'}
    actual = flatten(dic)
    assert actual == expected


def test_one_flatten_utf8_dif():
    a = {u'eñe': 1}
    info = dict(info=a)
    expected = {u'info_{}'.format(u'eñe'): 1}
    actual = flatten(info)
    assert actual == expected


def test_custom_separator():
    dic = {'a': '1',
           'b': '2',
           'c': {'c1': '3', 'c2': '4'}
           }
    expected = {'a': '1', 'b': '2', 'c*c1': '3', 'c*c2': '4'}
    actual = flatten(dic, '*')
    assert actual == expected


def test_list():
    dic = {
        'a': 1,
        'b': [{'c': [2, 3]}]
    }
    expected = {'a': 1, 'b_0_c_0': 2, 'b_0_c_1': 3}
    actual = flatten(dic)
    assert actual == expected


def test_list_and_dict():
    dic = {
        'a': 1,
        'b': 2,
        'c': [{'d': [2, 3, 4], 'e': [{'f': 1, 'g': 2}]}]
    }
    expected = {'a': 1, 'b': 2, 'c_0_d_0': 2, 'c_0_d_1': 3, 'c_0_d_2': 4,
                'c_0_e_0_f': 1, 'c_0_e_0_g': 2}
    actual = flatten(dic)
    assert actual == expected


# def test_empty_list_and_dict():
#     dic = {
#         'a': {},
#         'b': [],
#         'c': '',
#         'd': None,
#         'e': [{'f': [], 'g': [{'h': {}, 'i': [], 'j': '', 'k': None}]}]
#     }
#     expected = {'a': {}, 'b': [], 'c': '', 'd': None,
#                 'e_0_f': [], 'e_0_g_0_h': {}, 'e_0_g_0_i': [],
#                 'e_0_g_0_j': '', 'e_0_g_0_k': None}
#     actual = flatten(dic)
#     assert actual == expected


def test_blog_example():
    dic = {
        "a": 1,
        "b": 2,
        "c": [{"d": ['2', 3, 4], "e": [{"f": 1, "g": 2}]}]
    }
    expected = {'a': 1, 'b': 2, 'c_0_d_0': '2', 'c_0_d_1': 3,
                'c_0_d_2': 4, 'c_0_e_0_f': 1,
                'c_0_e_0_g': 2}
    actual = flatten(dic)
    assert actual == expected


def test_unflatten_no_list():
    dic = {
        'a': 1,
        'b_a': 2,
        'b_b': 3,
        'c_a_b': 5
    }
    expected = {
        'a': 1,
        'b': {'a': 2, 'b': 3},
        'c': {'a': {'b': 5}}
    }
    actual = unflatten(dic)
    assert actual == expected


def test_unflatten_with_list():
    """Dictionary with lists"""
    dic = {
        'a': 1,
        'b_0': 1,
        'b_1': 2,
        'c_a': 'a',
        'c_b_0': 1,
        'c_b_1': 2,
        'c_b_2': 3
    }
    expected = {
        'a': 1,
        'b': [1, 2],
        'c': {'a': 'a', 'b': [1, 2, 3]}
    }
    actual = unflatten_list(dic)
    assert actual == expected

    dic = {'a': 1, 'b_0': 5}
    expected = {'a': 1, 'b': [5]}
    actual = unflatten_list(dic)
    assert actual == expected

    dic = {'a': 1, 'b:0': 5}
    expected = {'a': 1, 'b': [5]}
    actual = unflatten_list(dic, ':')
    assert actual == expected


def test_unflatten_with_list_custom_separator():
    """Complex dictionary with lists"""
    dic = {
        'a:b': 'str0',
        'c:0:d:0:e': 'str1',
        'c:1:d:0:e': 'str4',
        'c:1:f': 'str5',
        'c:0:f': 'str2',
        'c:1:g': 'str6',
        'c:0:g': 'str3',
        'h:d:0:e': 'str7',
        'h:i:0:f': 'str8',
        'h:i:0:g': 'str9'
    }
    expected = {
        'a': {'b': 'str0'},
        'c': [
            {
                'd': [{'e': 'str1'}],
                'f': 'str2',
                'g': 'str3'
            }, {
                'd': [{'e': 'str4'}],
                'f': 'str5',
                'g': 'str6'
            }
        ],
        'h': {
            'd': [{'e': 'str7'}],
            'i': [{'f': 'str8', 'g': 'str9'}]
        }
    }
    actual = unflatten_list(dic, ':')
    assert actual == expected


def test_unflatten_with_list_nested():
    dic = {"a": [[{"b": 1}], [{"d": 1}]]}
    dic_flatten = flatten(dic)
    actual = unflatten_list(dic_flatten)
    assert actual == dic


def test_unflatten_with_list_issue15():
    """https://github.com/amirziai/flatten/issues/15"""
    dic = {"Required": {"a": "1",
                        "b": ["1", "2", "3"],
                        "c": {"d": {"e": [[{"s1": 1}, {"s2": 2}],
                                          [{"s3": 1}, {"s4": 2}]]}},
                        "f": ["1", "2"]},
           "Optional": {"x": "1", "y": ["1", "2", "3"]}}
    dic_flatten = flatten(dic)
    actual = unflatten_list(dic_flatten)
    assert actual == dic


def test_unflatten_with_list_deep():
    dic = {'a': [
        {'b': [{'c': [{'a': 5, 'b': {'a': [1, 2, 3]}, 'c': {'x': 3}}]}]}]}
    dic_flatten = flatten(dic)
    actual = unflatten_list(dic_flatten)
    assert actual == dic
