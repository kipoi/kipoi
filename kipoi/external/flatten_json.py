"""
Modified from https://github.com/amirziai/flatten

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
import sys
import json
import collections
from collections import Iterable, OrderedDict

import six


def check_if_numbers_are_consecutive(list_):
    """
    Returns True if numbers in the list are consecutive
    :param list_: list of integers
    :return: Boolean
    """
    return all((True if second - first == 1 else False
                for first, second in zip(list_[:-1], list_[1:])))


def _construct_key(previous_key, separator, new_key):
    """
    Returns the new_key if no previous key exists, otherwise concatenates
    previous key, separator, and new_key
    :param previous_key:
    :param separator:
    :param new_key:
    :return: a string if previous_key exists and simply passes through the
    new_key otherwise
    """
    if previous_key:
        return u"{}{}{}".format(previous_key, separator, new_key)
    else:
        return new_key


# Overrides flatten
def flatten(dd, separator='_', prefix='', is_list_fn=lambda x: isinstance(x, list)):
    """Flatten a nested dictionary/list

    Args:
      separator: how to separate different hirearchical levels
      prefix: what to pre-append to the function
      is_list_fn: function to determine whether to split the list/numpy.array into indvidual classes or
        to include the element as value.
    """
    if isinstance(dd, collections.Mapping):
        return {prefix + separator + k if prefix else k: v
                for kk, vv in six.iteritems(dd)
                for k, v in six.iteritems(flatten(vv, separator, kk, is_list_fn))
                }
    elif is_list_fn(dd):
        return {prefix + separator + k if prefix else k: v
                for kk, vv in enumerate(dd)
                for k, v in six.iteritems(flatten(vv, separator, str(kk), is_list_fn))
                }
    else:
        return {prefix: dd}


def flatten_ordered(dd, separator='_', prefix='', is_list_fn=lambda x: isinstance(x, list)):
    """Flatten a nested dictionary/list

    Args:
      separator: how to separate different hirearchical levels
      prefix: what to pre-append to the function
      is_list_fn: function to determine whether to split the list/numpy.array into indvidual classes or
        to include the element as value.
    """
    if isinstance(dd, collections.Mapping):
        if not dd:
            return dd
        return OrderedDict([(prefix + separator + k if prefix else k, v)
                            for kk, vv in six.iteritems(dd)
                            for k, v in six.iteritems(flatten_ordered(vv, separator, kk, is_list_fn))
                            ])
    elif is_list_fn(dd):
        if not dd:
            return dd
        return OrderedDict([(prefix + separator + k if prefix else k, v)
                            for kk, vv in enumerate(dd)
                            for k, v in six.iteritems(flatten_ordered(vv, separator, str(kk), is_list_fn))
                            ])
    else:
        return OrderedDict([(prefix, dd)])


# def flatten(nested_dict, separator="_", root_keys_to_ignore=set()):
#     """
#     Flattens a dictionary with nested structure to a dictionary with no
#     hierarchy
#     Consider ignoring keys that you are not interested in to prevent
#     unnecessary processing
#     This is specially true for very deep objects
#     :param nested_dict: dictionary we want to flatten
#     :param separator: string to separate dictionary keys by
#     :param root_keys_to_ignore: set of root keys to ignore from flattening
#     :return: flattened dictionary
#     """
#     assert isinstance(nested_dict, dict), "flatten requires a dictionary input"
#     assert isinstance(separator, six.string_types), "separator must be string"

#     # This global dictionary stores the flattened keys and values and is
#     # ultimately returned
#     flattened_dict = dict()

#     def _flatten(object_, key):
#         """
#         For dict, list and set objects_ calls itself on the elements and for
#         other types assigns the object_ to
#         the corresponding key in the global flattened_dict
#         :param object_: object to flatten
#         :param key: carries the concatenated key for the object_
#         :return: None
#         """
#         # Empty object can't be iterated, take as is
#         if not object_:
#             flattened_dict[key] = object_
#         # These object types support iteration
#         elif isinstance(object_, dict):
#             for object_key in object_:
#                 if not (not key and object_key in root_keys_to_ignore):
#                     _flatten(object_[object_key], _construct_key(key,
#                                                                  separator,
#                                                                  object_key))
#         elif isinstance(object_, list) or isinstance(object_, set):
#             for index, item in enumerate(object_):
#                 _flatten(item, _construct_key(key, separator, index))
#         # Anything left take as is
#         else:
#             flattened_dict[key] = object_

#     _flatten(nested_dict, None)
#     return flattened_dict


flatten_json = flatten


def _unflatten_asserts(flat_dict, separator):
    assert isinstance(flat_dict, dict), "un_flatten requires dictionary input"
    assert isinstance(separator, six.string_types), "separator must be string"
    # assert all((not value or not isinstance(value, Iterable) or
    #             isinstance(value, six.string_types)
    #             for value in flat_dict.values())), "provided dict is not flat"


def unflatten(flat_dict, separator='_'):
    """
    Creates a hierarchical dictionary from a flattened dictionary
    Assumes no lists are present
    :param flat_dict: a dictionary with no hierarchy
    :param separator: a string that separates keys
    :return: a dictionary with hierarchy
    """
    _unflatten_asserts(flat_dict, separator)

    # This global dictionary is mutated and returned
    unflattened_dict = dict()

    def _unflatten(dic, keys, value):
        for key in keys[:-1]:
            dic = dic.setdefault(key, {})

        dic[keys[-1]] = value

    for item in flat_dict:
        _unflatten(unflattened_dict, item.strip(separator).split(separator), flat_dict[item])

    return unflattened_dict


def unflatten_list(flat_dict, separator='_'):
    """
    Unflattens a dictionary, first assuming no lists exist and then tries to
    identify lists and replaces them
    This is probably not very efficient and has not been tested extensively
    Feel free to add test cases or rewrite the logic
    Issues that stand out to me:
    - Sorting all the keys in the dictionary, which specially for the root
    dictionary can be a lot of keys
    - Checking that numbers are consecutive is O(N) in number of keys
    :param flat_dict: dictionary with no hierarchy
    :param separator: a string that separates keys
    :return: a dictionary with hierarchy
    """
    _unflatten_asserts(flat_dict, separator)

    # First unflatten the dictionary assuming no lists exist
    unflattened_dict = unflatten(flat_dict, separator)

    def _convert_dict_to_list(object_, parent_object, parent_object_key):
        if isinstance(object_, dict):
            try:
                keys = [int(key) for key in object_]
                keys.sort()
            except (ValueError, TypeError):
                keys = []
            keys_len = len(keys)

            if (keys_len > 0 and sum(keys) ==
                int(((keys_len - 1) * keys_len) / 2) and keys[0] == 0 and
                    keys[-1] == keys_len - 1 and
                    check_if_numbers_are_consecutive(keys)):

                # The dictionary looks like a list so we're going to replace it
                parent_object[parent_object_key] = []
                for key_index, key in enumerate(keys):
                    parent_object[parent_object_key].append(object_[str(key)])
                    # The list item we just added might be a list itself
                    # https://github.com/amirziai/flatten/issues/15
                    _convert_dict_to_list(parent_object[parent_object_key][-1],
                                          parent_object[parent_object_key],
                                          key_index)

            for key in object_:
                if isinstance(object_[key], dict):
                    _convert_dict_to_list(object_[key], object_, key)

    _convert_dict_to_list(unflattened_dict, None, None)
    return unflattened_dict
