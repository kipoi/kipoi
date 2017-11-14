from attr import attrib, NOTHING
from related import _init_fields, types
from collections import OrderedDict
from .converters import to_sequence_field_w_str, to_leaf_mapping_field, to_eval_str, identity
from . import dispatchers  # to load the dispatcher


class UNSPECIFIED(object):
    pass


def AnyField(default=NOTHING, required=True, repr=True):
    """
    Just pass through the field, using default yaml conversion to python objects

    :param cls: class (or name) of the model to be related in Sequence.
    :param default: any TypedSequence or list
    :param bool required: whether or not the object is invalid if not provided.
    :param bool repr: include this field should appear in object's repr.
    :param bool cmp: include this field in generated comparison.
    """
    default = _init_fields.init_default(required, default, UNSPECIFIED())
    return attrib(default=default, convert=None, validator=None,
                  repr=repr)


def StrSequenceField(cls, default=NOTHING, required=True, repr=True):
    """
    Create new sequence field on a model. If only string is present,
    convert it to a list of length 1.

    :param cls: class (or name) of the model to be related in Sequence.
    :param default: any TypedSequence or list
    :param bool required: whether or not the object is invalid if not provided.
    :param bool repr: include this field should appear in object's repr.
    :param bool cmp: include this field in generated comparison.
    """
    default = _init_fields.init_default(required, default, [])
    # check that it's not sequence
    converter = to_sequence_field_w_str(cls)
    validator = _init_fields.init_validator(required, types.TypedSequence)
    return attrib(default=default, convert=converter, validator=validator,
                  repr=repr)


def NestedMappingField(cls, keyword, key, default=NOTHING, required=True, repr=False):
    """
    Create new sequence field on a model. If only string is present,
    convert it to a list of length 1.

    :param cls: class (or name) of the model to be related in Sequence.
    :param keyword: stopping condition in recursion (indicator that cls has been found)
    :param key: key field on the child object to be used as the mapping key.
    :param default: any TypedSequence or list
    :param bool required: whether or not the object is invalid if not provided.
    :param bool repr: include this field should appear in object's repr.
    :param bool cmp: include this field in generated comparison.
    """
    default = _init_fields.init_default(required, default, OrderedDict())
    # check that it's not sequence
    converter = to_leaf_mapping_field(cls, keyword, key)
    # validator = _init_fields.init_validator(required, types.TypedSequence)
    validator = None
    return attrib(default=default, convert=converter, validator=validator,
                  repr=repr)


def TupleIntField(default=NOTHING, required=True, repr=True):
    """
    Create new tuple field on a model. Convert it first to a string
    and then to a tuple

    :param cls: class (or name) of the model to be related in Sequence.
    :param default: any TypedSequence or list
    :param bool required: whether or not the object is invalid if not provided.
    :param bool repr: include this field should appear in object's repr.
    :param bool cmp: include this field in generated comparison.
    """
    default = _init_fields.init_default(required, default, tuple)
    converter = to_eval_str
    validator = _init_fields.init_validator(required, tuple)
    return attrib(default=default, convert=converter, validator=validator,
                  repr=repr)
