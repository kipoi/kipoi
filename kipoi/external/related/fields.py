from attr import attrib, NOTHING
from related import _init_fields, types
from collections import OrderedDict
from .converters import to_sequence_field_w_str, to_leaf_mapping_field


def StrSequenceField(cls, default=NOTHING, required=True, repr=False):
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


def NestedMappingField(cls, keyword, default=NOTHING, required=True, repr=False):
    """
    Create new sequence field on a model. If only string is present,
    convert it to a list of length 1.

    :param cls: class (or name) of the model to be related in Sequence.
    :param keyword: stopping condition in recursion (indicator that cls has been found)
    :param default: any TypedSequence or list
    :param bool required: whether or not the object is invalid if not provided.
    :param bool repr: include this field should appear in object's repr.
    :param bool cmp: include this field in generated comparison.
    """
    default = _init_fields.init_default(required, default, OrderedDict())
    # check that it's not sequence
    converter = to_leaf_mapping_field(cls, keyword)
    # validator = _init_fields.init_validator(required, types.TypedSequence)
    validator = None
    return attrib(default=default, convert=converter, validator=validator,
                  repr=repr)
