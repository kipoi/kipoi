from attr import attrib, NOTHING
from related import _init_fields, types
from related.types import TypedSequence, TypedMapping
from related.converters import CHILD_ERROR_MSG
from related.functions import to_model
from collections import OrderedDict


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


def to_sequence_field_w_str(cls):
    """
    Returns a callable instance that will convert a value to a Sequence.

    :param cls: Valid class type of the items in the Sequence.
    :return: instance of the SequenceConverter.
    """
    class SequenceConverter(object):

        def __init__(self, cls):
            self.cls = cls

        def __call__(self, values):
            values = values or []
            # if values is a string, consider it to be a single argument
            if isinstance(values, str):
                args = [values]
            elif isinstance(values, list):
                args = [to_model(self.cls, value) for value in values]
            else:
                raise ValueError("Field has to be a string or a list of strings")
            return TypedSequence(cls=self.cls, args=args)

    return SequenceConverter(cls)


# has description field

# keyword = "description"
def to_leaf_mapping_field(cls, keyword):
    """
    Returns a callable instance that will convert a value to a Sequence.

    :param cls: Valid class type of the items in the Sequence.
    :return: instance of the SequenceConverter.
    """
    # Input = dict
    # if type
    class LeafConverter(object):

        def __init__(self, cls):
            self.cls = cls

        def __call__(self, value):
            if keyword in value:
                # It's a normal dictionary, continue recursively
                return to_model(self.cls, value)
            else:
                if isinstance(value, list):
                    value = value or []
                    return [self.__call__(v) for v in value]
                elif isinstance(value, dict):
                    kwargs = OrderedDict()
                    for key_value, item in value.items():
                        kwargs[key_value] = self.__call__(item)
                    return kwargs
                else:
                    raise ValueError("Unable to parse: {0}".format(value))

    return LeafConverter(cls)
