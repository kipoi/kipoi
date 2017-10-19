from attr import attrib, NOTHING
from related import _init_fields, types
from related.types import TypedSequence
from related.functions import to_model


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
            else:
                args = [to_model(self.cls, value) for value in values]
            return TypedSequence(cls=self.cls, args=args)

    return SequenceConverter(cls)
