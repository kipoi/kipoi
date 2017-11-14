from collections import OrderedDict
from related.functions import to_model
from related.types import TypedSequence
from six import string_types


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
def to_leaf_mapping_field(cls, keyword, key):
    """
    Returns a callable instance that will convert a value to a Sequence.

    :param cls: Valid class type of the items in the Sequence.
    :param key: Attribute name of the key value in each item of cls instance.
    :return: instance of the SequenceConverter.
    """
    # Input = dict
    # if type
    class LeafConverter(object):

        def __init__(self, cls, key):
            self.cls = cls
            self.key = key

        def __call__(self, value, cur_key=None):
            """Cur_key: current key
            """
            if keyword in value:
                # Stop the recursion - the leaf keyword was found
                if cur_key is not None:
                    if self.key in value:
                        assert value[self.key] == cur_key
                    else:
                        value[self.key] = cur_key

                return to_model(self.cls, value)
            else:
                if isinstance(value, list):
                    value = value or []
                    return [self.__call__(v) for v in value]
                elif isinstance(value, dict):
                    kwargs = OrderedDict()
                    for key_value, item in value.items():
                        kwargs[key_value] = self.__call__(item, cur_key=key_value)
                    return kwargs
                else:
                    raise ValueError("Unable to parse: {0}".format(value))

    return LeafConverter(cls, key)


def to_eval_str(value):
    """
    Returns an eval(str(value)) if the value is not None.

    :param value: None or a value that can be converted to a str.
    :return: None or str(value)
    """
    if isinstance(value, string_types):
        value = eval(str(value))

    return value


def identity(x):
    """Simple identity
    """
    return x
