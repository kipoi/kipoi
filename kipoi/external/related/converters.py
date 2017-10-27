from collections import OrderedDict
from related.functions import to_model
from related.types import TypedSequence


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
