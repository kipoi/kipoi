"""Defines the base classes
"""


class Info():
    """Class holding information about the component. Parses the info section in component.yaml:

    info:
      author: Ziga Avsec
      name: rbp_eclip
      version: 0.1
      description: RBP binding prediction
    """

    # TODO - prevent code copying - www.attrs.org/?
    REQ_FIELDS = ["author", "name", "version", "description"]
    OPT_FIELDS = ["tags"]

    def __init__(self, author, name, version, description, tags=[]):
        self.author = author
        self.name = name
        self.version = version
        self.description = description
        self.tags = tags

    def __repr__(self):
        return "{cls}(author='{a}', name='{n}', version='{v}', description='{d}', tags={t})".\
            format(cls=self.__class__.__name__,
                   a=self.author,
                   n=self.name,
                   v=self.version,
                   d=self.description,
                   t=self.tags)

    @classmethod
    def from_config(cls, kwargs):

        # Verbose unrecognized field
        for k in kwargs.keys():
            if k not in cls.REQ_FIELDS + cls.OPT_FIELDS:
                raise ValueError("Unrecognized field in info: '{f}'. Avaiable fields are: {l}".
                                 format(f=k, l=cls.REQ_FIELDS))

        # Verbose undefined field
        undefined_set = set(cls.REQ_FIELDS) - kwargs.keys()
        if undefined_set:
            raise ValueError("The following arguments were not specified: {0}. Please specify them.".
                             format(undefined_set))

        return Info(**kwargs)

    def get_config(self, ):
        return {"author": self.author,
                "name": self.name,
                "version": self.version,
                "description": self.description,
                "tags": self.tags}


#
class ArraySchema():
    def __init__(self, shape, description, special_type=None, metadata_entries=[]):
        """

        Args:
          shape: Tuple of shape (same as in Keras for the input)
          description: Description of the array
          special_type: str, special type name. Could also be an array of special entries?
          metadata_entries: str or list of metadata
        """
        self.shape = shape
        self.description = description
        self.special_type = special_type

        me = metadata_entries
        self.metadata = [me] if isinstance(me, str) else list(me)


class ModelSchema():

    def __init__(self, inputs, targets, metadata=None, ):
        """

        """
        self.inputs = inputs
        self.targets = targets
        self.metadata = metadata


class DataLoaderSchema():

    def __init__(self, arguments, inputs, targets, metadata=None, ):
        """

        """
        self.arguments = arguments
        self.inputs = inputs
        self.targets = targets
        self.metadata = metadata


# allways requirespecify:
# - shape
# - description
