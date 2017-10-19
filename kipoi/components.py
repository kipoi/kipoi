"""Defines the base classes
"""
import related
import enum
from .fields import StrSequenceField
# TODO additionally validate the special type properties


class RelatedConfigMixin(object):
    """Provides from_config and get_config to @related.immutable decorated classes
    """
    @classmethod
    def from_config(cls, cfg):
        # TODO - create a nicer error message - see above
                # Verbose unrecognized field
        # for k in kwargs.keys():
        #     if k not in cls.REQ_FIELDS + cls.OPT_FIELDS:
        #         raise ValueError("Unrecognized field in info: '{f}'. Avaiable fields are: {l}".
        #                          format(f=k, l=cls.REQ_FIELDS))

        # # Verbose undefined field
        # undefined_set = set(cls.REQ_FIELDS) - kwargs.keys()
        # if undefined_set:
        #     raise ValueError("The following arguments were not specified: {0}. Please specify them.".
        #                      format(undefined_set))
        return related.to_model(cls, cfg)

    def get_config(self):
        return related.to_dict(self)


@related.immutable
class Info(RelatedConfigMixin):
    """Class holding information about the component.
    Parses the info section in component.yaml:

    info:
      author: Ziga Avsec
      name: rbp_eclip
      version: 0.1
      description: RBP binding prediction
    """
    author = related.StringField()
    name = related.StringField()
    version = related.StringField()
    description = related.StringField()
    tags = related.SequenceField(str, default=[], required=False)


@enum.unique
class ArraySpecialType(enum.Enum):
    DNASeq = "DNASeq"
    BIGWIG = "bigwig"


@related.immutable
class ArraySchema(RelatedConfigMixin):
    """

    Args:
      shape: Tuple of shape (same as in Keras for the input)
      description: Description of the array
      special_type: str, special type name. Could also be an array of special entries?
      metadata_entries: str or list of metadata
    """
    shape = related.ChildField(tuple)
    description = related.StringField()
    # MAYBE - allow a list of strings?
    #         - could be useful when a single array can have multiple 'attributes'
    name = related.StringField(required=False)
    special_type = related.ChildField(ArraySpecialType, required=False)
    associated_metadata = StrSequenceField(str, default=[], required=False)


@related.immutable
class ModelSchema(RelatedConfigMixin):
    """Describes the model schema
    """
    inputs = related.MappingField(ArraySchema, "name")
    # TODO - can be a dictionary, list or a single array
    targets = related.MappingField(ArraySchema, "name")


@related.immutable
class DataLoaderSchema(RelatedConfigMixin):
    """Describes the model schema
    """
    inputs = related.MappingField(ArraySchema, "name")
    # TODO - can be a dictionary, list or a single array
    targets = related.MappingField(ArraySchema, "name", required=False)
    # TODO - define a special metadata field
    #       - nested data structure
    metadata = related.MappingField(ArraySchema, "name", required=False)

# TODO
# - DataLoaderArgs
# - KipoiDataloader
# - KipoiModel
