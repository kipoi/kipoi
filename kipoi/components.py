"""Defines the base classes
"""
import related
import enum
from collections import OrderedDict
import kipoi.conda as kconda
from kipoi.external.related.fields import StrSequenceField, NestedMappingField, TupleIntField
# TODO additionally validate the special type properties


# --------------------------------------------
# Abstract classes & Mixins for models defined using related

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


class RelatedLoadSaveMixin(RelatedConfigMixin):
    """Adds load and dump on top of RelatedConfigMixin for reading and writing from a yaml file
    """

    @classmethod
    def load(cls, path):
        """Loads model from a yaml file
        """
        original_yaml = open(path).read().strip()
        parsed_dict = related.from_yaml(original_yaml)
        if "path" not in parsed_dict:
            parsed_dict["path"] = path
        return cls.from_config(parsed_dict)

    def dump(self, path):
        """Dump the object to a yaml file
        """
        generated_yaml = related.to_yaml(self,
                                         suppress_empty_values=True,
                                         suppress_map_key_values=True)  # .strip()
        with open(path, "w") as f:
            f.write(generated_yaml)


# --------------------------------------------
# Common components (model and dataloader)

@related.immutable
class Info(RelatedConfigMixin):
    """Class holding information about the component.
    Parses the info section in component.yaml:

    info:
      author: Ziga Avsec
      name: rbp_eclip
      version: 0.1
      descr: RBP binding prediction
    """
    author = related.StringField()
    name = related.StringField()
    version = related.StringField()
    descr = related.StringField()
    tags = StrSequenceField(str, default=[], required=False)


@enum.unique
class ArraySpecialType(enum.Enum):
    DNASeq = "DNASeq"
    BIGWIG = "bigwig"
    VPLOT = "v-plot"
    Array = "Array"


@related.immutable
class ArraySchema(RelatedConfigMixin):
    """

    Args:
      shape: Tuple of shape (same as in Keras for the input)
      descr: Description of the array
      special_type: str, special type name. Could also be an array of special entries?
      metadata_entries: str or list of metadata
    """
    shape = TupleIntField()
    descr = related.StringField()
    # MAYBE - allow a list of strings?
    #         - could be useful when a single array can have multiple 'attributes'
    name = related.StringField(required=False)
    special_type = related.ChildField(ArraySpecialType, required=False)
    associated_metadata = StrSequenceField(str, default=[], required=False)
    # TODO shall we have
    # - associated_metadata in ArraySchema
    # OR
    # - associated_array in MetadataField?


# --------------------------------------------
# Model specific components

@related.immutable
class ModelSchema(RelatedConfigMixin):
    """Describes the model schema
    """
    # can be a dictionary, list or a single array
    inputs = NestedMappingField(ArraySchema, keyword="shape", key="name")
    targets = NestedMappingField(ArraySchema, keyword="shape", key="name")


# --------------------------------------------
# DataLoader specific components

@enum.unique
class MetadataType(enum.Enum):
    # TODO - make capital
    RANGES = "Ranges"
    STR = "str"
    INT = "int"
    FLOAT = "float"
    ARRAY = "array"
    # TODO - add bed3 or bed6 ranges


@related.immutable
class MetadataStruct(RelatedConfigMixin):

    descr = related.StringField()
    type = related.ChildField(MetadataType, required=False)
    name = related.StringField(required=False)


@related.immutable
class DataLoaderSchema(RelatedConfigMixin):
    """Describes the model schema
    """
    inputs = NestedMappingField(ArraySchema, keyword="shape", key="name")
    targets = NestedMappingField(ArraySchema, keyword="shape", key="name", required=False)
    metadata = NestedMappingField(MetadataStruct, keyword="descr", key="name",
                                  required=False)
    #      - we would need to allow classes that contain also dictionaries
    #        -> leaf can be an
    #          - array
    #          - scalar
    #          - custom dictionary (recursive data-type)
    #          - SpecialType (say ArrayRanges, BatchArrayRanges, which will
    #                         effectively be a dicitonary of scalars)


@related.immutable
class DataLoaderArgument(RelatedConfigMixin):
    # MAYBE - make this a general argument class
    descr = related.StringField()
    name = related.StringField(required=False)
    type = related.StringField(default='str', required=False)
    optional = related.BooleanField(default=False, required=False)
    tags = StrSequenceField(str, default=[], required=False)  # TODO - restrict the tags


@related.immutable
class Dependencies(object):
    conda = related.SequenceField(str, default=[], required=False)
    pip = related.SequenceField(str, default=[], required=False)

    def install_pip(self, dry_run=False):
        print("pip dependencies to be installed:")
        print(self.pip)
        if dry_run:
            return
        else:
            kconda.install_pip(self.pip)

    def install_conda(self, dry_run=False):
        print("Conda dependencies to be installed:")
        print(self.conda)
        if dry_run:
            return
        else:
            kconda.install_conda(self.conda)

    def install(self, dry_run=False):
        self.install_conda(dry_run)
        self.install_pip(dry_run)


# --------------------------------------------
# Final description classes modelling the yaml files
@related.immutable
class ModelDescription(RelatedLoadSaveMixin):
    """Class representation of model.yaml
    """
    type = related.StringField()
    args = related.ChildField(dict)
    info = related.ChildField(Info)
    schema = related.ChildField(ModelSchema)
    default_dataloader = related.StringField(default='.')
    dependencies = related.ChildField(Dependencies,
                                      default=Dependencies(),
                                      required=False)
    path = related.StringField(required=False)
    # TODO - add after loading validation for the arguments class?


@related.immutable
class DataLoaderDescription(RelatedLoadSaveMixin):
    """Class representation of dataloader.yaml
    """
    type = related.StringField()
    defined_as = related.StringField()
    args = related.MappingField(DataLoaderArgument, "name")
    info = related.ChildField(Info)
    output_schema = related.ChildField(DataLoaderSchema)
    dependencies = related.ChildField(Dependencies, default=Dependencies(), required=False)
    path = related.StringField(required=False)


# TODO - special metadata classes should just extend the dictionary field
# (to be fully compatible with batching etc)
