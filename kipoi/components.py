"""Defines the base classes
"""
from __future__ import absolute_import
from __future__ import print_function

import collections
import logging
import os
from collections import OrderedDict

import enum
import numpy as np
import related
import six
from attr._make import fields

import kipoi.conda as kconda
from kipoi.external.related.mixins import RelatedConfigMixin, RelatedLoadSaveMixin
from kipoi.external.related.fields import StrSequenceField, NestedMappingField, TupleIntField, AnyField, UNSPECIFIED
from kipoi.metadata import GenomicRanges
from kipoi.utils import unique_list, yaml_ordered_dump, read_txt
from kipoi import postprocessing
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# --------------------------------------------
# Common components (model and dataloader)


@related.immutable(strict=True)
class Author(RelatedConfigMixin):
    name = related.StringField()
    github = related.StringField(required=False)
    email = related.StringField(required=False)


@related.immutable(strict=True)
class Info(RelatedConfigMixin):
    """Class holding information about the component.
    Parses the info section in component.yaml:

    info:
      authors:
        - name: Ziga Avsec
      doc: RBP binding prediction
      name: rbp_eclip
      version: 0.1
    """
    authors = related.SequenceField(Author, repr=True)
    doc = related.StringField("", required=False)  # free-text description of the model
    name = related.StringField(required=False)  # TODO - deprecate
    version = related.StringField(default="0.1", required=False)
    license = related.StringField(default="MIT", required=False)  # license of the model/dataloader - defaults to MIT
    tags = StrSequenceField(str, default=[], required=False)

    def __attrs_post_init__(self):
        if self.doc == "":
            logger.warn("doc empty for the `info:` field")


@related.immutable(strict=True)
class ModelInfo(Info):
    """Additional information for the model - not applicable to the dataloader
    """
    contributors = related.SequenceField(Author, default=[], repr=True, required=False)
    cite_as = related.StringField(required=False)  # a link or a description how to cite the paper (say a doi link)
    trained_on = related.StringField(required=False)  # a link or a description of the training dataset
    training_procedure = related.StringField(required=False)  # brief description about the training procedure for the trained_on dataset.


@enum.unique
class ArraySpecialType(enum.Enum):
    DNASeq = "DNASeq"
    DNAStringSeq = "DNAStringSeq"
    BIGWIG = "bigwig"
    VPLOT = "v-plot"
    Array = "Array"


@related.immutable(strict=True)
class ArraySchema(RelatedConfigMixin):
    """

    Args:
      shape: Tuple of shape (same as in Keras for the input)
      doc: Description of the array
      special_type: str, special type name. Could also be an array of special entries?
      metadata_entries: str or list of metadata
    """
    verbose = True
    shape = TupleIntField()
    doc = related.StringField("", required=False)
    # MAYBE - allow a list of strings?
    #         - could be useful when a single array can have multiple 'attributes'
    name = related.StringField(required=False)
    special_type = related.ChildField(ArraySpecialType, required=False)
    associated_metadata = StrSequenceField(str, default=[], required=False)
    column_labels = StrSequenceField(str, default=[], required=False)  # either a list or a path to a file --> need to check whether it's a list
    # TODO shall we have
    # - associated_metadata in ArraySchema
    # OR
    # - associated_array in MetadataField?

    # assert that there are no Nones in the shape, assume that channels is the only 4 or it is the last
    # update the model schema shape on calling batch_iter method
    # overwrite the batch_iter method of the returned dataloader --> decorator needed

    def print_msg(self, msg):
        if self.verbose:
            print("ArraySchema mismatch")
            print(msg)

    def _validate_list_column_labels(self):
        dim_ok = len(self.shape) >= 1
        if dim_ok and (self.shape[0] is not None):
            dim_ok &= len(self.column_labels) == self.shape[0]
        if not dim_ok:
            self.print_msg("Column annotation does not match array dimension with shape %s and %d labels (%s ...)"
                           % (str(self.shape), len(self.column_labels), str(self.column_labels)[:30]))

    def __attrs_post_init__(self):
        if len(self.column_labels) > 1:
            # check that length is ok with columns
            self._validate_list_column_labels()
        elif len(self.column_labels) == 1:
            label = self.column_labels.list[0]
            import os
            # check if path exists raise exception only test time, but only a warning in prediction time
            if os.path.exists(label):
                with open(label, "r") as ifh:
                    object.__setattr__(self, "column_labels", [l.rstrip() for l in ifh])
            self._validate_list_column_labels()
        else:
            object.__setattr__(self, "column_labels", None)

    def compatible_with_batch(self, batch, verbose=True):
        """Checks compatibility with a particular batch of data

        Args:
          batch: numpy array
          ignore_batch_axis: if True, the batch axis is not considered
          verbose: print the fail reason
        """
        def print_msg(msg):
            if verbose:
                print("ArraySchema mismatch")
                print(msg)

        # type = np.ndarray
        if not isinstance(batch, np.ndarray):
            print_msg("Expecting a np.ndarray. Got type(batch) = {0}".format(type(batch)))
            return False

        if not batch.ndim >= 1:
            print_msg("The array is a scalar (expecting at least the batch dimension)")
            return False

        return self.compatible_with_schema(ArraySchema(shape=batch.shape[1:],
                                                       doc=""))

    def compatible_with_schema(self, schema, name_self="", name_schema="", verbose=True):
        """Checks the compatibility with another schema

        Args:
          schema: Other ArraySchema
          name_self: How to call self in the error messages
          name_schema: analogously to name_self for the schema ArraySchema
          verbose: bool, describe what went wrong through print()
        """
        def print_msg(msg):
            if verbose:
                # print("ArraySchema mismatch")
                print(msg)

        if not isinstance(schema, ArraySchema):
            print_msg("Expecting ArraySchema. Got type({0} schema) = {1}".format(name_schema,
                                                                                 type(schema)))
            return False

        def print_msg_template():
            print("ArraySchema mismatch")
            print("Array shapes don't match for the fields:")
            print("--")
            print(name_self)
            print("--")
            print(self.get_config_as_yaml())
            print("--")
            print(name_schema)
            print("--")
            print(schema.get_config_as_yaml())
            print("--")
            print("Provided shape (without the batch axis): {0}, expected shape: {1} ".format(bshape, self.shape))

        bshape = schema.shape
        if not len(bshape) == len(self.shape):
            print_msg_template()
            return False
        for i in range(len(bshape)):
            if bshape[i] is not None and self.shape[i] is not None:
                # shapes don't match
                if not bshape[i] == self.shape[i]:
                    print_msg_template()
                    return False
        return True


# --------------------------------------------
# Model specific components

@related.immutable(strict=True)
class ModelSchema(RelatedConfigMixin):
    """Describes the model schema
    """
    # can be a dictionary, list or a single array
    inputs = NestedMappingField(ArraySchema, keyword="shape", key="name")
    targets = NestedMappingField(ArraySchema, keyword="shape", key="name")

    def compatible_with_schema(self, dataloader_schema, verbose=True):
        """Check the compatibility: model.schema <-> dataloader.output_schema

        Checks preformed:
        - nested structure is the same (i.e. dictionary names, list length etc)
        - array shapes are compatible
        - returned obj classess are compatible

        # Arguments
            dataloader_schema: a dataloader_schema of data returned by one iteraton of dataloader's dataloader_schema_iter
                nested dictionary
            verbose: verbose error logging if things don't match

        # Returns
           bool: True only if everyhing is ok
        """
        def print_msg(msg):
            if verbose:
                print(msg)

        # Inputs check
        def compatible_nestedmapping(dschema, descr, cls, verbose=True):
            """Recursive function of checks

            shapes match, dschema-dim matches
            """
            if isinstance(descr, cls):
                # Recursion stop
                return descr.compatible_with_schema(dschema,
                                                    name_self="Model",
                                                    name_schema="Dataloader",
                                                    verbose=verbose)
            elif isinstance(dschema, collections.Mapping) and isinstance(descr, collections.Mapping):
                if not set(descr.keys()).issubset(set(dschema.keys())):
                    print_msg("Dataloader doesn't provide all the fields required by the model:")
                    print_msg("dataloader fields: {0}".format(dschema.keys()))
                    print_msg("model fields: {0}".format(descr.keys()))
                    return False
                return all([compatible_nestedmapping(dschema[key], descr[key], cls, verbose) for key in descr])
            elif isinstance(dschema, collections.Sequence) and isinstance(descr, collections.Sequence):
                if not len(descr) <= len(dschema):
                    print_msg("Dataloader doesn't provide all the fields required by the model:")
                    print_msg("len(dataloader): {0}".format(len(dschema)))
                    print_msg("len(model): {0}".format(len(descr)))
                    return False
                return all([compatible_nestedmapping(dschema[i], descr[i], cls, verbose) for i in range(len(descr))])

            print_msg("Invalid types:")
            print_msg("type(Dataloader schema): {0}".format(type(dschema)))
            print_msg("type(Model schema): {0}".format(type(descr)))
            return False

        if not compatible_nestedmapping(dataloader_schema.inputs, self.inputs, ArraySchema, verbose):
            return False

        if (isinstance(dataloader_schema.targets, ArraySchema) or
            len(dataloader_schema.targets) > 0) and not compatible_nestedmapping(dataloader_schema.targets,
                                                                                 self.targets,
                                                                                 ArraySchema,
                                                                                 verbose):
            return False

        return True


# --------------------------------------------
# DataLoader specific components

@enum.unique
class MetadataType(enum.Enum):
    # TODO - make capital
    GENOMIC_RANGES = "GenomicRanges"
    STR = "str"
    INT = "int"
    FLOAT = "float"
    ARRAY = "array"
    # TODO - add bed3 or bed6 ranges


@related.immutable(strict=True)
class MetadataStruct(RelatedConfigMixin):

    doc = related.StringField()
    type = related.ChildField(MetadataType, required=False)
    name = related.StringField(required=False)

    def compatible_with_batch(self, batch, verbose=True):
        """Checks compatibility with a particular numpy array

        Args:
          batch: numpy array of a batch

          verbose: print the fail reason
        """

        def print_msg(msg):
            if verbose:
                print("MetadataStruct mismatch")
                print(msg)

        # custom classess
        if self.type == MetadataType.GENOMIC_RANGES:
            if not isinstance(batch, GenomicRanges):
                # TODO - do we strictly require the GenomicRanges class?
                #          - relates to metadata.py TODO about numpy_collate
                #        for now we should just be able to convert to the GenomicRanges class
                #        without any errors
                try:
                    GenomicRanges.from_dict(batch)
                except Exception as e:
                    print_msg("expecting a GenomicRanges object or a GenomicRanges-like dict")
                    print_msg("convertion error: {0}".format(e))
                    return False
                else:
                    return True
            else:
                return True

        # type = np.ndarray
        if not isinstance(batch, np.ndarray):
            print_msg("Expecting a np.ndarray. Got type(batch) = {0}".format(type(batch)))
            return False

        if not batch.ndim >= 1:
            print_msg("The array is a scalar (expecting at least the batch dimension)")
            return False

        bshape = batch.shape[1:]

        # scalars
        if self.type in {MetadataType.INT, MetadataType.STR, MetadataType.FLOAT}:
            if bshape != () and bshape != (1,):
                print_msg("expecting a scalar, got an array with shape (without the batch axis): {0}".format(bshape))
                return False

        # arrays
        # - no checks

        return True


@related.immutable(strict=True)
class DataLoaderSchema(RelatedConfigMixin):
    """Describes the model schema

    Properties:
     - we allow classes that contain also dictionaries
       -> leaf can be an
         - array
         - scalar
         - custom dictionary (recursive data-type)
         - SpecialType (say ArrayRanges, BatchArrayRanges, which will
                        effectively be a dicitonary of scalars)
    """
    inputs = NestedMappingField(ArraySchema, keyword="shape", key="name")
    targets = NestedMappingField(ArraySchema, keyword="shape", key="name", required=False)
    metadata = NestedMappingField(MetadataStruct, keyword="doc", key="name",
                                  required=False)

    def compatible_with_batch(self, batch, verbose=True):
        """Validate if the batch of data complies with the schema

        Checks preformed:
        - nested structure is the same (i.e. dictionary names, list length etc)
        - array shapes are compatible
        - returned obj classess are compatible

        # Arguments
            batch: a batch of data returned by one iteraton of dataloader's batch_iter
                nested dictionary
            verbose: verbose error logging if things don't match

        # Returns
           bool: True only if everyhing is ok
        """
        def print_msg(msg):
            if verbose:
                print(msg)

        # check the individual names
        if not isinstance(batch, dict):
            print("not isinstance(batch, dict)")
            return False

        # contains only the three specified fields
        if not set(batch.keys()).issubset({"inputs", "targets", "metadata"}):
            print('not set(batch.keys()).issubset({"inputs", "targets", "metadata"})')
            return False

        # Inputs check
        def compatible_nestedmapping(batch, descr, cls, verbose=True):
            """Recursive function of checks

            shapes match, batch-dim matches
            """
            # we expect a numpy array/special class, a list or a dictionary

            # Special case for the metadat
            if isinstance(descr, cls):
                return descr.compatible_with_batch(batch, verbose=verbose)
            elif isinstance(batch, collections.Mapping) and isinstance(descr, collections.Mapping):
                if not set(batch.keys()) == set(descr.keys()):
                    print_msg("The dictionary keys don't match:")
                    print_msg("batch: {0}".format(batch.keys()))
                    print_msg("descr: {0}".format(descr.keys()))
                    return False
                return all([compatible_nestedmapping(batch[key], descr[key], cls, verbose) for key in batch])
            elif isinstance(batch, collections.Sequence) and isinstance(descr, collections.Sequence):
                if not len(batch) == len(descr):
                    print_msg("Lengths dont match:")
                    print_msg("len(batch): {0}".format(len(batch)))
                    print_msg("len(descr): {0}".format(len(descr)))
                    return False
                return all([compatible_nestedmapping(batch[i], descr[i], cls, verbose) for i in range(len(batch))])

            print_msg("Invalid types:")
            print_msg("type(batch): {0}".format(type(batch)))
            print_msg("type(descr): {0}".format(type(descr)))
            return False

        # inputs needs to be present allways
        if "inputs" not in batch:
            print_msg('not "inputs" in batch')
            return False

        if not compatible_nestedmapping(batch["inputs"], self.inputs, ArraySchema, verbose):
            return False

        if "targets" in batch and not \
                (len(batch["targets"]) == 0):  # unspecified
            if self.targets is None:
                # targets need to be specified if we want to use them
                print_msg('self.targets is None')
                return False
            if not compatible_nestedmapping(batch["targets"], self.targets, ArraySchema, verbose):
                return False

        # metadata needs to be present if it is defined in the description
        if self.metadata is not None:
            if "metadata" not in batch:
                print_msg('not "metadata" in batch')
                return False
            if not compatible_nestedmapping(batch["metadata"], self.metadata, MetadataStruct, verbose):
                return False
        else:
            if "metadata" in batch:
                print_msg('"metadata" in batch')
                return False

        return True


@related.immutable(strict=True)
class PostProcDataLoaderStruct(RelatedConfigMixin):
    variant_effects = related.ChildField(postprocessing.components.VarEffectDataLoaderArgs, required=False)


@related.immutable(strict=True)
class PostProcModelStruct(RelatedConfigMixin):
    variant_effects = related.ChildField(postprocessing.components.VarEffectModelArgs, required=False)


@related.immutable(strict=True)
class DataLoaderArgument(RelatedConfigMixin):
    # MAYBE - make this a general argument class
    doc = related.StringField("", required=False)
    example = AnyField(required=False)
    name = related.StringField(required=False)
    type = related.StringField(default='str', required=False)
    optional = related.BooleanField(default=False, required=False)
    tags = StrSequenceField(str, default=[], required=False)  # TODO - restrict the tags

    def __attrs_post_init__(self):
        if self.doc == "":
            logger.warn("doc empty for one of the dataloader `args` fields")


@related.immutable(strict=True)
class Dependencies(RelatedConfigMixin):
    conda = StrSequenceField(str, default=[], required=False, repr=True)
    pip = StrSequenceField(str, default=[], required=False, repr=True)
    # not really required
    conda_channels = related.SequenceField(str, default=["defaults"],
                                           required=False, repr=True)

    def __attrs_post_init__(self):
        """
        In case conda or pip are filenames pointing to existing files,
        read the files and populate the package names
        """
        if len(self.conda) == 1 and self.conda[0].endswith(".txt") and \
           os.path.exists(self.conda[0]):
            # found a conda txt file
            object.__setattr__(self, "conda", read_txt(self.conda[0]))

        if len(self.pip) == 1 and self.pip[0].endswith(".txt") and \
           os.path.exists(self.pip[0]):
            # found a pip txt file
            object.__setattr__(self, "pip", read_txt(self.pip[0]))

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
            channels, packages = self._get_channels_packages()
            kconda.install_conda(packages, channels)

    def install(self, dry_run=False):
        self.install_conda(dry_run)
        self.install_pip(dry_run)

    def merge(self, dependencies):
        """Merge one dependencies with another one

        Use case: merging the dependencies of model and dataloader

        Args:
          dependencies: Dependencies instance

        Returns:
          new Dependencies instance
        """
        return Dependencies(
            conda=unique_list(list(self.conda) + list(dependencies.conda)),
            pip=kconda.normalize_pip(list(self.pip) + list(dependencies.pip)),
            conda_channels=unique_list(list(self.conda_channels) + list(dependencies.conda_channels))
        )

    def _get_channels_packages(self):
        """Get conda channels and packages separated from each other (by '::')
        """
        if len(self.conda) == 0:
            return self.conda_channels, self.conda
        channels, packages = list(zip(*map(kconda.parse_conda_package, self.conda)))
        channels = unique_list(list(channels) + list(self.conda_channels))
        packages = unique_list(list(packages))
        return channels, packages

    def to_env_dict(self, env_name):
        channels, packages = self._get_channels_packages()
        if isinstance(packages, related.types.TypedSequence):
            packages = packages.list
        if isinstance(channels, related.types.TypedSequence):
            channels = channels.list

        env_dict = OrderedDict(
            name=env_name,
            channels=channels,
            dependencies=packages + [OrderedDict(pip=kconda.normalize_pip(self.pip))]
        )
        return env_dict

    def to_env_file(self, env_name, path):
        """Dump the dependencies to a file
        """
        with open(path, 'w') as f:
            d = self.to_env_dict(env_name)

            # add python if not present
            add_py = True
            for dep in d['dependencies']:
                if isinstance(dep, str) and dep.startswith("python"):
                    add_py = False

            if add_py:
                d['dependencies'] = ["python"] + d['dependencies']
            # -----
            # remove fields that are empty
            out = []
            for k in d:
                if not (isinstance(d[k], list) and len(d[k]) == 0):
                    out.append((k, d[k]))
            # -----

            f.write(yaml_ordered_dump(OrderedDict(out),
                                      indent=2,
                                      default_flow_style=False))

    # @classmethod
    # def from_file(cls, path):
    #     """TODO instantiate Dependencies from a yaml file
    #     """
    #     pass


# --------------------------------------------
# Final description classes modelling the yaml files
@related.immutable(strict=True)
class ModelDescription(RelatedLoadSaveMixin):
    """Class representation of model.yaml
    """
    type = related.StringField()
    args = related.ChildField(dict)
    info = related.ChildField(ModelInfo)
    schema = related.ChildField(ModelSchema)
    default_dataloader = related.StringField(default='.')
    postprocessing = related.ChildField(PostProcModelStruct, default=PostProcModelStruct(), required=False)
    dependencies = related.ChildField(Dependencies,
                                      default=Dependencies(),
                                      required=False)
    path = related.StringField(required=False)
    # TODO - add after loading validation for the arguments class?


def example_kwargs(dl_args):
    """Return the example kwargs
    """
    return {k: v.example for k, v in six.iteritems(dl_args) if not isinstance(v.example, UNSPECIFIED)}


def default_kwargs(args):
    """Return the example kwargs
    """
    return {k: v.default for k, v in six.iteritems(args) if v.default is not None}


@related.immutable(strict=True)
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
    postprocessing = related.ChildField(PostProcDataLoaderStruct,
                                        default=PostProcDataLoaderStruct(),
                                        required=False)

    def get_example_kwargs(self):
        return example_kwargs(self.args)


# ---------------------
# Global source config

# TODO - write a unit-test for these three
@related.immutable
class TestModelConfig(RelatedConfigMixin):
    batch_size = related.IntegerField(default=None, required=False)


@related.immutable
class TestConfig(RelatedConfigMixin):
    """Models config.yaml in the model root
    """
    constraints = related.MappingField(TestModelConfig, "name", required=False,
                                       repr=True)


@related.immutable
class SourceConfig(RelatedLoadSaveMixin):
    test = related.ChildField(TestConfig, required=False)

# TODO - special metadata classes should just extend the dictionary field
# (to be fully compatible with batching etc)
