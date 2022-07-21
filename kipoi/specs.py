"""Defines the classes for Yaml parsing using related: https://github.com/genomoncology/related
"""
import collections
import logging
import os
from collections import OrderedDict

import enum
import numpy as np
import related
import re

import kipoi
from kipoi_utils.external.torchvision.dataset_utils import download_url, check_integrity
from kipoi.plugin import get_model_yaml_parser, get_dataloader_yaml_parser, is_installed
import kipoi_conda as kconda
from kipoi_utils.external.related.fields import StrSequenceField, NestedMappingField, TupleIntField, AnyField, \
    UNSPECIFIED
from kipoi_utils.external.related.mixins import RelatedConfigMixin, RelatedLoadSaveMixin
from kipoi.metadata import GenomicRanges
from kipoi_utils.utils import (unique_list, yaml_ordered_dump, read_txt, read_yaml,
                               load_obj, inherits_from, override_default_kwargs, recursive_dict_parse)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --------------------------------------------
# Common specs (model and dataloader)


@related.immutable(strict=False)
class Author(RelatedConfigMixin):
    name = related.StringField()
    github = related.StringField(required=False)
    email = related.StringField(required=False)


@related.mutable(strict=False)
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
    authors = related.SequenceField(Author, repr=True, required=False)
    doc = related.StringField("", required=False)  # free-text description of the model
    name = related.StringField(required=False)  # TODO - deprecate
    version = related.StringField(default="0.1", required=False)
    license = related.StringField(default="MIT", required=False)  # license of the model/dataloader - defaults to MIT
    tags = StrSequenceField(str, default=[], required=False)

    def __attrs_post_init__(self):
        if self.authors and self.doc == "":
            logger.warning("doc empty for the `info:` field")


@related.mutable(strict=False)
class ModelInfo(Info):
    """Additional information for the model - not applicable to the dataloader
    """
    contributors = related.SequenceField(Author, default=[], repr=True, required=False)
    cite_as = related.StringField(required=False)  # a link or a description how to cite the paper (say a doi link)
    trained_on = related.StringField(required=False)  # a link or a description of the training dataset
    training_procedure = related.StringField(
        required=False)  # brief description about the training procedure for the trained_on dataset.


@enum.unique
class ArraySpecialType(enum.Enum):
    DNASeq = "DNASeq"
    DNAStringSeq = "DNAStringSeq"
    BIGWIG = "bigwig"
    VPLOT = "v-plot"
    Array = "Array"


@related.mutable(strict=False)
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
    column_labels = StrSequenceField(str, default=[],
                                     required=False)  # either a list or a path to a file --> need to check whether it's a list

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
        from io import open

        if len(self.column_labels) > 1:
            # check that length is ok with columns
            self._validate_list_column_labels()
        elif len(self.column_labels) == 1:
            label = self.column_labels.list[0]
            import os
            # check if path exists raise exception only test time, but only a warning in prediction time
            if os.path.exists(label):
                with open(label, "r", encoding="utf-8") as ifh:
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
# Model specific specs

@related.mutable(strict=False)
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
            elif isinstance(dschema, collections.abc.Mapping) and isinstance(descr, collections.abc.Mapping):
                if not set(descr.keys()).issubset(set(dschema.keys())):
                    print_msg("Dataloader doesn't provide all the fields required by the model:")
                    print_msg("dataloader fields: {0}".format(dschema.keys()))
                    print_msg("model fields: {0}".format(descr.keys()))
                    return False
                return all([compatible_nestedmapping(dschema[key], descr[key], cls, verbose) for key in descr])
            elif isinstance(dschema, collections.abc.Sequence) and isinstance(descr, collections.abc.Sequence):
                if not len(descr) <= len(dschema):
                    print_msg("Dataloader doesn't provide all the fields required by the model:")
                    print_msg("len(dataloader): {0}".format(len(dschema)))
                    print_msg("len(model): {0}".format(len(descr)))
                    return False
                return all([compatible_nestedmapping(dschema[i], descr[i], cls, verbose) for i in range(len(descr))])
            elif isinstance(dschema, collections.abc.Mapping) and isinstance(descr, collections.abc.Sequence):
                if not len(descr) <= len(dschema):
                    print_msg("Dataloader doesn't provide all the fields required by the model:")
                    print_msg("len(dataloader): {0}".format(len(dschema)))
                    print_msg("len(model): {0}".format(len(descr)))
                    return False
                compatible = []
                for i in range(len(descr)):
                    if descr[i].name in dschema:
                        compatible.append(compatible_nestedmapping(dschema[descr[i].name], descr[i], cls, verbose))
                    else:
                        print_msg("Model array name: {0} not found in dataloader keys: {1}".
                                  format(descr[i].name, list(dschema.keys())))
                        return False
                return all(compatible)

            print_msg("Invalid types:")
            print_msg("type(Dataloader schema): {0}".format(type(dschema)))
            print_msg("type(Model schema): {0}".format(type(descr)))
            return False

        if not compatible_nestedmapping(dataloader_schema.inputs, self.inputs, ArraySchema, verbose):
            return False

        # checking targets
        if dataloader_schema.targets is None:
            return True

        if (isinstance(dataloader_schema.targets, ArraySchema) or
            len(dataloader_schema.targets) > 0) and not compatible_nestedmapping(dataloader_schema.targets,
                                                                                 self.targets,
                                                                                 ArraySchema,
                                                                                 verbose):
            return False

        return True


# --------------------------------------------
# DataLoader specific specs

@enum.unique
class MetadataType(enum.Enum):
    # TODO - make capital
    GENOMIC_RANGES = "GenomicRanges"
    STR = "str"
    INT = "int"
    FLOAT = "float"
    ARRAY = "array"
    # TODO - add bed3 or bed6 ranges


@related.mutable(strict=False)
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


@related.mutable(strict=False)
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
            elif isinstance(batch, collections.abc.Mapping) and isinstance(descr, collections.abc.Mapping):
                if not set(batch.keys()) == set(descr.keys()):
                    print_msg("The dictionary keys don't match:")
                    print_msg("batch: {0}".format(batch.keys()))
                    print_msg("descr: {0}".format(descr.keys()))
                    return False
                return all([compatible_nestedmapping(batch[key], descr[key], cls, verbose) for key in batch])
            elif isinstance(batch, collections.abc.Sequence) and isinstance(descr, collections.abc.Sequence):
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


# --------------------------------------------
@related.mutable(strict=False)
class RemoteFile(RelatedConfigMixin):
    url = related.StringField()
    md5 = related.StringField("", required=False)
    name = related.StringField("", required=False)

    def __attrs_post_init__(self):
        if self.md5 == "":
            logger.warning("md5 not specified for url: {}".format(self.url))
        if os.path.basename(self.name) != self.name:
            logger.warning("'name' does not seem to be a valid file name: {}".format(self.name))
            self.name = os.path.basename(self.name)

    def validate(self, path):
        """Validate if the path complies with the provided md5 hash
        """
        return check_integrity(path, self.md5)

    def get_file(self, path):
        """Download the remote file to cache_dir and return
        the file path to it
        """
        if self.md5:
            file_hash = self.md5
        else:
            file_hash = None
        root, filename = os.path.dirname(path), os.path.basename(path)
        root = os.path.abspath(root)
        download_url(self.url, root, filename, file_hash)
        return os.path.join(root, filename)


@related.mutable(strict=False)
class DataLoaderArgument(RelatedConfigMixin):
    # MAYBE - make this a general argument class
    doc = related.StringField("", required=False)
    example = AnyField(required=False)
    default = AnyField(required=False)
    name = related.StringField(required=False)
    type = related.StringField(default='str', required=False)
    optional = related.BooleanField(default=False, required=False)
    tags = StrSequenceField(str, default=[], required=False)  # TODO - restrict the tags

    def __attrs_post_init__(self):
        if self.doc == "":
            logger.warning("doc empty for one of the dataloader `args` fields")
            # parse args
        self.example = recursive_dict_parse(self.example, 'url', RemoteFile.from_config)
        self.default = recursive_dict_parse(self.default, 'url', RemoteFile.from_config)


@related.mutable(strict=False)
class Dependencies(RelatedConfigMixin):
    conda = StrSequenceField(str, default=[], required=False, repr=True)
    pip = StrSequenceField(str, default=[], required=False, repr=True)
    # not really required
    conda_channels = related.SequenceField(str, default=["defaults"],
                                           required=False, repr=True)
    conda_file = related.StringField(required=False)

    def __attrs_post_init__(self):
        """
        In case conda or pip are filenames pointing to existing files,
        read the files and populate the package names
        """
        if self.conda_file:
            # use the dependencies from the conda file. Override conda, pip and conda_file
            deps = read_yaml(self.conda_file)
            pip_deps = [x['pip'] for x in deps['dependencies']
                        if isinstance(x, dict)][0]
            conda_deps = [x for x in deps['dependencies']
                          if not isinstance(x, dict)]
            object.__setattr__(self, "pip", pip_deps)
            object.__setattr__(self, "conda", conda_deps)
            object.__setattr__(self, "conda_channels", deps['channels'])
        if len(self.conda) == 1 and self.conda[0].endswith(".txt") and \
                os.path.exists(self.conda[0]):
            # found a conda txt file
            object.__setattr__(self, "conda", read_txt(self.conda[0]))

        if len(self.pip) == 1 and self.pip[0].endswith(".txt") and \
                os.path.exists(self.pip[0]):
            # found a pip txt file
            object.__setattr__(self, "pip", read_txt(self.pip[0]))

    def all_installed(self, verbose=False):
        """Validate if all the dependencies are installed as requested

        Args:
          verbose: if True, display warnings if the dependencies are not installed

        Returns:
          (bool): True if all the required package versions are installed
            and False otherwise
        """
        norm = self.normalized()
        for pkg in list(norm.conda) + list(norm.pip):
            if not kconda.is_installed(pkg):
                if verbose:
                    pkg_name, req_version = kconda.version_split(pkg)
                    found_version = kconda.get_package_version(pkg_name)
                    if found_version is None:
                        print("Package '{}' is not installed".
                              format(pkg_name))
                    else:
                        print("Installed package '{}={}' doesn't "
                              "comply with '{}'".
                              format(pkg_name, found_version, pkg))
                return False
        return True

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

    def normalized(self):
        """Normalize the list of dependencies
        """
        channels, packages = self._get_channels_packages()
        if isinstance(packages, related.types.TypedSequence):
            packages = packages.list
        if isinstance(channels, related.types.TypedSequence):
            channels = channels.list

        return Dependencies(
            conda=packages,
            pip=kconda.normalize_pip(list(self.pip)),
            conda_channels=channels)

    def _get_channels_packages(self):
        """Get conda channels and packages separated from each other(by '::')
        """
        if len(self.conda) == 0:
            return self.conda_channels, self.conda
        channels, packages = list(zip(*map(kconda.parse_conda_package, self.conda)))
        channels = unique_list(list(channels) + list(self.conda_channels))
        packages = unique_list(list(packages))

        # Handle channel order
        if "bioconda" in channels and "conda-forge" not in channels:
            # Insert 'conda-forge' right after bioconda if it is not included
            channels.insert(channels.index("bioconda") + 1, "conda-forge")
        
        # Add special handling for pysam. With kipoi<=0.8.2, if pysam is 
        # versioned, prioritizing of bioconda over defaults and conda-forge
        # channel was ignored. Also, I have noticed unless pysam is downloaded
        # from bioconda channel the resulting conda environment sometimes
        # fails to get resolved. Specifically mentioning bioconda::pysam
        # resolves this. However, users may not be aware of this
        # Bioconda now is always added as a channel if pysam is mentioned as a
        # dependency.
        
        pysam_matcher = re.compile("^pysam")
        for pkg in filter(pysam_matcher.match, packages):
            if "bioconda" not in channels:
                channels.remove("defaults")
                channels.append("bioconda")
                channels.append("defaults")
            elif channels.index("defaults") < channels.index("bioconda"):     
                logger.warning("Swapping channel order - putting defaults last. " +
                               "Using pysam bioconda instead of anaconda")
                channels.remove("defaults")
                channels.append("defaults")

        # Add special case for pytorch-cpu and torchvision-cpu. These 
        # packages are not being updated in conda pytorch channel 
        # anymore. There is no longer any need to provide pytorch-cpu 
        # in model( or dataloader).yaml. Recent 
        # versions of pytorch (since 1.3.0) will install necessary libraries 
        # on its own. 
        for torchpkg in ["^pytorch-cpu",  "^torchvision-cpu"]:
            matcher = re.compile(torchpkg)
            for pkg in filter(matcher.match, packages):
                packages.remove(pkg)
                packages.append(pkg.replace("-cpu", ""))
                if "cpuonly" not in packages:
                    packages.append("cpuonly")
        return channels, packages   

    def to_env_dict(self, env_name):
        deps = self.normalized()
        channels, packages = deps._get_channels_packages()
        if isinstance(packages, related.types.TypedSequence):
            packages = packages.list
        if isinstance(channels, related.types.TypedSequence):
            channels = channels.list

        env_dict = OrderedDict(
            name=env_name,
            channels=channels,
            dependencies=packages + [OrderedDict(pip=kconda.normalize_pip(deps.pip))]
        )
        return env_dict

    @classmethod
    def from_env_dict(self, dict):
        cfg = {}
        cfg["conda_channels"] = dict['channels']
        cfg["conda"] = [el for el in dict['dependencies'] if not isinstance(el, OrderedDict)]
        pip = [el for el in dict['dependencies'] if isinstance(el, OrderedDict)]
        if len(pip) == 1:
            cfg["pip"] = pip[0]['pip']
        elif len(pip) > 1:
            raise Exception("Malformatted conda environment yaml!")
        return self.from_config(cfg)

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

    def gpu(self):
        """Get the gpu - version of the dependencies
        """

        def replace_gpu(dep):
            if dep.startswith("tensorflow") and "gpu" not in dep:
                new_dep = dep.replace("tensorflow", "tensorflow-gpu")
                logger.info("use gpu: Replacing the dependency {0} with {1}".format(dep, new_dep))
                return new_dep
            if dep.startswith("pytorch-cpu"):
                new_dep = dep.replace("pytorch-cpu", "pytorch")
                logger.info("use gpu: Replacing the dependency {0} with {1}".format(dep, new_dep))
                return new_dep
            return dep

        deps = self.normalized()
        deps.conda = [dep for dep in deps.conda if dep != "cpuonly"]
        return Dependencies(
            conda=[replace_gpu(dep) for dep in deps.conda],
            pip=[replace_gpu(dep) for dep in deps.pip],
            conda_channels=deps.conda_channels)

    def osx(self):
        """Get the os - x compatible dependencies
        """
        # As of pytorch 1.11 from here https://pytorch.org/get-started/locally/
        # Linux installation: conda install pytorch torchvision torchaudio 
        # cpuonly -c pytorch
        # Mac installation: conda install pytorch torchvision torchaudio 
        # -c pytorch
        from sys import platform
        if platform != 'darwin':
            logger.warning("Calling osx dependency conversion on non-osx platform: {}".
                           format(platform))

        def replace_osx(dep):
            if dep.startswith("pytorch-cpu"):
                new_dep = dep.replace("pytorch-cpu", "pytorch")
                logger.info("osx: Replacing the dependency {0} with {1}".
                            format(dep, new_dep))
                return new_dep
            return dep

        deps = self.normalized()
        deps.conda = [dep for dep in deps.conda if dep != "cpuonly"]

        return Dependencies(
            conda=[replace_osx(dep) for dep in deps.conda],
            pip=[replace_osx(dep) for dep in deps.pip],
            conda_channels=deps.conda_channels)

    # @classmethod
    # def from_file(cls, path):
    #     """TODO instantiate Dependencies from a yaml file
    #     """
    #     pass


@related.mutable(strict=False)
class DataLoaderImport(RelatedConfigMixin):
    """Dataloader specification for the import
    """
    defined_as = related.StringField()
    default_args = related.ChildField(dict, default=OrderedDict(), required=False)
    # specify also dataloader dependencies explicitly
    dependencies = related.ChildField(Dependencies,
                                      default=Dependencies(),
                                      required=False)
    # whether to parse the dependencies from the dataloader when installing it
    parse_dependencies = related.BooleanField(default=True, required=False)

    def get(self):
        """Get the dataloader
        """
        from kipoi.data import BaseDataLoader
        from copy import deepcopy
        obj = load_obj(self.defined_as)

        # check that it inherits from BaseDataLoader
        if not inherits_from(obj, BaseDataLoader):
            raise ValueError("Dataloader: {} doen't inherit from kipoi.data.BaseDataLoader".format(self.defined_as))

        # override the default arguments
        if self.default_args:
            obj = override_default_kwargs(obj, self.default_args)

        # override also the values in the example in case
        # they were previously specified
        for k, v in self.default_args.items():
            
            if not isinstance(obj.args[k].example, UNSPECIFIED):
                obj.args[k].example = v

        return obj


@related.mutable(strict=False)
class ModelTest(RelatedLoadSaveMixin):
    # predictions = related.
    expect = AnyField(default=None, required=False)
    precision_decimal = related.IntegerField(default=7, required=False)

    # Arrays should be almost equal to `precision_decimal` places
    # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.testing.assert_almost_equal.html
    # abs(desired-actual) < 1.5 * 10**(-precision_decimal)

    def __attrs_post_init__(self):
        if self.expect is not None:
            if not isinstance(self.expect, str):
                # it has to be the url
                if not (isinstance(self.expect, dict) and "url" in self.expect):
                    raise ValueError("expect is not a file path, expecting a url field with entries: url and md5")
                self.expect = RemoteFile.from_config(self.expect)


# --------------------------------------------
# Final description classes modelling the yaml files
@related.mutable(strict=False)
class ModelDescription(RelatedLoadSaveMixin):
    """Class representation of model.yaml
    """
    args = related.ChildField(dict)
    info = related.ChildField(ModelInfo)
    schema = related.ChildField(ModelSchema)
    defined_as = related.StringField(required=False)
    type = related.StringField(required=False)
    default_dataloader = AnyField(default='.', required=False)
    dependencies = related.ChildField(Dependencies,
                                      default=Dependencies(),
                                      required=False)
    test = related.ChildField(ModelTest,
                              default=ModelTest(),
                              required=False)
    path = related.StringField(required=False)
    writers = related.ChildField(dict, default=OrderedDict(), required=False)

    # TODO - add after loading validation for the arguments class?

    def __attrs_post_init__(self):
        if self.defined_as is None and self.type is None:
            raise ValueError("Either defined_as or type need to be specified")
        self.args = recursive_dict_parse(self.args, 'url', RemoteFile.from_config)
        # parse default_dataloader
        if isinstance(self.default_dataloader, dict):
            self.default_dataloader = DataLoaderImport.from_config(self.default_dataloader)


def example_kwargs(dl_args, cache_path=None, absolute_path=True, dry_run=False):
    """Return the example kwargs.

    Args:
      dl_args: dictionary of dataloader args
      cache_path: if specified, save the examples to that directory
    """
    example_files = {}
    for k, v in dl_args.items():
        if isinstance(v.example, UNSPECIFIED):
            continue
        if isinstance(v.example, RemoteFile) and cache_path is not None:
            if absolute_path:
                dl_dir = os.path.abspath(cache_path)
            else:
                dl_dir = cache_path
            if not os.path.exists(dl_dir):
                os.makedirs(dl_dir)

            # determine target path of the example file
            if v.example.name != "":
                # use file name as provided in the example
                path = os.path.join(dl_dir, v.example.name)
            else:
                # otherwise just call it like the argument
                path = os.path.join(dl_dir, k)

            example_files[k] = path
            if os.path.exists(path):
                if v.example.validate(path):
                    logger.info("Example file for argument {} already exists".format(k))
                else:
                    logger.info("Example file for argument {} doesn't match the md5 "
                                "hash {}. Re-downloading".format(k, v.example.md5))
                    if not dry_run:
                        v.example.get_file(path)  # TODO
            else:
                if not dry_run:
                    v.example.get_file(path)  # TODO
        else:
            example_files[k] = v.example
    return example_files
    # return {k: v.example for k, v in six.iteritems(dl_args) if not isinstance(v.example, UNSPECIFIED)}


def download_default_args(args, output_dir):
    """Download the default files
    """
    override = {}
    for k in args:
        # arg.default is None
        # TODO: Any need to do this when args[k] is a dict
        if args[k].default is not None:
            if isinstance(args[k].default, UNSPECIFIED):
                continue
            if isinstance(args[k].default, RemoteFile):
                # if it's a remote file, download it
                # and set the default to the file path

                # specify the file name and create the directory
                logger.info("Downloading dataloader default arguments {} from {}".format(k, args[k].default.url))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if args[k].default.md5:
                    fname = args[k].default.md5
                else:
                    fname = "file"

                # download the parameters and override the model
                args[k].default = args[k].default.get_file(os.path.join(output_dir, fname))

            # build up a override dict from .default args
            if isinstance(args[k].default, bool):
                # necessary check for booleans
                override[k] = args[k].default
            elif os.path.exists(args[k].default):
                # for files, make sure we are using absolute paths
                override[k] = os.path.abspath(args[k].default)
            else:
                override[k] = args[k].default
    return override


@related.mutable(strict=False)
class DataLoaderDescription(RelatedLoadSaveMixin):
    """Class representation of dataloader.yaml
    """
    defined_as = related.StringField()
    args = related.MappingField(DataLoaderArgument, "name")
    output_schema = related.ChildField(DataLoaderSchema)
    type = related.StringField(required=False)
    info = related.ChildField(Info, default=Info(), required=False)
    dependencies = related.ChildField(Dependencies, default=Dependencies(), required=False)
    path = related.StringField(required=False)
    writers = related.ChildField(dict, default=OrderedDict(), required=False)

    def get_example_kwargs(self):
        # return self.download_example()
        if self.path is None:
            path = "."
        else:
            path = self.path
        return example_kwargs(self.args, os.path.join(os.path.dirname(path), "downloaded/example_files"))

    def download_example(self, output_dir, absolute_path=False, dry_run=False):
        return example_kwargs(self.args,
                              output_dir,
                              absolute_path=absolute_path,
                              dry_run=dry_run)

    def print_kwargs(self, format_examples_json=False):
        from kipoi_utils.external.related.fields import UNSPECIFIED
        if not hasattr(self, "args"):
            logger.warning("No keyword arguments defined for the given dataloader.")
            return None

        args = self.args
        for k in args:
            print("{0}:".format(k))
            for elm in ["doc", "type", "optional", "example"]:
                if hasattr(args[k], elm) and \
                        (not isinstance(getattr(args[k], elm), UNSPECIFIED)):
                    print("    {0}: {1}".format(elm, getattr(args[k], elm)))

        # example_kwargs = self.get_example_kwargs()
        # if format_examples_json:
        #     import json
        #     example_kwargs = json.dumps(example_kwargs)
        # print("Example keyword arguments are: {0}".format(str(example_kwargs)))

    print_args = print_kwargs



    # download example files
    # def download_example(self):
    #     example_files = {}
    #     for k, v in six.iteritems(self.args):
    #         if isinstance(v.example, RemoteFile):
    #             if self.path is None:
    #                 raise ValueError("Unable to download example files. path attribute not specified")

    #             dl_dir = os.path.join(self.path, "dataloader_files")
    #             if not os.path.exists(dl_dir):
    #                 os.makedirs(dl_dir)
    #             path = os.path.join(dl_dir, k)
    #             example_files[k] = path
    #             if os.path.exists(path):
    #                 if v.example.validate(path):
    #                     logger.info("Example file for argument {} already exists".format(k))
    #                 else:
    #                     logger.info("Example file for argument {} doesn't match the md5 hash {}. Re-downloading".format(k))
    #                     v.example.get_file(path)  # TODO
    #             else:
    #                 v.example.get_file(path)  # TODO
    #         else:
    #             example_files[k] = v
    #     return example_files


# ---------------------
# Global source config


# TODO - write a unit-test for these three
@related.mutable
class TestModelConfig(RelatedConfigMixin):
    batch_size = related.IntegerField(default=None, required=False)


@related.mutable
class TestConfig(RelatedConfigMixin):
    """Models config.yaml in the model root
    """
    constraints = related.MappingField(TestModelConfig, "name", required=False,
                                       repr=True)


@related.mutable
class SourceConfig(RelatedLoadSaveMixin):
    test = related.ChildField(TestConfig, required=False)
    # default dependencies
    dependencies = related.ChildField(Dependencies,
                                      default=Dependencies(),
                                      required=False)
    path = related.StringField(required=False)

# TODO - special metadata classes should just extend the dictionary field
# (to be fully compatible with batching etc)
