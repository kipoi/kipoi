from collections import Mapping, OrderedDict, Sequence
from dataclasses import dataclass, field
import os
import logging
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import enum

import kipoi_conda as kconda
from kipoi_utils.utils import inherits_from, load_obj, override_default_kwargs, unique_list
from kipoi_utils.external.torchvision.dataset_utils import download_url, check_integrity


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@enum.unique
class ArraySpecialType(enum.Enum):
    DNASeq = "DNASeq"
    DNAStringSeq = "DNAStringSeq"
    BIGWIG = "bigwig"
    VPLOT = "v-plot"
    Array = "Array"

@dataclass
class KipoiRemoteFile:
    url: str
    md5: str = "" 
    name: str = ""

    def __post_init__(self) -> None:
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

def recursive_url_lookup(args):
    if isinstance(args, dict):
        if 'url' in args:
            return KipoiRemoteFile(url=args['url'], name=args.get('name', ''), md5=args.get('md5', ''))
        else:
            return OrderedDict([(k, recursive_url_lookup(v)) for k, v in args.items()])
    elif isinstance(args, list):
        return [recursive_url_lookup(v, 'url') for v in args]
    else:
        return args

@dataclass
class Dependencies:
    conda: Tuple[str] = ()
    pip: Tuple[str] = ()
    conda_channels: Tuple[str] = ()
    # TODO:  __attrs_post_init__ is not necessary right? Populating conda and pip 
    # dependencies from file will not work and unncessary in this case
    # TODO: Is all_installed necessary?

    def __post_init__(self) -> None:
        self.conda = list(self.conda)
        self.pip = list(self.pip)
        self.conda_channels = list(self.conda_channels)
    

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
            channels, packages = self.get_channels_packages()
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
        channels, packages = self.get_channels_packages()

        return Dependencies(
            conda=packages,
            pip=kconda.normalize_pip(list(self.pip)),
            conda_channels=channels)

    def get_channels_packages(self):
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
        if "pysam" in packages and "bioconda" in channels:
            if channels.index("defaults") < channels.index("bioconda"):
                logger.warning("Swapping channel order - putting defaults last. " +
                               "Using pysam bioconda instead of anaconda")
                channels.remove("defaults")
                channels.insert(len(channels), "defaults")
        return channels, packages

    def to_env_dict(self, env_name):
        deps = self.normalized()
        channels, packages = deps.get_channels_packages()

        env_dict = OrderedDict(
            name=env_name,
            channels=channels,
            dependencies=packages + [OrderedDict(pip=kconda.normalize_pip(deps.pip))]
        )
        return env_dict

    

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
        return Dependencies(
            conda=[replace_gpu(dep) for dep in deps.conda],
            pip=[replace_gpu(dep) for dep in deps.pip],
            conda_channels=deps.conda_channels)

    def osx(self):
        """Get the os - x compatible dependencies
        """
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
        return Dependencies(
            conda=[replace_osx(dep) for dep in deps.conda],
            pip=[replace_osx(dep) for dep in deps.pip],
            conda_channels=deps.conda_channels)


@dataclass
class KipoiDataLoaderImport:
    """Dataloader specification for the import
    """
    defined_as: str
    default_args: Dict =  field(default_factory=dict)
    dependencies: Dependencies = Dependencies() # Dependencies class, a default value need to be added
    parse_dependencies: bool = True 

    def get(self):
        """Get the dataloader
        """
        from kipoi.data import BaseDataLoader
        from copy import deepcopy
        obj = load_obj(self.defined_as)

        # check that it inherits from BaseDataLoader
        if not inherits_from(obj, BaseDataLoader):
            raise ValueError(f"Dataloader: {self.defined_as} doen't inherit from kipoi.data.BaseDataLoader")

        # override the default arguments
        if self.default_args:
            obj = override_default_kwargs(obj, self.default_args)

        # override also the values in the example in case
        # they were previously specified
        for k, v in self.default_args.items():
            if 'example' in obj.args[k] and obj.args[k]['example'] != '':
                obj.args[k]['example'] = v

        return obj



@dataclass
class KipoiModelTest:
    expect: Dict = None
    precision_decimal: int = 7

@dataclass
class KipoiArraySchema:
    """
    Args:
      shape: Tuple of shape (same as in Keras for the input)
      doc: Description of the array
      special_type: str, special type name. Could also be an array of special entries?
      metadata_entries: str or list of metadata
    """
    shape: Tuple[int]
    verbose: bool = True
    doc: str = ""
    name: str = ""
    special_type: str = ArraySpecialType.DNASeq
    associated_metadata: tuple[str] = ()
    column_labels: tuple[str] = () 

    def print_msg(self, msg):
        if self.verbose:
            print("KipoiArraySchema mismatch")
            print(msg)

    def _validate_list_column_labels(self):
        dim_ok = len(self.shape) >= 1
        if dim_ok and (self.shape[0] is not None):
            dim_ok &= len(self.column_labels) == self.shape[0]
        if not dim_ok:
            self.print_msg("Column annotation does not match array dimension with shape %s and %d labels (%s ...)"
                           % (str(self.shape), len(self.column_labels), str(self.column_labels)[:30]))

    def __post_init__(self):
        self.associated_metadata = list(self.associated_metadata)
        self.column_labels = list(self.column_labels)

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
                print("KipoiArraySchema mismatch")
                print(msg)

        # type = np.ndarray
        if not isinstance(batch, np.ndarray):
            print_msg("Expecting a np.ndarray. Got type(batch) = {0}".format(type(batch)))
            return False

        if not batch.ndim >= 1:
            print_msg("The array is a scalar (expecting at least the batch dimension)")
            return False

        return self.compatible_with_schema(KipoiArraySchema(shape=batch.shape[1:],
                                                       doc=""))

    def compatible_with_schema(self, schema, name_self="", name_schema="", verbose=True):
        """Checks the compatibility with another schema

        Args:
          schema: Other KipoiArraySchema
          name_self: How to call self in the error messages
          name_schema: analogously to name_self for the schema KipoiArraySchema
          verbose: bool, describe what went wrong through print()
        """

        def print_msg(msg):
            if verbose:
                # print("KipoiArraySchema mismatch")
                print(msg)

        if not isinstance(schema, KipoiArraySchema):
            print_msg("Expecting KipoiArraySchema. Got type({0} schema) = {1}".format(name_schema,
                                                                                 type(schema)))
            return False

        def print_msg_template():
            print("KipoiArraySchema mismatch")
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

@dataclass
class KipoiModelSchema:
    """Describes the model schema
    """
    # can be a dictionary, list or a single array
    inputs: Dict 
    targets: Dict

    def __post_init__(self):
        self.inputs = KipoiArraySchema(**self.inputs)


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
            elif isinstance(dschema, Mapping) and isinstance(descr, Mapping):
                if not set(descr.keys()).issubset(set(dschema.keys())):
                    print_msg("Dataloader doesn't provide all the fields required by the model:")
                    print_msg("dataloader fields: {0}".format(dschema.keys()))
                    print_msg("model fields: {0}".format(descr.keys()))
                    return False
                return all([compatible_nestedmapping(dschema[key], descr[key], cls, verbose) for key in descr])
            elif isinstance(dschema, Sequence) and isinstance(descr, Sequence):
                if not len(descr) <= len(dschema):
                    print_msg("Dataloader doesn't provide all the fields required by the model:")
                    print_msg("len(dataloader): {0}".format(len(dschema)))
                    print_msg("len(model): {0}".format(len(descr)))
                    return False
                return all([compatible_nestedmapping(dschema[i], descr[i], cls, verbose) for i in range(len(descr))])
            elif isinstance(dschema, Mapping) and isinstance(descr, Sequence):
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
        if not compatible_nestedmapping(dataloader_schema.inputs, self.inputs, KipoiArraySchema, verbose):
            return False

        # checking targets
        if dataloader_schema.targets is None:
            return True

        if (isinstance(dataloader_schema.targets, KipoiArraySchema) or
            len(dataloader_schema.targets) > 0) and not compatible_nestedmapping(dataloader_schema.targets,
                                                                                 self.targets,
                                                                                 KipoiArraySchema,
                                                                                 verbose):
            return False

        return True





@dataclass
class Author:
    name: str
    github: str = "" 
    email: str = ""

@dataclass
class Info:
    """Class holding information about the component.

    info:
      authors:
        - name: Ziga Avsec
      doc: RBP binding prediction
      name: rbp_eclip
      version: 0.1
    """
    authors: tuple[Author] = ()
    doc: str = ""
    name: str = ""  # TODO - deprecate
    version : str = "0.1"
    license : str = "MIT"
    tags: Tuple[str] = ()

    def __post_init__(self):
        self.authors = list(self.authors)
        self.tags = list(self.tags)
        if self.authors and self.doc == "":
            logger.warning("doc empty for the `info:` field")


@dataclass
class KipoiModelInfo(Info):
    """Additional information for the model - not applicable to the dataloader
    """
    contributors: tuple[Author] = ()
    cite_as: str = ""
    trained_on: str = ""
    training_procedure: str = ""

    def __post_init__(self) -> None:
        self.contributors = list(self.contributors)

@dataclass
class KipoiModelDescription:
    args: Dict
    schema: KipoiModelSchema 
    info: KipoiModelInfo
    defined_as: str 
    type: str = ""
    default_dataloader: str = '.'
    dependencies: Dependencies = Dependencies()
    test: KipoiModelTest = KipoiModelTest() 
    writers: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.defined_as and not self.type:
            raise ValueError("Either defined_as or type need to be specified")
        if self.writers:
            self.writers = OrderedDict(self.writers)
        
        self.args = recursive_url_lookup(self.args)

        # parse default_dataloader
        if isinstance(self.default_dataloader, dict):
            self.default_dataloader = KipoiDataLoaderImport(**self.default_dataloader)