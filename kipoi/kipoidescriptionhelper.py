from collections.abc import Mapping, Sequence
from collections import OrderedDict
from dataclasses import dataclass, field
import os
import logging
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import enum
import numpy as np
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
class KipoiArraySchema:
    """
    Args:
      shape: Tuple of shape (same as in Keras for the input)
      doc: Description of the array
      special_type: str, special type name. Could also be an array of special entries?
      metadata_entries: str or list of metadata
    """
    shape: Tuple[int] = () # TODO: Should it be a default field??
    verbose: bool = True
    doc: str = ""
    name: str = ""
    special_type: str = ArraySpecialType.DNASeq
    associated_metadata: Tuple[str] = ()
    column_labels: List[str] = field(default_factory=list)

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
        if not self.column_labels:
            self.column_labels = None
        
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
    authors: Tuple[Author] = ()
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