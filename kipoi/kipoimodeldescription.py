from collections import OrderedDict
from dataclasses import dataclass
import os
import logging
from typing import Any, Dict, List, TYPE_CHECKING

import kipoi_conda as kconda
from kipoi_utils.utils import inherits_from, load_obj, override_default_kwargs, unique_list
from kipoi_utils.external.torchvision.dataset_utils import download_url, check_integrity


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class Dependencies:
    conda: List = []
    pip: List = []
    conda_channels: List = ["defaults"]
    conda_file: ''
    # TODO:  __attrs_post_init__ is not necessary right? Populating conda and pip 
    # dependencies from file will not work and unncessary in this case
    # TODO: Is all_installed necessary?

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
class KipoiDataLoaderImport:
    """Dataloader specification for the import
    """
    defined_as: str
    default_args: dict = {}
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
    expect: Any = None
    precision_decimal: int = 7


@dataclass
class KipoiModelDescription:
    args: Dict
    schema: Dict # Model schema class perhaps?
    defined_as: str 
    model_type: str = ""
    default_dataloader: str = '.'
    dependencies: Dependencies = Dependencies()
    model_test: Any = KipoiModelTest() 
    writers: Dict = OrderedDict()

    def __post_init__(self) -> None:
        if not self.defined_as and not self.model_type:
            raise ValueError("Either defined_as or type need to be specified")

        # parse default_dataloader
        if isinstance(self.default_dataloader, dict):
            self.default_dataloader = KipoiDataLoaderImport(**self.default_dataloader)