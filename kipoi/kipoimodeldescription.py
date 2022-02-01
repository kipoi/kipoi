from collections import OrderedDict
from dataclasses import dataclass
import os
import logging
from typing import Any, Dict

from kipoi_utils.utils import inherits_from, load_obj, override_default_kwargs
from kipoi_utils.external.torchvision.dataset_utils import download_url, check_integrity


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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
    dependencies Any # Dependencies class, a default value need to be added
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
    dependencies: Any # Dependencies class, a default value need to be added
    model_test: Any = KipoiModelTest() 
    writers: Dict = OrderedDict()

    def __post_init__(self) -> None:
        if not self.defined_as and not self.model_type:
            raise ValueError("Either defined_as or type need to be specified")

        # parse default_dataloader
        if isinstance(self.default_dataloader, dict):
            self.default_dataloader = KipoiDataLoaderImport(self.default_dataloader)


