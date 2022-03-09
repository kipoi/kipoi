from collections.abc import Mapping, Sequence
from collections import OrderedDict
from dataclasses import dataclass, field
import os
import logging
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import enum

import kipoi_conda as kconda
from kipoi_utils.utils import inherits_from, load_obj, override_default_kwargs, unique_list
from kipoi_utils.external.torchvision.dataset_utils import download_url, check_integrity

from kipoi.kipoidescriptionhelper import Author, Dependencies, Info, KipoiArraySchema, KipoiRemoteFile

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())



def recursive_url_lookup(args):
    if isinstance(args, dict):
        if 'url' in args:
            return KipoiRemoteFile(url=args['url'], name=args.get('name', ''), md5=args.get('md5', ''))
        else:
            return OrderedDict([(k, recursive_url_lookup(v)) for k, v in args.items()])
    elif isinstance(args, list):
        return [recursive_url_lookup(v) for v in args]
    else:
        return args

def recursive_dict_lookup(args, kw='shape'):
    if isinstance(args, dict):
        if kw in args:
            return KipoiArraySchema(**args)
        else:
            return OrderedDict([(k, recursive_dict_lookup(v, kw)) for k, v in args.items()])
    else:
        return args


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
        from kipoi_utils.external.related.fields import UNSPECIFIED

        obj = load_obj(self.defined_as)

        # check that it inherits from BaseDataLoader
        if not inherits_from(obj, BaseDataLoader):
            raise ValueError(f"Dataloader: {self.defined_as} doen't inherit from kipoi.data.BaseDataLoader")

        # override the default arguments
        if self.default_args:
            obj = override_default_kwargs(obj, self.default_args)

        # override also the values in the example in case
        # they were previously specified
        # TODO: How to modify this code with KipoiDataLoaderImport in mind?
        for k, v in self.default_args.items():            
            if not isinstance(obj.args[k].example, UNSPECIFIED):
                obj.args[k].example = v

        return obj



@dataclass
class KipoiModelTest:
    expect: Dict = None
    precision_decimal: int = 7


@dataclass
class KipoiModelSchema:
    """Describes the model schema
    """
    # can be a dictionary, list or a single array
    inputs: Dict 
    targets: Dict

    def __post_init__(self):
        self.inputs = KipoiArraySchema(**self.inputs)
        self.targets = recursive_dict_lookup(self.targets)

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

            print_msg("Invalid types model:")
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
class KipoiModelInfo(Info):
    """Additional information for the model - not applicable to the dataloader
    """
    contributors: Tuple[Author] = ()
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
    defined_as: str = "" 
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