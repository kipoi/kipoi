from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import enum
from typing import Any, Dict, Tuple
import numpy as np

from kipoi.metadata import GenomicRanges
from kipoi.kipoimodeldescription import KipoiModelInfo, KipoiRemoteFile
from kipoi.kipoidescriptionhelper import Dependencies, KipoiArraySchema


@dataclass
class KipoiDataLoaderArgument:
    doc: str = ""
    example: Dict = field(default_factory=dict)
    default: Dict = field(default_factory=dict)
    name: str = ""
    type: str = 'str'
    optional: bool = False 
    tags: Tuple[str] = ()

    def __post_init__(self):
        self.tags = list(self.tags)
        if self.doc == "":
            logger.warning("doc empty for one of the dataloader `args` fields")
        if self.example:
            self.example = KipoiRemoteFile(**self.example)

@enum.unique
class ArraySpecialType(enum.Enum):
    DNASeq = "DNASeq"
    DNAStringSeq = "DNAStringSeq"
    BIGWIG = "bigwig"
    VPLOT = "v-plot"
    Array = "Array"

@enum.unique
class MetadataType(enum.Enum):
    GENOMIC_RANGES = "GenomicRanges"
    STR = "str"
    INT = "int"
    FLOAT = "float"
    ARRAY = "array"

@dataclass
class MetadataStruct:
    doc: str = "" 
    type: str = MetadataType.GENOMIC_RANGES.value
    name: str = ""

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
        if self.type == MetadataType.GENOMIC_RANGES.value:
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
        if self.type in {MetadataType.INT.value, MetadataType.STR.value, MetadataType.FLOAT.value}:
            if bshape != () and bshape != (1,):
                print_msg("expecting a scalar, got an array with shape (without the batch axis): {0}".format(bshape))
                return False
        return True


@dataclass
class KipoiDataLoaderSchema:
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
    inputs: Dict
    targets: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.inputs = KipoiArraySchema(**self.inputs)
        if self.targets:
            self.targets = KipoiArraySchema(**self.targets)
        if self.metadata:
            self.metadata = OrderedDict(self.metadata)
            for key, value in self.metadata.items():
                self.metadata[key] = MetadataStruct(**value)
            
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
            elif isinstance(batch, Mapping) and isinstance(descr, Mapping):
                if not set(batch.keys()) == set(descr.keys()):
                    print_msg("The dictionary keys don't match:")
                    print_msg("batch: {0}".format(batch.keys()))
                    print_msg("descr: {0}".format(descr.keys()))
                    return False
                return all([compatible_nestedmapping(batch[key], descr[key], cls, verbose) for key in batch])
            elif isinstance(batch, Sequence) and isinstance(descr, Sequence):
                if not len(batch) == len(descr):
                    print_msg("Lengths dont match:")
                    print_msg("len(batch): {0}".format(len(batch)))
                    print_msg("len(descr): {0}".format(len(descr)))
                    return False
                return all([compatible_nestedmapping(batch[i], descr[i], cls, verbose) for i in range(len(batch))])

            print_msg("Invalid types dataloader:")
            print_msg("type(batch): {0}".format(type(batch)))
            print_msg("type(descr): {0}".format(type(descr)))
            return False

        # inputs needs to be present allways
        if "inputs" not in batch:
            print_msg('not "inputs" in batch')
            return False

        if not compatible_nestedmapping(batch["inputs"], self.inputs, KipoiArraySchema, verbose):
            return False

        if "targets" in batch and not \
                (len(batch["targets"]) == 0):  # unspecified
            if self.targets is None:
                # targets need to be specified if we want to use them
                print_msg('self.targets is None')
                return False
            if not compatible_nestedmapping(batch["targets"], self.targets, KipoiArraySchema, verbose):
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


@dataclass
class KipoiDataLoaderDescription:
    """Class representation of dataloader.yaml
    """
    defined_as: str 
    args: Dict
    output_schema: KipoiDataLoaderSchema
    type: str = "" 
    info: KipoiModelInfo = KipoiModelInfo()
    dependencies: Dependencies = Dependencies()
    path: str = ''
    writers: Dict = field(default_factory=dict)

    def __post_init__(self):
        for key, value in self.args.items():
            if 'name' not in value:
                value['name'] = key
            self.args[key] = KipoiDataLoaderArgument(**value)
        self.args = OrderedDict(self.args)

    def get_example_kwargs(self):
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

    print_args = print_kwargs

