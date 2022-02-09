from dataclasses import dataclass, field
import enum
from typing import Any, Dict, Tuple

from kipoi.metadata import GenomicRanges
from kipoi.kipoimodeldescription import KipoiModelInfo
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
    type: MetadataType = MetadataType.GENOMIC_RANGES
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
        # self.targets = ?
        self.metadata =  MetadataStruct(**self.metadata)
        # keyword="doc", key="name",
    # inputs = NestedMappingField(ArraySchema, keyword="shape", key="name")
    # targets = NestedMappingField(ArraySchema, keyword="shape", key="name", required=False)
    # metadata = NestedMappingField(MetadataStruct, keyword="doc", key="name",
    #                               required=False)

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
            # TODO
            pass




@dataclass
class KipoiDataLoaderDescription:
    """Class representation of dataloader.yaml
    """
    defined_as: str 
    args: KipoiDataLoaderArgument
    output_schema: KipoiDataLoaderSchema
    type: str = "" 
    info: KipoiModelInfo = KipoiModelInfo()
    dependencies: Dependencies = Dependencies()
    path: str = ''
    writers: Dict = field(default_factory=dict)

    def get_example_kwargs(self):
        # TODO
        pass

    def download_example(self, output_dir, absolute_path=False, dry_run=False):
        # TODO
        pass