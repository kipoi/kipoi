from dataclasses import dataclass, field
import enum
from typing import Any, Dict, Tuple

from kipoi.kipoimodeldescription import Dependencies, KipoiModelInfo

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