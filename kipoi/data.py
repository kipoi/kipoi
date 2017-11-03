from __future__ import absolute_import
from __future__ import print_function

import os
import logging
import abc

import kipoi  # for .config module
from kipoi.components import DataLoaderDescription
from .utils import load_module, cd, getargs
from .external.torch.data import DataLoader
from kipoi.data_utils import numpy_collate, numpy_collate_concat
from tqdm import tqdm
# TODO - put to remote


_logger = logging.getLogger('kipoi')

#
PREPROC_IFILE_TYPES = ['DNA_regions']
PREPROC_IFILE_FORMATS = ['bed3']


# TODO - what are other properties of an abstract datalaoder?
# inspired by PyTorch
# http://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset
class BaseDataLoader(object):
    __metaclass__ = abc.ABCMeta

    # TODO - define proper abstract classess
    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abc.abstractmethod
    def batch_iter(self, batch_size=32):
        raise NotImplementedError

    @abc.abstractmethod
    def load_all(self):
        raise NotImplementedError

# --------------------------------------------
# Different implementations


class Dataset(BaseDataLoader):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    def batch_iter(self, batch_size=32, shuffle=False, num_workers=0, drop_last=False):
        """Return a batch-iterator

        Arguments:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: False).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main process
                (default: 0)
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If False and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: False)

        Returns:
            iterator
        """
        dl = DataLoader(self, batch_size=batch_size,
                        collate_fn=numpy_collate,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        drop_last=drop_last)
        return iter(dl)

    def load_all(self, batch_size=32, num_workers=0):
        """Load the whole dataset into memory
        Arguments:
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main process
                (default: 0)
        """
        return numpy_collate_concat([x for x in tqdm(self.batch_iter(batch_size,
                                                                     num_workers=num_workers))])


# TODO - implement some of the synthactic shugar - i.e. get_avail_arguments etc...?

# TODO - for efficiency also allow for an iterator?

# Index is typically based on the intervals_file...


# ## Why specifying how to get a single example instead of a batch?
#
# - batch_size shouldn't be a parameter of the pre-processor
#   - it only controls how much memory will your inference require
#   - better specified independently from the command-line
# - Easier code
#   - batching code is boilerplate.
# - We can easily re-use it for training later
#   - by using the `torch.utils.data.DataLoader(batch_size, shuffle, sampler, num_workers...)`
# - Loss of efficiency?
#   - maybe true as we can't make use of vectorized operations.
#   However, model's forward pass might take longer than reading a samples
#   from disk using multiple-workers

# --------------------------------------------

def get_dataloader_factory(dataloader, source="kipoi"):
    if source == "dir":
            # TODO - maybe add it already to the config - to prevent code copying
        source = kipoi.remote.LocalSource(".")
    else:
        source = kipoi.config.get_source(source)

    # pull the dataloader & get the dataloader directory
    yaml_path = source.pull_dataloader(dataloader)
    dataloader_dir = os.path.dirname(yaml_path)

    # Setup dataloader description
    dl = DataLoaderDescription.load(yaml_path)
    # --------------------------------------------
    # input the
    file_path, obj_name = tuple(dl.defined_as.split("::"))
    CustomDataLoader = getattr(load_module(os.path.join(dataloader_dir, file_path)),
                               obj_name)

    # check that dl.type is correct
    if dl.type not in AVAILABLE_DATALOADERS:
        raise ValueError("dataloader type: {0} is not in supported dataloaders:{1}".
                         format(dl.type, list(AVAILABLE_DATALOADERS.keys())))
    # check that CustomDataLoader indeed interits from the right DataLoader
    if not issubclass(CustomDataLoader, AVAILABLE_DATALOADERS[dl.type]):
        raise ValueError("DataLoader does't inherit from the specified dataloader: {0}".
                         format(AVAILABLE_DATALOADERS[dl.type].__name__))
    # check that the extractor arguments match yml arguments
    if not getargs(CustomDataLoader) == set(dl.args.keys()):
        raise ValueError("DataLoader arguments: \n{0}\n don't match " +
                         "the specification in the dataloader.yaml file:\n{1}".
                         format(set(getargs(CustomDataLoader)), set(dl.args.keys())))

    # Inherit the attributes from dl
    # TODO - make this more automatic / DRY
    # write a method to load those things?
    CustomDataLoader.type = dl.type
    CustomDataLoader.defined_as = dl.defined_as
    CustomDataLoader.args = dl.args
    CustomDataLoader.info = dl.info
    CustomDataLoader.output_schema = dl.output_schema
    CustomDataLoader.dependencies = dl.dependencies
    CustomDataLoader.postprocessing = dl.postprocessing
    # keep it hidden?
    CustomDataLoader._yaml_path = yaml_path
    CustomDataLoader.source = source
    # TODO - rename?
    CustomDataLoader.source_dir = dataloader_dir
    return CustomDataLoader


# TODO - implement other stuff
AVAILABLE_DATALOADERS = {"Dataset": Dataset}
