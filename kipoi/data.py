from __future__ import absolute_import
from __future__ import print_function

import os
import logging
import abc

import kipoi  # for .config module
from kipoi.components import DataLoaderDescription
from .utils import load_module, cd, getargs
from kipoi.torch_data import DataLoader

# TODO - moved to a different directory
from kipoi.data_helper import numpy_collate, numpy_collate_concat


# TODO - put to remote


_logger = logging.getLogger('kipoi')

#
PREPROC_IFILE_TYPES = ['DNA_regions']
PREPROC_IFILE_FORMATS = ['bed3']


# inspired by PyTorch
# http://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset

# TODO - OR maybe call it Dataloader???

# Dataloader - better name than a pre-processor?
# Or call it the Dataset - convention by pytorch and tensorflow
# BatchDataloader

# TODO - what are other properties of an abstract datalaoder?
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

    # TODO - implement batch_iter(self, batch_size=32):
    # TODO - implement load_all(self):


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


def DataLoader_factory(dataloader, source="kipoi"):
    if source == "dir":
            # TODO - maybe add it already to the config - to prevent code copying
        source = kipoi.remote.LocalModelSource(".")
    else:
        source = kipoi.config.get_source(source)

    # pull the dataloader & get the dataloader directory
    dataloader_dir = source.pull_model(dataloader)

    yaml_path = kipoi.remote.get_dataloader_file(dataloader_dir)

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
    CustomDataLoader.schema = dl.schema

    # keep it hidden?
    CustomDataLoader._yaml_path = yaml_path
    CustomDataLoader.source = source
    # TODO - rename?
    CustomDataLoader.source_dir = dataloader_dir
    return CustomDataLoader


# TODO - implement other stuff
AVAILABLE_DATALOADERS = {"Dataset": Dataset}
