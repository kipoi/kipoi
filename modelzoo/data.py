import os
import logging
import yaml
import inspect

from modelzoo.utils import load_module

from torch.utils.data import DataLoader

import numpy as np
import sys
import collections
# string_classes
if sys.version_info[0] == 2:
    string_classes = basestring
else:
    string_classes = (str, bytes)

_logger = logging.getLogger('model-zoo')

# PREPROC_FIELDS = ['function_name', 'type', 'arguments']
PREPROC_FIELDS = ['type', 'defined_as', 'arguments']
PREPROC_TYPES = ['Dataset']  # ['generator', 'return', 'Dataset'] - TODO support also other extractor types
PREPROC_IFILE_TYPES = ['DNA_regions']
PREPROC_IFILE_FORMATS = ['bed3']


def numpy_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    if type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return np.stack(batch, 0)
        if elem.shape == ():  # scalars
            return np.array(batch)
    elif isinstance(batch[0], int):
        return np.asarray(batch)
    elif isinstance(batch[0], float):
        return np.asarray(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


# inspired by PyTorch
# http://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset

# TODO - OR maybe call it Dataloader???

# Dataloader - better name than a pre-processor?
# Or call it the Dataset - convention by pytorch and tensorflow
# BatchDataloader


class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


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

# TODO - how to name it?
# Extractor, Dataset, Preprocessor, Dataloader?

# main functionality - factory class
def load_extractor(preproc_dir):
    """Load the extractor from disk

    1. Parse the yaml file
    2. Import the extractor
    3. Validate the pre-processor <-> yaml
      - check if the arguments match
    4. Append yaml description to __doc__
    5. Return the pre-processor
    """

    # Parse the model.yaml
    with open(os.path.join(preproc_dir, 'extractor.yaml')) as ifh:
        unparsed_yaml = ifh.read()
    description_yaml = yaml.load(unparsed_yaml)
    preproc_spec = description_yaml['extractor']
    validate_extractor_spec(preproc_spec)

    extractor = getattr(load_module(os.path.join(preproc_dir, 'extractor.py')),
                        preproc_spec['defined_as'])

    # TODO - check that the extractor arguments match yml arguments
    _logger.info('successfully imported {} from extractor.py'.
                 format(preproc_spec['defined_as']))

    extractor.__doc__ = """Model instance

    # Methods
        - .__getitem__(idx) - Get items via subsetting obj.[idx]
        - .__len__() - Get the length - len(obj)

    # extractor.yaml

        {0}
    """.format((' ' * 8).join(unparsed_yaml.splitlines(True)))

    return extractor


def validate_extractor_spec(preproc_spec):
    # check extractor fields
    assert (all(field in preproc_spec for field in PREPROC_FIELDS))

    # check preproc type
    assert preproc_spec['type'] in PREPROC_TYPES
