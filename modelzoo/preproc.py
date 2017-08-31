# inspired by PyTorch
# http://pytorch.org/docs/master/_modules/torch/utils/data/dataset.html#Dataset

# Dataloader - better name than a pre-processor?
# BatchDataloader


class Preprocessor(object):
    """An abstract class representing a Preprocessor.

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


# main functionality - factory class
def load_preproc(path):
    """Load the preprocessor from disk

    1. Parse the yaml file
    2. Import the preprocessor
    3. Validate the pre-processor <-> yaml
      - check if the arguments match
    4. Append yaml description to __doc__
    5. Return the pre-processor
    """

    pass

# TODO - what if we could already put everything into the pre-processor's __doc__ string?

