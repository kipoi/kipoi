import numpy as np
import sys
import collections
# string_classes
if sys.version_info[0] == 2:
    string_classes = basestring
else:
    string_classes = (str, bytes)


def _numpy_collate(stack_fn=np.stack):
    def numpy_collate_fn(batch):
        "Puts each data field into a tensor with outer dimension batch size"
        if type(batch[0]).__module__ == 'numpy':
            elem = batch[0]
            if type(elem).__name__ == 'ndarray':
                return stack_fn(batch, 0)
            if elem.shape == ():  # scalars
                return np.array(batch)
        elif isinstance(batch[0], int):
            return np.asarray(batch)
        elif isinstance(batch[0], float):
            return np.asarray(batch)
        elif isinstance(batch[0], string_classes):
            # Also convert to a numpy array
            return np.asarray(batch)
            # return batch
        elif isinstance(batch[0], collections.Mapping):
            return {key: numpy_collate_fn([d[key] for d in batch]) for key in batch[0]}
        elif isinstance(batch[0], collections.Sequence):
            transposed = zip(*batch)
            return [numpy_collate_fn(samples) for samples in transposed]

        raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                         .format(type(batch[0]))))
    return numpy_collate_fn


numpy_collate = _numpy_collate(np.stack)
numpy_collate_concat = _numpy_collate(np.concatenate)

# --------------------------------------------
# Tools for working with a nested dataset


def get_dataset_lens(data, require_numpy=False):
    if type(data).__module__ == 'numpy':
        if require_numpy and not data.shape:
            raise ValueError("all numpy arrays need to have at least one axis")
        return [1] if not data.shape else [data.shape[0]]
    elif isinstance(data, int) and not require_numpy:
        return [1]
    elif isinstance(data, float) and not require_numpy:
        return [1]
    elif isinstance(data, string_classes) and not require_numpy:
        # Also convert to a numpy array
        return [1]
        # return data
    elif isinstance(data, collections.Mapping) and not type(data).__module__ == 'numpy':
        return sum([get_dataset_lens(data[key], require_numpy) for key in data], [])
    elif isinstance(data, collections.Sequence) and not type(data).__module__ == 'numpy':
        return sum([get_dataset_lens(sample, require_numpy) for sample in data], [])
    else:
        raise ValueError("Leafs of the nested structure need to be numpy arrays")


def get_dataset_item(data, idx):
    if type(data).__module__ == 'numpy':
        return data[idx]
    elif isinstance(data, collections.Mapping):
        return {key: get_dataset_item(data[key], idx) for key in data}
    elif isinstance(data, collections.Sequence):
        return [get_dataset_item(sample, idx) for sample in data]
    else:
        raise ValueError("Leafs of the nested structure need to be numpy arrays")


def iter_cycle(it):
    """Alternative to itertools.cycle

    This function doesn't store the iterator elements into a list
    as itertools.cycle does
    """
    from itertools import tee
    while True:
        it, it_to_use = tee(it, 2)
        for x in it_to_use:
            yield x
