import numpy as np
import pybedtools



def batch_iter(iterable, batch_size):
    """
    iterates in batches.
    """
    it = iter(iterable)
    try:
        while True:
            values = []
            for n in range(batch_size):
                values += (next(it),)
            yield values
    except StopIteration:
        # yield remaining values
        yield values