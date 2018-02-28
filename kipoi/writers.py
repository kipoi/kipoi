"""Writers used in `kipoi predict`

- TsvWriter
- HDF5Writer
"""
from __future__ import absolute_import
from __future__ import print_function
from abc import abstractmethod
import numpy as np
import pandas as pd
from collections import OrderedDict
from kipoi.utils import flatten_nested

class BatchWriter(object):

    @abstractmethod
    def batch_write(self, dl_batch, pred_batch):
        """Write a single batch of data

        Args:
          dl_batch is one batch returned by the dataloader
          pred_batch are the predictions made for the dl_batch
        """
        pass

    @abstractmethod
    def close(self):
        """Close the file
        """
        pass

# --------------------------------------------

# helper functions


def io_batch2df(dl_batch, pred_batch,
                add_inputs=False, add_targets=False, add_metadata=True,
                nested_sep="/"):
    """Convert the batch + prediction batch to a pd.DataFrame
    """
    if not isinstance(pred_batch, np.ndarray) or pred_batch.ndim > 2:
        raise ValueError("Model's output is not a 1D or 2D np.ndarray")

    # TODO - generalize to multiple arrays (list of arrays)

    if pred_batch.ndim == 1:
        pred_batch = pred_batch[:, np.newaxis]
    df = pd.DataFrame(pred_batch,
                      columns=["pred{0}{1}".format(nested_sep, i)
                               for i in range(pred_batch.shape[1])])

    # TODO - flatten the output - use slashes in the column names
    if "metadata" in dl_batch and "ranges" in dl_batch["metadata"]:
        rng = dl_batch["metadata"]["ranges"]
        df_ranges = pd.DataFrame(OrderedDict([("chr", rng.get("chr")),
                                              ("start", rng.get("start")),
                                              ("end", rng.get("end")),
                                              ("name", rng.get("id", None)),
                                              ("score", rng.get("score", None)),
                                              ("strand", rng.get("strand", None))]))
        df = pd.concat([df_ranges, df], axis=1)

    # TODO - re-write with flatten_nested
    flatten_nested(pred_batch)
    return df


# --------------------------------------------

class TsvBatchWriter(BatchWriter):

    def __init__(self, file_path):
        """

        """
        self.file_path = file_path
        self.first_pass = True

    def batch_write(self, dl_batch, pred_batch):
        df = io_batch2df(dl_batch, pred_batch)
        if self.first_pass:
            df.to_csv(self.file_path, sep="\t", index=False)
            self.first_pass = False
        else:
            df.to_csv(self.file_path, sep="\t", index=False, header=None, mode="a")

    def close(self):
        # nothing to do
        pass
