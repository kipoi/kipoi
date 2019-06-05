"""DeepSEA dataloader
"""
# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
from kipoi.data import Dataset
class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in the csv format
    """

    def __init__(self, dummy = None):
        pass

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return {
            "inputs": np.random.rand(1, 10).astype(np.float32)
        }
