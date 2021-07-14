"""DeepSEA dataloader
"""
# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool
from kipoi.data import Dataset
from kipoi.metadata import GenomicRanges
from kipoiseq.extractors import FastaStringExtractor
import linecache
from kipoi.helper import one_hot_encode_sequence

# --------------------------------------------


class BedToolLinecache(BedTool):
    """Fast BedTool accessor by Ziga Avsec

    Normal BedTools loops through the whole file to get the
    line of interest. Hence the access it o(n)
    """

    def __getitem__(self, idx):
        line = linecache.getline(self.fn, idx + 1)
        return pybedtools.create_interval_from_list(line.strip().split("\t"))


class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in the csv format
    """

    def __init__(self, intervals_file, fasta_file, target_file=None, use_linecache=False):

        # intervals
        if use_linecache:
            self.bt = BedToolLinecache(intervals_file)
        else:
            self.bt = BedTool(intervals_file)
        self.fasta_file = fasta_file
        self.fasta_extractor = None

        # Targets
        if target_file is not None:
            self.targets = pd.read_csv(target_file)
        else:
            self.targets = None

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        if self.fasta_extractor is None:
            self.fasta_extractor = FastaStringExtractor(self.fasta_file)

        interval = self.bt[idx]

        # Intervals need to be 1000bp wide
        assert interval.stop - interval.start == 1000

        if self.targets is not None:
            y = self.targets.iloc[idx].values
        else:
            y = {}

        # Run the fasta extractor
        seq = one_hot_encode_sequence(self.fasta_extractor.extract(interval))
        return {
            "inputs": seq,
            "targets": y,
            "metadata": {
                "ranges": GenomicRanges.from_interval(interval)
            }
        }
