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
import linecache
import pyfaidx
# --------------------------------------------


class BedToolLinecache(BedTool):
    """Fast BedTool accessor by Ziga Avsec

    Normal BedTools loops through the whole file to get the
    line of interest. Hence the access it o(n)
    """

    def __getitem__(self, idx):
        line = linecache.getline(self.fn, idx + 1)
        return pybedtools.create_interval_from_list(line.strip().split("\t"))

def to_onehot(seq):
    x = np.zeros((seq.shape[0], 4), dtype=np.float32)
    alphabet = ["A", "C", "G", "T"]
    for i in range(len(alphabet)):
        sel = np.where(seq == alphabet[i])
        x[sel[0], i] = 1
    return x

class FastaExtractor(object):
    """
    Class by Roman Kreuzhuber
    Fasta extractor using pyfaidx. Complies with genomelake.extractors.FastaExtractor I/O as used here.
    """

    def __init__(self, fasta_file_path):
        self.faidx_obj = pyfaidx.Fasta(fasta_file_path)

    def __call__(self, intervals):
        assert isinstance(intervals, list)
        one_hots = []
        for interval in intervals:
            # pyfaidx uses 1-based cooridnates!
            seq = np.array(list(self.faidx_obj.get_seq(interval.chrom ,interval.start+1,interval.end).seq.upper()))
            one_hots.append(to_onehot(seq))
        return np.array(one_hots)


class SeqDataset(Dataset):
    """
    Args:
        intervals_file: bed3 file containing intervals
        fasta_file: file path; Genome sequence
        target_file: file path; path to the targets in the csv format
    """

    SEQ_WIDTH = 500

    def __init__(self, intervals_file, fasta_file,
                 target_file=None, use_linecache=False):

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
            self.fasta_extractor = FastaExtractor(self.fasta_file)
        interval = self.bt[idx]

        if interval.stop - interval.start != self.SEQ_WIDTH:
            center = (interval.start + interval.stop) // 2
            interval.start = center - self.SEQ_WIDTH // 2
            interval.end = center + self.SEQ_WIDTH // 2 + self.SEQ_WIDTH % 2

        if self.targets is not None:
            y = self.targets.iloc[idx].values
        else:
            y = {}

        # Run the fasta extractor
        seq = np.squeeze(self.fasta_extractor([interval]), axis=0)
        return {
            "inputs": seq,
            "targets": y,
            "metadata": {
                "ranges": GenomicRanges.from_interval(interval)
            }
        }
