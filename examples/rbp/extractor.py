# python2, 3 compatibility
from __future__ import absolute_import, division, print_function
from builtins import str, open, range, dict


import numpy as np
import pybedtools
from pybedtools import BedTool

from genomelake.extractors import ArrayExtractor, BaseExtractor, one_hot_encode_sequence, NUM_SEQ_CHARS
from pysam import FastaFile


# GTF -> landmark extractor (bed)
# landmark extractor + bed -> distances
#
# landmark as a string (exon_intron, ...)
# gtf + bed + landmarks


# TODO - extract the pre-trained pre-processor with the model
#        - allow for pickled pre-processors


class DistToClosestBedExtractor(BaseExtractor):
    """TODO - create a feature - distance to closes genomic_landmark
    TODO - index by chromosome and strand (faster query)
    TODO - argument - ignore strand 
    """
    multiprocessing_safe = True

    def __init__(self, datafile, **kwargs):
        super(DistToClosestBedExtractor, self).__init__(datafile, **kwargs)
        self._bed = BedTool(datafile)

    def _extract(self, intervals, out, **kwargs):
        # Sort intervals
        sorted_idx, sorted_intervals = zip(
            *sorted(enumerate(intervals),
                    key=lambda idx_interval: (idx_interval[1].chrom,
                                              idx_interval[1].start)))
        dists = [
            int(result[-1]) for result in
            # Report distance, and take only the first tie
            BedTool(sorted_intervals).closest(self._bed, d=True, t='first',
                                              stream=True)
        ]
        out[sorted_idx, 0, 0, 0] = dists
        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        return (num_intervals, 1, 1, 1)

    @staticmethod
    def mirror(data, to_mirror):
        # TODO: implement - I suspect this should just be a no-op
        raise NotImplementedError


class FastaExtractor(BaseExtractor):
    """
    Arguments:
        datafile: File path
        use_strand: bool, if True, queried sequence from the negative strand
    is reverse complemented
    """
    # TODO - make a pull-request later

    def __init__(self, datafile, use_strand=False, **kwargs):
        super(FastaExtractor, self).__init__(datafile, **kwargs)
        self.use_strand = use_strand

    def _extract(self, intervals, out, **kwargs):
        fasta = FastaFile(self._datafile)

        for index, interval in enumerate(intervals):
            seq = fasta.fetch(str(interval.chrom), interval.start,
                              interval.stop)
            one_hot_encode_sequence(seq, out[index, :, :])

            # reverse-complement seq the negative strand
            if self.use_strand and interval.strand == "-":
                out[index, :, :] = out[index, ::-1, ::-1]

        return out

    @staticmethod
    def _get_output_shape(num_intervals, width):
        return (num_intervals, width, NUM_SEQ_CHARS)


# TODO:
# - get the intervals.tsv
# - get the fasta (single chromosome, 22)
# - get the gtf (single chromosome, 22)


path = "/s/project/deepcis/encode/eclip/processed/design_matrix/test/PUM2_extended.csv"
gtf_path = "/s/genomes/human/hg38/GRCh38.p7/gencode.v25.annotation.gtf"
fafile = "/s/genomes/human/hg38/GRCh38.p7/GRCh38.p7.genome.fa"

import HTSeq

gf = HTSeq.GFF_Reader(gtf)

for feature in itertools.islice(gtffile, 100):

btg = BedTool(gtf)

# TODO write a unit-test
fastaFile = FastaFile(fafile)

import pandas as pd

intervals_file = "/data/nasif12/home_if12/avsec/projects-work/model-zoo/examples/rbp/test_files/intervals.tsv"
dt_bt = pd.read_table(intervals_file, header=None)
bt = BedTool(intervals_file)
bt[1].strand

fe = FastaExtractor(fafile, use_strand=True)

a = fe([i for i in bt[:10]])
from concise.preprocessing import encodeDNA
b = encodeDNA(dt_bt.iloc[:10, -1])

# debugging code
one_hot2string(a, ["A", "C", "G", "T"])
one_hot2string(b, ["A", "C", "G", "T"])
# TODO - convert to sequence
np.all(a == b)

# Next - get the nearest feature-point

# TODO - how to deal with the lengths
#        - there the ReLU is useful

# distances ...
# TODO -
# - extract the distances,
#   - specify the range beforehand
# - convert to spline transformation?
# -
#

# --------------------------------------------
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


intervals_file = "intervals.tsv"
input_data_sources = {"H3K27AC_subsampled": "H3K27AC_subsampled.bw"}


def extractor(intervals_file, input_data_sources, target_data_sources=None, batch_size=128):
    """
    Args:
        intervals_file: tsv file
            Assumes bed-like `chrom start end id` format.
        input_data_sources: dict
            mapping from input name to genomelake directory
        target_data_sources: dict, optional
            mapping from input name to genomelake directory
        batch_size: int
    """
    bt = pybedtools.BedTool(intervals_file)
    input_data_extractors = {key: ArrayExtractor(data_source)
                             for key, data_source in input_data_sources.items()}
    if target_data_sources is not None:
        target_data_extractors = {key: ArrayExtractor(data_source)
                                  for key, data_source in target_data_sources.items()}
    intervals_generator = batch_iter(bt, batch_size)
    for intervals_batch in intervals_generator:
        out = {}
        # get data
        out['inputs'] = {key: extractor(intervals_batch)[..., None]  # adds channel axis for conv1d
                         for key, extractor in input_data_extractors.items()}
        if target_data_sources is not None:
            out['targets'] = {key: extractor(intervals_batch)[..., None]  # adds channel axis for conv1d
                              for key, extractor in target_data_extractors.items()}
        # get metadata
        out['metadata'] = {}
        chrom = []
        start = []
        end = []
        ids = []
        for interval in intervals_batch:
            chrom.append(interval.chrom)
            start.append(interval.start)
            end.append(interval.stop)
            ids.append(interval.name)
        out['metadata']['chrom'] = np.array(chrom)
        out['metadata']['start'] = np.array(start)
        out['metadata']['end'] = np.array(end)
        out['metadata']['id'] = np.array(ids)

        yield out
