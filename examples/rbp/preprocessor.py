# python2, 3 compatibility
from __future__ import absolute_import, division, print_function
import six
from builtins import str, open, range, dict

import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool

from genomelake.extractors import ArrayExtractor, BaseExtractor, FastaExtractor, one_hot_encode_sequence, NUM_SEQ_CHARS
from pysam import FastaFile
from concise.utils.position import extract_landmarks, read_gtf, ALL_LANDMARKS


class DistToClosestLandmarkExtractor(BaseExtractor):
    """Extract distances to the closest genomic landmark

    # Arguments
        gtf_file: Genomic annotation file path (say gencode gtf)
        landmarks: List of landmarks to extract. See `concise.utils.position.extract_landmarks`
        use_strand: Take into account the strand of the intervals
    """
    multiprocessing_safe = True

    def __init__(self, gtf_file, landmarks=ALL_LANDMARKS, use_strand=True, **kwargs):
        super(DistToClosestLandmarkExtractor, self).__init__(gtf_file, **kwargs)
        self._gtf_file = gtf_file
        self.landmarks = extract_landmarks(gtf_file, landmarks=ALL_LANDMARKS)
        self.columns = ALL_LANDMARKS  # column names. Reqired for concating distances into array
        self.use_strand = use_strand

        # set index to chromosome and strand - faster access
        self.landmarks = {k: v.set_index(["seqnames", "strand"]) for k, v in six.iteritems(self.landmarks)}

    def _extract(self, intervals, out, **kwargs):

        def find_closest(ldm, interval, use_strand=True):
            """Uses
            """
            # subset the positions to the appropriate strand
            # and extract the positions
            ldm_positions = ldm.loc[interval.chrom]
            if use_strand and interval.strand != ".":
                ldm_positions = ldm_positions.loc[interval.strand]
            ldm_positions = ldm_positions.position.values

            int_midpoint = (interval.end + interval.start) // 2
            dist = (ldm_positions - 1) - int_midpoint  # -1 for 0, 1 indexed positions
            if use_strand and interval.strand == "-":
                dist = - dist

            return dist[np.argmin(np.abs(dist))]

        out[:] = np.array([[find_closest(self.landmarks[ldm_name], interval, self.use_strand)
                            for ldm_name in self.columns]
                           for interval in intervals], dtype=float)

        return out

    def _get_output_shape(self, num_intervals, width):
        return (num_intervals, len(self.columns))



# Get the intervals
intervals_file = "test_files/intervals.tsv"
gtf_file = "test_files/gencode_v25_chr22.gtf.pkl.gz"
fasta_file = "test_files/hg38_chr22.fa"
bt = pybedtools.BedTool(intervals_file)
intervals = [i for i in bt[:10]]

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


def extractor(intervals_file, fasta_file, gtf_file, target_data_sources=None, batch_size=4):
    """
    Args:
        intervals_file: tsv file
            Assumes bed - like `chrom start end id` format.
        input_data_sources: dict
            mapping from input name to genomelake directory
        target_data_sources: dict, optional
            mapping from input name to genomelake directory
        batch_size: int
    """
    # use only protein-coding genes
    gtf = pd.read_pickle(gtf_file)
    # gtf.rename(columns={"seqname": "seqnames"}, inplace=True)
    gtf = read_gtf(gtf_file)
    gtf = gtf[gtf["info"].str.contains('gene_type "protein_coding"')]

    # intervals
    bt = pybedtools.BedTool(intervals_file)

    # extractors
    input_data_extractors = {"seq": FastaFile(fasta_file),
                             "distance": DistToClosestLandmarkExtractor(gtf_file=gtf)
                             }

    intervals_generator = batch_iter(bt, batch_size)
    for intervals_batch in intervals_generator:
        out = {}
        # get data
        out['inputs'] = {key: extractor(intervals_batch)
                         for key, extractor in input_data_extractors.items()}

        # TODO - use trained spline transformation to transform it
        out["inputs"]["distance"] = out["inputs"]["distance"]

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
