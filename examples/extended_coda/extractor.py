# python2, 3 compatibility
from __future__ import absolute_import, division, print_function
from builtins import str, open, range, dict


import itertools
import numpy as np
import pybedtools

from genomelake.extractors import ArrayExtractor


def batch_iter(iterable, batch_size):
    """
    iterates in batches.
    """
    it = iter(iterable)
    try:
        while True:
            values = []
            for n in range(batch_size):
                values += (it.next(),)
            yield values
    except StopIteration:
        # yield remaining values
        yield values


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
