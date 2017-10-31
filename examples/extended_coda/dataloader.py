# python2, 3 compatibility
from __future__ import absolute_import, division, print_function

import numpy as np
import pybedtools
import six
from genomelake.extractors import ArrayExtractor

from kipoi.data import Dataset


class CodaDataset(Dataset):

    def __init__(self, intervals_file, input_data_sources, target_data_sources=None):
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
        self.bt = pybedtools.BedTool(intervals_file)
        self.input_data_extractors = {key: ArrayExtractor(data_source)
                                      for key, data_source in six.iteritems(input_data_sources)}

        self.target_data_sources = target_data_sources
        if self.target_data_sources is not None:
            self.target_data_extractors = {key: ArrayExtractor(data_source)
                                           for key, data_source in six.iteritems(target_data_sources)}

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        interval = self.bt[idx]

        out = {}
        out['inputs'] = {key: np.squeeze(extractor([interval])[..., None], 0)  # adds channel axis for conv1d
                         for key, extractor in six.iteritems(self.input_data_extractors)}

        if self.target_data_sources is not None:
            out['targets'] = {key: np.squeeze(extractor(interval)[..., None], 0)  # adds channel axis for conv1d
                              for key, extractor in six.iteritems(self.target_data_extractors)}
        # get metadata
        out['metadata'] = {}
        out['metadata']['ranges'] = {}
        out['metadata']['ranges']['chr'] = interval.chrom
        out['metadata']['ranges']['start'] = interval.start
        out['metadata']['ranges']['end'] = interval.stop
        out['metadata']['ranges']['id'] = interval.name

        return out
