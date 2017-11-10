"""Module defining different metadat classess
"""
from collections import Mapping
from kipoi.data_utils import numpy_collate, numpy_collate_concat


class GenomicRanges(Mapping):
    def __init__(self, chr, start, end, id, strand="*"):
        """

        """
        self._storage = dict(chr=chr, start=start, end=end, id=id, strand=strand)
        self.chr = chr
        self.start = start
        self.end = end
        self.id = id
        self.strand = strand

    def __getitem__(self, key):
        return self._storage[key]

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    @classmethod
    def from_interval(cls, interval):
        """Create the ranges object from the interval
        """
        return cls(chr=interval.chrom,
                   start=interval.start,
                   end=interval.end,
                   id=interval.name,
                   strand=interval.strand)

    @classmethod
    def from_dict(cls, obj):
        return cls(**obj)

    @classmethod
    def collate(cls, obj_list, collate_fn=numpy_collate):
        """Defines the collate property - used in BatchIterator?
        """
        assert all([isinstance(x, cls) for x in obj_list])
        return cls.from_dict(collate_fn(obj_list))
