import os
import inspect
from builtins import str, open, range, dict

import pickle
import numpy as np
import pandas as pd
import pybedtools
from pybedtools import BedTool

from sklearn.preprocessing import FunctionTransformer
from pysam import FastaFile
from concise.preprocessing.splines import encodeSplines
from concise.utils.position import extract_landmarks, ALL_LANDMARKS
from kipoi.metadata import GenomicRanges
import linecache
from kipoi.data import Dataset
from kipoiseq.extractors import FastaStringExtractor
from kipoiseq.transforms.functional import one_hot_dna

# Reference: https://github.com/kundajelab/genomelake/blob/3f53f490c202fcbca83d6e4a9f1e5f2c68066133/genomelake/extractors.py#L15
NUM_SEQ_CHARS = 4
# Reference: https://github.com/kundajelab/genomelake/blob/3f53f490c202fcbca83d6e4a9f1e5f2c68066133/genomelake/extractors.py#L18-L51
class BaseExtractor(object):
    dtype = np.float32

    def __init__(self, datafile, **kwargs):
        self._datafile = datafile

    def __call__(self, intervals, out=None, **kwargs):
        data = self._check_or_create_output_array(intervals, out)
        self._extract(intervals, data, **kwargs)
        return data

    def _check_or_create_output_array(self, intervals, out):
        width = intervals[0].stop - intervals[0].start
        output_shape = self._get_output_shape(len(intervals), width)

        if out is None:
            out = np.zeros(output_shape, dtype=self.dtype)
        else:
            if out.shape != output_shape:
                raise ValueError('out array has incorrect shape: {} '
                                 '(need {})'.format(out.shape, output_shape))
            if out.dtype != self.dtype:
                raise ValueError('out array has incorrect dtype: {} '
                                 '(need {})'.format(out.dtype, self.dtype))
        return out

    def _extract(self, intervals, out, **kwargs):
        'Subclassses should implement this and return the data'
        raise NotImplementedError

    @staticmethod
    def _get_output_shape(num_intervals, width):
        'Subclasses should implement this and return the shape of output'
        raise NotImplementedError


filename = inspect.getframeinfo(inspect.currentframe()).filename
DATALOADER_DIR = os.path.dirname(os.path.abspath(filename))


def sign_log_func(x):
    return np.sign(x) * np.log10(np.abs(x) + 1)


def sign_log_func_inverse(x):
    return np.sign(x) * (np.power(10, np.abs(x)) - 1)


class BedToolLinecache(BedTool):
    """Fast BedTool accessor by Ziga Avsec

    Normal BedTools loops through the whole file to get the
    line of interest. Hence the access it o(n)
    """

    def __getitem__(self, idx):
        line = linecache.getline(self.fn, idx + 1)
        return pybedtools.create_interval_from_list(line.strip().split("\t"))


class DistanceTransformer:
    """Transforms the raw distances to the appropriate modeling form
    """

    def __init__(self, pos_features, pipeline_obj_path):
        """
        Args:
          pos_features: list of positional features to use
          pipeline_obj_path: path to the serialized pipeline obj_path
        """
        self.pos_features = pos_features
        self.pipeline_obj_path = pipeline_obj_path

        # deserialize the pickle file
        with open(self.pipeline_obj_path, "rb") as f:
            pipeline_obj = pickle.load(f)
        self.POS_FEATURES = pipeline_obj[0]
        self.minmax_scaler = pipeline_obj[1]
        self.imp = pipeline_obj[2]

        self.funct_transform = FunctionTransformer(func=sign_log_func,
                                                   inverse_func=sign_log_func_inverse)
        # for simplicity, assume all current pos_features are the
        # same as from before
        assert self.POS_FEATURES == self.pos_features

    def transform(self, x):
        # impute missing values and rescale the distances
        xnew = self.minmax_scaler.transform(self.funct_transform.transform(self.imp.transform(x)))

        # convert distances to spline bases
        dist = {"dist_" + k: encodeSplines(xnew[:, i, np.newaxis], start=0, end=1, warn=False)
                for i, k in enumerate(self.POS_FEATURES)}
        return dist


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
        self.landmarks = extract_landmarks(gtf_file, landmarks=landmarks)
        self.columns = landmarks  # column names. Reqired for concating distances into array
        self.use_strand = use_strand

        # set index to chromosome and strand - faster access
        self.landmarks = {k: v.set_index(["seqname", "strand"])
                          for k, v in self.landmarks.items()}

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


class TxtDataset(Dataset):

    def __init__(self, path):
        with open(path, "r") as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


# File paths
intervals_file = "test_files/intervals.tsv"
target_file = "test_files/targets.tsv"
gtf_file = "test_files/gencode_v25_chr22.gtf.pkl.gz"
fasta_file = "test_files/hg38_chr22.fa"
preproc_transformer = "extractor_files/encodeSplines.pkl"
# bt = pybedtools.BedTool(intervals_file)
# intervals = [i for i in bt[:10]]

# --------------------------------------------


class SeqDistDataset(Dataset):
    """
    Args:
        intervals_file: file path; tsv file
            Assumes bed-like `chrom start end id score strand` format.
        fasta_file: file path; Genome sequence
        gtf_file: file path; Genome annotation GTF file pickled using pandas.
        preproc_transformer: file path; tranformer used for pre-processing.
        target_file: file path; path to the targets
        batch_size: int
    """

    def __init__(self, intervals_file, fasta_file, gtf_file, preproc_transformer, target_file=None):
        gtf = pd.read_pickle(gtf_file)
        self.gtf = gtf[gtf["info"].str.contains('gene_type "protein_coding"')]
        self.gtf = self.gtf.rename(columns={"seqnames": "seqname"})  # concise>=0.6.5
        # distance transformer
        with open(preproc_transformer, "rb") as f:
            self.transformer = pickle.load(f)

        # intervals
        self.bt = pybedtools.BedTool(intervals_file)

        # extractors
        self.fasta_file = fasta_file
        self.input_data_extractors = None

        # target
        if target_file:
            self.target_dataset = TxtDataset(target_file)
            assert len(self.target_dataset) == len(self.bt)
        else:
            self.target_dataset = None

    def __len__(self):
        return len(self.bt)

    def __getitem__(self, idx):
        if self.input_data_extractors is None:
            self.input_data_extractors = {
                "seq": FastaStringExtractor(self.fasta_file),
                "dist_polya_st": DistToClosestLandmarkExtractor(gtf_file=self.gtf,
                                                                landmarks=["polya"])
            }
        interval = self.bt[idx]

        out = {}

        out['inputs'] = {  "seq": one_hot_dna(FastaStringExtractor(self.fasta_file).extract(interval)),
                           "dist_polya_st": np.squeeze(DistToClosestLandmarkExtractor(gtf_file=self.gtf,
                                                                landmarks=["polya"])([interval]), axis=0)
                        }
                        
        # use trained spline transformation to transform it
        out["inputs"]["dist_polya_st"] = np.squeeze(self.transformer.transform(out["inputs"]["dist_polya_st"][np.newaxis],
                                                                               warn=False), axis=0)

        if self.target_dataset is not None:
            out["targets"] = np.array([self.target_dataset[idx]])

        # get metadata
        out['metadata'] = {}
        out['metadata']['ranges'] = {}
        out['metadata']['ranges']['chr'] = interval.chrom
        out['metadata']['ranges']['start'] = interval.start
        out['metadata']['ranges']['end'] = interval.stop
        out['metadata']['ranges']['id'] = interval.name
        out['metadata']['ranges']['strand'] = interval.strand

        return out

# test batching
# from torch.utils.data import DataLoader

# dl = DataLoader(a, batch_size=3, collate_fn=numpy_collate)

# it = iter(dl)
# sample = next(it)

# sample["inputs"]["dist_polya_st"].shape
# sample["inputs"]["seq"].shape
