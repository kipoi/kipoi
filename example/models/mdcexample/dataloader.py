from kipoi.data import SampleIterator
from kipoiseq import Interval, Variant
from kipoiseq.transforms import OneHot
from kipoiseq.extractors import VariantSeqExtractor, SingleVariantMatcher, BaseExtractor, FastaStringExtractor
from kipoi.metadata import GenomicRanges

import pandas as pd
import pyranges as pr

from kipoiseq.variant_source import VariantFetcher


class APARENT_DL(SampleIterator):
    def __init__(
            self,
            regions_of_interest: pr.PyRanges,
            reference_sequence: BaseExtractor,
            interval_attrs=('gene_id', 'transcript_id')
    ):
        self.regions_of_interest = regions_of_interest
        self.reference_sequence = reference_sequence
        self.interval_attrs = interval_attrs

        self.variant_seq_extractor = VariantSeqExtractor(reference_sequence=reference_sequence)

        self.one_hot = OneHot()

    def __iter__(self):
        interval: Interval
        variant: Variant

        for index, row in self.regions_of_interest.as_df().iterrows():
            interval = Interval(
                chrom=row["Chromosome"],
                start=row["Start"],
                end=row["End"],
                strand=row["Strand"],
            )
            yield {
                "inputs": self.one_hot(self.reference_sequence.extract(interval)),
                "metadata": {
                    "ranges": GenomicRanges.from_interval(interval),
                    **{k: row[k] for k in self.interval_attrs},
                }
            }


def get_roi_from_cse(cse_start: int, cse_end: int, is_on_negative_strand: bool) -> (int, int):
    """
    Get region-of-interest for APARENT in relation to the canonical sequence element (CSE) position
    :param cse_start: 0-based start position of CSE
    :param cse_end: 1-based end position of CSE
    :param is_on_negative_strand: is the gene on the negative strand?
    :return: Tuple of (start, end) position for the region of interest
    """
    # CSE should be around position 70 of the 205bp sequence.
    if is_on_negative_strand:
        end = cse_end + 70
        start = end - 205
    else:
        start = cse_start - 70
        end = start + 205

    return start, end


def get_roi_from_transcript(transcript_start: int, transcript_end: int, is_on_negative_strand: bool) -> (int, int):
    """
    Get region-of-interest for APARENT in relation to the 3'UTR of a transcript
    :param transcript_start: 0-based start position of the transcript
    :param transcript_end: 1-based end position of the transcript
    :param is_on_negative_strand: is the gene on the negative strand?
    :return: Tuple of (start, end) position for the region of interest
    """
    # CSE should be roughly around position 70 of the 205bp sequence.
    # Since CSE is likely 30bp upstream of the cut site, we shift the cut site
    #   by 100bp upstream and 105bp downstream
    if is_on_negative_strand:
        end = transcript_start + 100
        # convert 0-based to 1-based
        end += 1

        start = end - 205
    else:
        start = transcript_end - 100
        # convert 1-based to 0-based
        start -= 1

        end = start + 205

    return start, end


def get_roi_from_genome_annotation(genome_annotation: pd.DataFrame):
    """
    Get region-of-interest for APARENT from some genome annotation
    :param genome_annotation: Pandas dataframe with the following columns:
        - Chromosome
        - Start
        - End
        - Strand
        - Feature
        - gene_id
        - transcript_id
    :return:
    """
    roi = genome_annotation.query("`Feature` == 'transcript'")
    roi = roi.assign(
        transcript_start=roi["Start"],
        transcript_end=roi["End"],
    )

    def adjust_row(row):
        start, end = get_roi_from_transcript(row.Start, row.End, row.Strand)
        row.Start = start
        row.End = end

        return row

    roi = roi.apply(adjust_row, axis=1)

    return roi


class Kipoi_APARENT_DL(APARENT_DL):
    def __init__(
            self,
            fasta_file,
            gtf_file,
    ):
        genome_annotation = pr.read_gtf(gtf_file, as_df=True)
        roi = get_roi_from_genome_annotation(genome_annotation)
        roi = pr.PyRanges(roi)

        super().__init__(
            regions_of_interest=roi,
            reference_sequence=FastaStringExtractor(fasta_file),
        )