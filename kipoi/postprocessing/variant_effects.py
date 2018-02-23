from __future__ import absolute_import
from __future__ import print_function

import copy
import itertools
import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Pred_analysis(object):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        raise NotImplementedError("Analysis routine has to be implemented")


class Rc_merging_pred_analysis(Pred_analysis):
    allowed_str_opts = ["min", "max", "mean", "median", "absmax"]
    #

    def __init__(self, rc_merging="mean"):
        if isinstance(rc_merging, str):
            if rc_merging == "absmax":
                self.rc_merging = self.absmax
            elif rc_merging in self.allowed_str_opts:
                self.rc_merging = lambda x, y: getattr(np, rc_merging)([x, y], axis=0)
        elif callable(rc_merging):
            self.rc_merging = rc_merging
        else:
            raise Exception("rc_merging has to be a callable function of a string: %s" % str(self.allowed_str_opts))
    #

    @staticmethod
    def absmax(x, y, inplace=True):
        if not inplace:
            x = copy.deepcopy(x)
        replace_filt = np.abs(x) < np.abs(y)
        x[replace_filt] = y[replace_filt]
        return x


class Logit(Rc_merging_pred_analysis):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
            return self.rc_merging(diffs, diffs_rc)
        else:
            return diffs


class LogitAlt(Rc_merging_pred_analysis):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logits = np.log(preds["alt"] / (1 - preds["alt"]))

        if preds["alt_rc"] is not None:
            logits_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"]))
            return self.rc_merging(logits, logits_rc)
        else:
            return logits


class LogitRef(Rc_merging_pred_analysis):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logits = np.log(preds["ref"] / (1 - preds["ref"]))

        if preds["ref_rc"] is not None:
            logits_rc = np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
            return self.rc_merging(logits, logits_rc)
        else:
            return logits


class Diff(Rc_merging_pred_analysis):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        diffs = preds["alt"] - preds["ref"]

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            diffs_rc = preds["alt_rc"] - preds["ref_rc"]
            return self.rc_merging(diffs, diffs_rc)
        else:
            return diffs


class DeepSEA_effect(Rc_merging_pred_analysis):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logit_diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))
        diffs = preds["alt"] - preds["ref"]

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            logit_diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
            diffs_rc = preds["alt_rc"] - preds["ref_rc"]
            logit_diffs = self.rc_merging(logit_diffs, logit_diffs_rc)
            diffs = self.rc_merging(diffs, diffs_rc)
            #self.rc_merging(np.abs(logit_diffs) * np.abs(diffs), np.abs(logit_diffs_rc) * np.abs(diffs_rc))

        return np.abs(logit_diffs) * np.abs(diffs)


def analyse_model_preds(model, ref, alt, mutation_positions, diff_types,
                        output_reshaper, output_filter=None, ref_rc=None, alt_rc=None, **kwargs):
    seqs = {"ref": ref, "alt": alt}
    if ref_rc is not None:
        seqs["ref_rc"] = ref_rc
    if alt_rc is not None:
        seqs["alt_rc"] = alt_rc
    if not isinstance(diff_types, dict):
        raise Exception("diff_types has to be a dictionary of callables. Keys will be used to annotate output.")
    # This is deprecated as no simple deduction of sequence length is possible anymore
    #assert np.all([np.array(_get_seq_len(ref)) == np.array(_get_seq_len(seqs[k])) for k in seqs.keys() if k != "ref"])
    #assert _get_seq_len(ref)[0] == mutation_positions.shape[0]
    #assert len(mutation_positions.shape) == 1

    # Make predictions
    preds = {}
    out_annotation = None
    for k in seqs:
        # Flatten the model output
        preds_out, pred_labels = output_reshaper.flatten(model.predict_on_batch(seqs[k]))
        if out_annotation is None:
            out_annotation = pred_labels
            output_filter = np.zeros(pred_labels.shape[0]) == 0
            # determine which outputs should be selected
            if output_filter is None:
                if output_filter.dtype == bool:
                    assert(output_filter.shape == out_annotation.shape)
                else:
                    assert np.all(np.in1d(output_filter, out_annotation))
                    output_filter = np.in1d(out_annotation, output_filter)
        # Filter outputs if required
        preds[k] = np.array(preds_out[..., output_filter])

    # Run the analysis callables
    outputs = {}
    for k in diff_types:
        outputs[k] = pd.DataFrame(diff_types[k](**preds), columns=out_annotation[output_filter])

    return outputs


def _vcf_to_regions(vcf_fpath, seq_length, id_delim=":"):
    # VCF files are 1-based, so the return value here is 1-based
    colnames = ["chrom", "pos", "id", "ref", "alt"]
    vcf = pd.read_csv(vcf_fpath, sep="\t", comment='#', header=None, usecols=range(len(colnames)))
    vcf.columns = colnames
    # Subset the VCF to SNVs:
    vcf = vcf.loc[(vcf["ref"].str.len() == 1) & (vcf["alt"].str.len() == 1), :]
    vcf["chrom"] = "chr" + vcf["chrom"].astype(str).str.lstrip("chr")
    seq_length_half = int(seq_length / 2)
    l_offset = seq_length_half
    r_offset = seq_length_half - 1 + seq_length % 2
    ids = vcf["chrom"] + id_delim + vcf["pos"].astype(str) + id_delim + vcf["ref"] + id_delim +\
        vcf["alt"].apply(lambda x: str(x.split(",")))
    regions = pd.DataFrame({"line_id": ids, "chrom": vcf["chrom"].astype(np.str),
                            "start": vcf["pos"] - l_offset, "end": vcf["pos"] + r_offset})
    regions["ref"] = vcf["ref"]
    regions["alt"] = vcf["alt"].apply(lambda x: x.split(",")[0])
    regions["varpos"] = vcf["pos"]
    regions["id"] = np.arange(regions.shape[0])
    return regions


def _bed3(regions, fpath):
    """write the regions vcf file to a bed file
    """
    regions_0based = copy.deepcopy(regions)
    regions_0based["start"] = regions_0based["start"] - 1
    to_write = regions_0based[["chrom", "start", "end"]]
    if "id" in regions_0based.columns:
        to_write = regions_0based[["chrom", "start", "end", "id"]]
    to_write.to_csv(fpath, sep="\t", header=False, index=False)


def _prepare_regions(annotated_regions):
    # A bit of regions annotation and string matching to modify the DNA sequences at the correct positions.
    # annotated_regions are 1-based coordinates!
    # using x.values.astype(np.str) instead of x.astype(str) yields a decent speedup
    # https://stackoverflow.com/questions/26744370/most-efficient-way-to-convert-pandas-series-of-integers-to-strings
    annotated_regions["region"] = annotated_regions["chrom"] + ":" + annotated_regions["start"].values.astype(np.str) + "-" + \
        annotated_regions["end"].values.astype(np.str)
    annotated_regions["varpos_rel"] = annotated_regions["varpos"] - annotated_regions["start"]
    annotated_regions["id"] = annotated_regions["id"].values.astype(np.str)
    annotated_regions.set_index("id", inplace=True)
    return annotated_regions


def _modify_bases(seq_obj, lines, pos, base, is_rc, return_ref_warning=False):
    # Assumes a fixed order of ACGT and requires one-hot encoding
    assert lines.shape[0] == pos.shape[0]
    assert base.shape[0] == pos.shape[0]
    alphabet = np.array(['A', "C", "G", "T"])
    base_sel = np.where(alphabet[None, :] == base[:, None])
    base_sel_idx = (np.zeros(base.shape[0], dtype=np.int) - 1)
    base_sel_idx[base_sel[0]] = base_sel[1]
    pos = copy.deepcopy(pos)
    # get bases that are not in the alphabet
    known_bases = base_sel_idx != -1
    if is_rc.sum() != 0:
        # is_rc is given relative to the input data, so it as to be transferred back into the variant definition order.
        pos[is_rc] = seq_obj.shape[1] - pos[is_rc] - 1
        base_sel_idx[is_rc] = alphabet.shape[0] - base_sel_idx[is_rc] - 1
    warn_lines = []
    if return_ref_warning:
        wrong_ref = seq_obj[lines[known_bases], pos[known_bases], base_sel_idx[known_bases]] != 1
        warn_lines = np.array(lines[known_bases])[wrong_ref].tolist()
    # Reset the base which was there from the preprocessor
    # Reset the unknown bases just to 0
    seq_obj[lines, pos, :] = 0
    # Set the allele
    seq_obj[lines[known_bases], pos[known_bases], base_sel_idx[known_bases]] = 1
    return warn_lines


def rc_str(dna):
    """Reverse complement a string
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    ks = list(complement.keys())
    for k in ks:
        complement[k.lower()] = complement[k].lower()
    return ''.join([complement[base] for base in dna[::-1]])

def _modify_single_string_base(seq_obj, pos, base, is_rc):
    """modify the single base of a string"""
    if is_rc:
        pos = len(seq_obj) - pos - 1
        base = rc_str(base)
    ret_obj = seq_obj[:pos] + base + seq_obj[(pos + 1):]
    return ret_obj


