from __future__ import absolute_import
from __future__ import print_function

import copy
import itertools
import logging
import warnings

import numpy as np
import pandas as pd

from kipoi.postprocessing.utils.generic import _get_seq_len

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Pred_analysis(object):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        raise NotImplementedError("Analysis routine has to be implemented")


class Rc_merging_pred_analysis(Pred_analysis):
    allowed_str_opts = ["min", "max", "mean", "median", "absmax"]
    #

    def __init__(self, rc_merging="max"):
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
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
            return self.rc_merging(diffs, diffs_rc)
        else:
            return diffs


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
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logit_diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))
        logit_diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
        diffs = preds["alt"] - preds["ref"]

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            diffs_rc = preds["alt_rc"] - preds["ref_rc"]
            return self.rc_merging(np.abs(logit_diffs) * np.abs(diffs), np.abs(logit_diffs_rc) * np.abs(diffs_rc))
        else:
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
    assert np.all([np.array(_get_seq_len(ref)) == np.array(_get_seq_len(seqs[k])) for k in seqs.keys() if k != "ref"])
    assert _get_seq_len(ref)[0] == mutation_positions.shape[0]
    assert len(mutation_positions.shape) == 1

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


def _process_sequence_set(input_set, preproc_conv, allele, s_dir, array_trafo=None):
    # make sure the sequence objects have the correct length (acording to the ranges specifications)
    if array_trafo is not None:
        input_set = array_trafo.to_standard(input_set)
    assert input_set.shape[1] == \
        (preproc_conv["end"] - preproc_conv["start"] + 1).values[0]
    assert preproc_conv["strand"].isin(["+", "-", "*", "."]).all()
    # Modify bases according to allele
    _modify_bases(seq_obj=input_set,
                  lines=preproc_conv["pp_line"].values,
                  pos=preproc_conv["varpos_rel"].values,
                  base=preproc_conv[allele].values,
                  is_rc=preproc_conv["strand"].values == "-")
    # subset to the lines that have been identified
    if input_set.shape[0] != preproc_conv.shape[0]:
        raise Exception("Mismatch between requested and generated DNA sequences.")
        # input_set[k][seq_key] = input_set[k][seq_key][preproc_conv["pp_line"].values, ...]
    # generate reverse complement if needed
    if s_dir == "rc":
        input_set = input_set[:, ::-1, ::-1]
    if array_trafo is not None:
        input_set = array_trafo.from_standard(input_set)
    return input_set


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


def _generate_seq_sets(relv_seq_keys, dataloader, model_input, annotated_regions, array_trafo=None):
    # annotated_regions comes from the vcf file
    # This function has to convert the DNA regions in the model input according to ref, alt, fwd, rc and
    # return a dictionary of which the keys are compliant with evaluation_function arguments
    #
    # DataLoaders that implement fwd and rc sequence output at once are not treated in any special way.
    #
    # Generate 4 full copies of the input set
    input_set = {}
    for s_dir, allele in itertools.product(["fwd", "rc"], ["ref", "alt"]):
        k = "%s_%s" % (s_dir, allele)
        input_set[k] = copy.deepcopy(model_input['inputs'])
    #
    preproc_conv_prev = None
    # Start from the sequence inputs mentioned in the model.yaml
    for seq_key in relv_seq_keys:
        if isinstance(dataloader.output_schema.inputs, dict):
            ranges_slots = dataloader.output_schema.inputs[seq_key].associated_metadata
        elif isinstance(dataloader.output_schema.inputs, list):
            ranges_slots = [x.associated_metadata for x in dataloader.output_schema.inputs if x.name == seq_key][0]
        else:
            ranges_slots = dataloader.output_schema.inputs.associated_metadata
        # check the ranges slots
        if len(ranges_slots) != 1:
            raise ValueError("Exactly one metadata ranges field must defined for a sequence that has to be used for effect precition.")
        #
        # there will at max be one element in the ranges_slots object
        # extract the metadata output
        ranges_input_obj = model_input['metadata'][ranges_slots[0]]
        #
        #
        # Object that holds annotation of the sequences
        preproc_out_ranges = {}
        preproc_out_ranges["pp_line"] = list(range(len(ranges_input_obj["chr"])))
        preproc_out_ranges["region"] = ["%s:%d-%d" % (ranges_input_obj["chr"][i],
                                                      ranges_input_obj["start"][i] + 1,
                                                      ranges_input_obj["end"][i])
                                        for i in range(len(ranges_input_obj["chr"]))]
        #
        # Get the strand of the output sequences from the preprocessor
        if "strand" in ranges_input_obj:
            preproc_out_ranges["strand"] = ranges_input_obj["strand"]
        else:
            preproc_out_ranges["strand"] = ["*"] * len(ranges_input_obj["chr"])
        #
        preproc_out_ranges["id"] = ranges_input_obj["id"].astype(str)
        preproc_out_ranges = pd.DataFrame(preproc_out_ranges)
        preproc_out_ranges.set_index("id", inplace=True)
        #
        # Annotate the sequences generated by the preprocessor by string matching, keep the order after preprocessing

        # Using a join with pre-set indexes works faster than merge
        preproc_conv = preproc_out_ranges.join(annotated_regions,
                                               how="left",
                                               lsuffix='_DL',
                                               rsuffix="_Q").reset_index()
        if not (preproc_conv["region_DL"] == preproc_conv["region_Q"]).all():
            raise Exception("Couldn't merge dataloader metadata back to query sequences!")
        # Make sure that all the merged region ids are annotated with the alternative allele
        assert (~(preproc_conv["alt"].isnull()).any())
        # Make sure that the mutated position for all the DNA sequence tracks that should be modified is the same
        if preproc_conv_prev is not None:
            assert np.all(preproc_conv["varpos_rel"].values == preproc_conv_prev["varpos_rel"].values)
        preproc_conv_prev = preproc_conv
        # Actually modify sequences according to annotation
        for s_dir, allele in itertools.product(["fwd", "rc"], ["ref", "alt"]):
            k = "%s_%s" % (s_dir, allele)
            if isinstance(dataloader.output_schema.inputs, dict):
                if seq_key not in input_set[k]:
                    raise Exception("Sequence field %s is missing in DataLoader output!" % seq_key)
                input_set[k][seq_key] = _process_sequence_set(input_set[k][seq_key], preproc_conv, allele, s_dir, array_trafo)
            elif isinstance(dataloader.output_schema.inputs, list):
                modified_set = []
                for seq_el, input_schema_el in zip(input_set[k], dataloader.output_schema.inputs):
                    if input_schema_el.name == seq_key:
                        modified_set.append(_process_sequence_set(seq_el, preproc_conv, allele, s_dir, array_trafo))
                    else:
                        modified_set.append(seq_el)
                input_set[k] = modified_set
            else:
                input_set[k] = _process_sequence_set(input_set[k], preproc_conv, allele, s_dir, array_trafo)
    #
    # Reformat so that effect prediction function will get its required inputs
    pred_set = {"ref": input_set["fwd_ref"], "ref_rc": input_set["rc_ref"], "alt": input_set["fwd_alt"], "alt_rc": input_set["rc_alt"]}
    pred_set["mutation_positions"] = preproc_conv["varpos_rel"].values
    pred_set["line_id"] = preproc_conv["line_id"].values
    return pred_set


def _modify_bases(seq_obj, lines, pos, base, is_rc):
    # Assumes a fixed order of ACGT and requires one-hot encoding
    alphabet = np.array(['A', "C", "G", "T"])
    base_sel = np.where(alphabet[None, :] == base[:, None])
    base_sel_idx = base_sel[1][np.argsort(base_sel[0])]
    pos = copy.deepcopy(pos)
    if is_rc.sum() != 0:
        # is_rc is given relative to the input data, so it as to be transferred back into the variant definition order.
        is_rc_corr = is_rc[lines]
        pos[is_rc_corr] = seq_obj.shape[1] - pos[is_rc_corr] - 1
        base_sel_idx[is_rc_corr] = alphabet.shape[0] - base_sel_idx[is_rc_corr] - 1
    # Reset the base which was there from the preprocessor
    seq_obj[lines, pos, :] = 0
    # Set the allele
    seq_obj[lines, pos, base_sel_idx] = 1


def _get_seq_length(dataloader, seq_field):
    if isinstance(dataloader.output_schema.inputs, dict):
        orig_shape = dataloader.output_schema.inputs[seq_field].shape
    elif isinstance(dataloader.output_schema.inputs, list):
        orig_shape = [x.shape for x in dataloader.output_schema.inputs if x.name == seq_field][0]
    else:
        orig_shape = dataloader.output_schema.inputs.shape
    shape = [s for s in orig_shape if s is not None]
    shape = [s for s in shape if s != 4]
    if len(shape) != 1:
        raise Exception("DNA sequence output shape not well defined! %s" % str(orig_shape))
    return shape[0]

