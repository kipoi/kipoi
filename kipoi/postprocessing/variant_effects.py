from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
import copy
import tempfile
from tqdm import tqdm
import itertools
import os

import warnings
from kipoi.components import ArraySpecialType, MetadataType
import six

def get_seq_len(input_data):
    if isinstance(input_data, (list, tuple)):
        return input_data[0].shape
    elif isinstance(input_data, dict):
        for k in input_data:
            return input_data[k].shape
    elif isinstance(input_data, np.ndarray):
        return input_data.shape
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")

def ism(model, ref, ref_rc, alt, alt_rc, mutation_positions, out_annotation_all_outputs,
        output_filter_mask=None, out_annotation=None, diff_type="log_odds", rc_handling="maximum"):
    """In-silico mutagenesis

    Using ISM in with diff_type 'log_odds' and rc_handling 'maximum' will produce predictions as used
    in [DeepSEA](http://www.nature.com/nmeth/journal/v12/n10/full/nmeth.3547.html). ISM offers two ways to
    calculate the difference between the outputs created by reference and alternative sequence and two
    different methods to select whether to use the output generated from the forward or from the
    reverse-complement sequences. To calculate "e-values" as mentioned in DeepSEA the same ISM prediction
    has to be performed on a randomised set of 1 million 1000genomes, MAF-matched variants to get a
    background of predicted effects of random SNPs.

    # Arguments
        model: Keras model
        ref: Input sequence with the reference genotype in the mutation position
        ref_rc: Reverse complement of the 'ref' argument
        alt: Input sequence with the alternative genotype in the mutation position
        alt_rc: Reverse complement of the 'alt' argument
        mutation_positions: Position on which the mutation was placed in the forward sequences
        out_annotation_all_outputs: Output labels of the model.
        output_filter_mask: Mask of boolean values indicating which model outputs should be used.
            Use this or 'out_annotation'
        out_annotation: List of outputs labels for which of the outputs (in case of a multi-task model) the
            predictions should be calculated.
        diff_type: "log_odds" or "diff". When set to 'log_odds' calculate scores based on log_odds, which assumes
            the model output is a probability. When set to 'diff' the model output for 'ref' is subtracted
            from 'alt'. Using 'log_odds' with outputs that are not in the range [0,1] nan will be returned.
        rc_handling: "average" or "maximum". Either average over the predictions derived from forward and
            reverse-complement predictions ('average') or pick the prediction with the bigger absolute
            value ('maximum').

    # Returns
        Dictionary with the key `ism` which contains a pandas DataFrame containing the calculated values
            for each (selected) model output and input sequence
    """

    seqs = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
    assert diff_type in ["log_odds", "diff"]
    assert rc_handling in ["average", "maximum"]
    assert np.all([np.array(get_seq_len(ref)) == np.array(get_seq_len(seqs[k])) for k in seqs.keys() if k != "ref"])
    assert get_seq_len(ref)[0] == mutation_positions.shape[0]
    assert len(mutation_positions.shape) == 1

    # determine which outputs should be selected
    if output_filter_mask is None:
        if out_annotation is None:
            output_filter_mask = np.arange(out_annotation_all_outputs.shape[0])
        else:
            output_filter_mask = np.where(np.in1d(out_annotation_all_outputs, out_annotation))[0]

    # make sure the labels are assigned correctly
    out_annotation = out_annotation_all_outputs[output_filter_mask]

    preds = {}
    for k in seqs:
        preds[k] = np.array(model.predict_on_batch(seqs[k])[..., output_filter_mask])

    if diff_type == "log_odds":
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))
        diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
    elif diff_type == "diff":
        diffs = preds["alt"] - preds["ref"]
        diffs_rc = preds["alt_rc"] - preds["ref_rc"]

    if rc_handling == "average":
        diffs = np.mean([diffs, diffs_rc], axis=0)
    elif rc_handling == "maximum":
        replace_filt = np.abs(diffs) < np.abs(diffs_rc)
        diffs[replace_filt] = diffs_rc[replace_filt]

    diffs = pd.DataFrame(diffs, columns=out_annotation)

    return {"ism": diffs}


def _vcf_to_regions(vcf_fpath, seq_length, id_delim=":"):
    # VCF files are 1-based, so the return value here is 1-based
    colnames = ["chrom", "pos", "id", "ref", "alt"]
    vcf = pd.read_csv(vcf_fpath, sep="\t", comment='#', header=None, usecols=range(len(colnames)))
    vcf.columns = colnames
    vcf["chrom"] = "chr" + vcf["chrom"].str.lstrip("chr")
    seq_length_half = int(seq_length / 2)
    l_offset = seq_length_half
    r_offset = seq_length_half - 1 + seq_length % 2
    ids = vcf["chrom"] + id_delim + vcf["pos"].astype(str) + id_delim + vcf["ref"] + id_delim + vcf["alt"]
    regions = pd.DataFrame({"line_id": ids, "chrom": vcf["chrom"].astype(np.str),
                            "start": vcf["pos"] - l_offset, "end": vcf["pos"] + r_offset})
    regions["ref"] = vcf["ref"]
    regions["alt"] = vcf["alt"]
    regions["varpos"] = vcf["pos"]
    return regions


def _bed3(regions, fpath):
    regions_0based = copy.deepcopy(regions)
    regions_0based["start"] = regions_0based["start"] - 1
    regions_0based[["chrom", "start", "end"]].to_csv(fpath, sep="\t", header=False, index=False)


def _generate_seq_sets(model_input, annotated_regions, dataloader):
    # annotated_regions comes from the vcf file
    # This function has to convert the DNA regions in the model input according to ref, alt, fwd, rc and
    # return a dictionary of which the keys are compliant with evaluation_function arguments
    #
    # TODO: Implement checking for preprocessors that output fwd and rc sequences themselves
    # TODO: Implement support for models that have fwd + rc sequence input at once
    #
    dna_slots = []
    ranges_slots = []

    # check for inputs with special_type ArraySpecialType.DNASeq
    for k in dataloader.output_schema.inputs:
        if dataloader.output_schema.inputs[k].special_type == ArraySpecialType.DNASeq:
            dna_slots.append(k)

    # check for metadata with type MetadataType.RANGES
    for k, v in six.iteritems(dataloader.output_schema.metadata):
        if v.type == MetadataType.RANGES:
            ranges_slots.append(k)
    #
    if len(ranges_slots) == 0:
        raise ValueError("No metadata with type: Ranges found")
    #
    # there will at max be one element in the ranges_slots object
    # extract the metadata output
    # TODO - use associated_metadata to check
    ranges_input_obj = model_input['metadata'][ranges_slots[0]]
    #
    # annotated_regions are 1-based coordinates!
    annotated_regions = copy.deepcopy(annotated_regions)
    #
    annotated_regions["region"] = annotated_regions["chrom"] + ":" + annotated_regions["start"].astype(str) + "-" + annotated_regions["end"].astype(str)
    annotated_regions["varpos_rel"] = annotated_regions["varpos"] - annotated_regions["start"]
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
        preproc_out_ranges["strand"] = ["*"] * len(ranges_input_obj["chrom"])
    #
    preproc_out_ranges = pd.DataFrame(preproc_out_ranges)
    #
    # Annotate the sequences generated by the preprocessor
    preproc_conv = preproc_out_ranges.merge(annotated_regions, on="region")
    #
    # Generate the object for model prediction: Copy object if it will have to be modified
    input_set = {}
    seq_set_labels = ["fwd_ref", "fwd_alt", "rc_ref", "rc_alt"]
    for k in seq_set_labels:
        input_set[k] = {}
        for k2 in model_input['inputs']:
            if k2 in dna_slots:
                assert model_input['inputs'][k2].shape[1] == (annotated_regions["end"] - annotated_regions["start"] + 1).values[0]
                input_set[k][k2] = copy.deepcopy(model_input['inputs'][k2])
            else:
                input_set[k][k2] = model_input['inputs'][k2]
    #
    # Actually modify sequences according to annotation
    for s_dir, allele in itertools.product(["fwd", "rc"], ["ref", "alt"]):
        k = "%s_%s" % (s_dir, allele)
        for k2 in dna_slots:
            if k2 in input_set[k]:
                # Modify bases according to allele
                modify_bases(input_set[k][k2], preproc_conv["pp_line"].values, preproc_conv["varpos_rel"].values,
                             preproc_conv[allele].values, preproc_conv["strand"].values == "-")
                # subset to the lines that have been identified
                if input_set[k][k2].shape[0] != preproc_conv.shape[0]:
                    input_set[k][k2] = input_set[k][k2][preproc_conv["pp_line"].values, ...]
                # generate reverse complement if needed
                if s_dir == "rc":
                    input_set[k][k2] = input_set[k][k2][:, ::-1, ::-1]
    #
    # Reformat so that effect prediction function will get its required inputs
    pred_set = {"ref": input_set["fwd_ref"], "ref_rc": input_set["rc_ref"], "alt": input_set["fwd_alt"], "alt_rc": input_set["rc_alt"]}
    pred_set["mutation_positions"] = preproc_conv["varpos_rel"].values
    return pred_set


def modify_bases(seq_obj, lines, pos, base, is_rc):
    # Check whether this code does what it should...
    alphabet = np.array(['A', "C", "G", "T"])
    base_sel = np.where(alphabet[None, :] == base[:, None])
    base_sel_idx = base_sel[1][np.argsort(base_sel[0])]
    if is_rc.sum() != 0:
        pos[is_rc] = seq_obj.shape[1] - pos[is_rc] - 1
        base_sel_idx[is_rc] = alphabet.shape[0] - base_sel_idx[is_rc] - 1
    # Reset the base which was there from the preprocessor
    seq_obj[lines, pos, :] = 0
    # Set the allele
    seq_obj[lines, pos, base_sel_idx] = 1

def check_extractor_compatibility(extractor):
    # TODO: Implement a way to check whether the 'intervals_file' argument is availabel and if it's of type bed?
    pass

def predict_variants(model_handle,
                     vcf_fpath,
                     seq_length,
                     evaluation_function,
                     exec_files_path,
                     dataloader_function,
                     batch_size,
                     model_out_annotation,
                     evaluation_function = ism,
                     debug=False,
                     evaluation_function_kwargs=None):
    # if 'intervals_file' not in model_handle.preproc.get_avail_arguments():
    #    raise Exception("Preprocessor does not support DNA regions as input.")
    # seq_pp_outputs = model_handle.preproc.get_output_label_by_type("dna")
    # if len(seq_pp_outputs)==0:
    #    raise Exception("Preprocessor does not generate DNA sequences.")
    #
    # Check whether the extractor is compatible with variant effect prediction
    check_extractor_compatibility(dataloader_function)
    #
    regions = _vcf_to_regions(vcf_fpath, seq_length)
    temp_bed3_file = tempfile.mktemp()  # file path of the temp file
    _bed3(regions, temp_bed3_file)
    exec_files_path['intervals_file'] = temp_bed3_file
    #
    res = []
    #
    dataloader_function(**exec_files_path)
    #
    it = dataloader_function(**exec_files_path).batch_iter(batch_size=batch_size)
    # test that all predictions go through
    for i, batch in enumerate(tqdm(it)):
        eval_kwargs = _generate_seq_sets(batch, regions, dataloader_function)
        if evaluation_function_kwargs is not None:
            assert isinstance(evaluation_function_kwargs, dict)
            for k in evaluation_function_kwargs:
                eval_kwargs[k] = evaluation_function_kwargs[k]
        eval_kwargs["out_annotation_all_outputs"] = model_out_annotation
        if debug:
            for k in ["ref", "ref_rc", "alt", "alt_rc"]:
                print(k)
                print(model_handle.predict_on_batch(eval_kwargs[k]))
                print("".join(["-"] * 80))
        res.append(evaluation_function(model_handle, **eval_kwargs))
    #
    try:
        os.unlink(temp_bed3_file)
    except:
        pass
    return res
