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
from kipoi.components import PostProcType

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
        output_filter_mask=None, out_annotation=None, diff_type="log_odds", rc_handling="maximum", **kwargs):
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
    # Subset the VCF to SNVs:
    vcf = vcf.loc[(vcf["ref"].str.len() == 1) &(vcf["ref"].str.len() == 1),:]
    vcf["chrom"] = "chr" + vcf["chrom"].str.lstrip("chr")
    seq_length_half = int(seq_length / 2)
    l_offset = seq_length_half
    r_offset = seq_length_half - 1 + seq_length % 2
    ids = vcf["chrom"] + id_delim + vcf["pos"].astype(str) + id_delim + vcf["ref"] + id_delim +\
          vcf["alt"].apply( lambda x: x.split(",")[0])
    regions = pd.DataFrame({"line_id": ids, "chrom": vcf["chrom"].astype(np.str),
                            "start": vcf["pos"] - l_offset, "end": vcf["pos"] + r_offset})
    regions["ref"] = vcf["ref"]
    regions["alt"] = vcf["alt"].apply( lambda x: x.split(",")[0])
    regions["varpos"] = vcf["pos"]
    return regions


def _bed3(regions, fpath):
    regions_0based = copy.deepcopy(regions)
    regions_0based["start"] = regions_0based["start"] - 1
    regions_0based[["chrom", "start", "end"]].to_csv(fpath, sep="\t", header=False, index=False)


def _process_sequence_set(input_set, preproc_conv, allele, s_dir):
    # make sure the sequence objects have the correct length (acording to the ranges specifications)
    assert input_set.shape[1] == \
           (preproc_conv["end"] - preproc_conv["start"] + 1).values[0]
    # Modify bases according to allele
    modify_bases(input_set, preproc_conv["pp_line"].values,
                 preproc_conv["varpos_rel"].values,
                 preproc_conv[allele].values, preproc_conv["strand"].values == "-")
    # subset to the lines that have been identified
    if input_set.shape[0] != preproc_conv.shape[0]:
        raise Exception("Mismatch between requested and generated DNA sequences.")
        # input_set[k][seq_key] = input_set[k][seq_key][preproc_conv["pp_line"].values, ...]
    # generate reverse complement if needed
    if s_dir == "rc":
        input_set = input_set[:, ::-1, ::-1]
    return input_set


def _generate_seq_sets(relv_seq_keys, dataloader, model_input, annotated_regions):
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
            ranges_slots = [x.metadata for x in dataloader.output_schema.inputs if x.name == seq_key][0]
        else:
            ranges_slots = dataloader.output_schema.inputs.metadata
        # check the ranges slots
        if len(ranges_slots) != 1:
            raise ValueError("Exactly one metadata ranges field must defined for a sequence that has to be used for effect precition.")
        #
        # there will at max be one element in the ranges_slots object
        # extract the metadata output
        ranges_input_obj = model_input['metadata'][ranges_slots[0]]
        #
        # A bit of regions annotation and string matching to modify the DNA sequences at the correct positions.
        # annotated_regions are 1-based coordinates!
        annotated_regions_cp = copy.deepcopy(annotated_regions)
        #
        annotated_regions_cp["region"] = annotated_regions_cp["chrom"] + ":" + annotated_regions_cp["start"].astype(str) + "-" + \
                                         annotated_regions_cp["end"].astype(str)
        annotated_regions_cp["varpos_rel"] = annotated_regions_cp["varpos"] - annotated_regions_cp["start"]
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
        # Annotate the sequences generated by the preprocessor by string matching, keep the order after preprocessing
        preproc_conv = preproc_out_ranges.merge(annotated_regions_cp, on="region", how="left")
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
                input_set[k][seq_key] = _process_sequence_set(input_set[k][seq_key], preproc_conv, allele, s_dir)
            elif isinstance(dataloader.output_schema.inputs, list):
                modified_set = []
                for seq_el, input_schema_el in zip(input_set[k], dataloader.output_schema.inputs):
                    if input_schema_el.name == seq_key:
                        modified_set.append(_process_sequence_set(seq_el, preproc_conv, allele, s_dir))
                    else:
                        modified_set.append(seq_el)
                input_set[k] = modified_set
            else:
                input_set[k] = _process_sequence_set(input_set[k], preproc_conv, allele, s_dir)
    #
    # Reformat so that effect prediction function will get its required inputs
    pred_set = {"ref": input_set["fwd_ref"], "ref_rc": input_set["rc_ref"], "alt": input_set["fwd_alt"], "alt_rc": input_set["rc_alt"]}
    pred_set["mutation_positions"] = preproc_conv["varpos_rel"].values
    pred_set["line_id"] = preproc_conv["line_id"].values
    return pred_set



def modify_bases(seq_obj, lines, pos, base, is_rc):
    # Assumes a fixed order of ACGT and requires one-hot encoding
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


def get_seq_fields(model):
    seq_dict = None
    for pp_obj in model.postprocessing:
        if pp_obj.type == PostProcType.VAR_EFFECT_PREDICTION:
            seq_dict = pp_obj.args
            break
    if seq_dict is None:
        raise Exception("Model does not support var_effect_prediction")
    # TODO: Is there a non-hardcoding way of selecting the seuqence labels?
    return seq_dict['seq_input']

def get_dl_bed_fields(dataloader):
    seq_dict = None
    for pp_obj in dataloader.postprocessing:
        if pp_obj.type == PostProcType.VAR_EFFECT_PREDICTION:
            seq_dict = pp_obj.args
            break
    if seq_dict is None:
        raise Exception("Dataloader does not support any postprocessing")
    # TODO: Is there a non-hardcoding way of selecting the seuqence labels?
    return seq_dict['bed_input']


def get_seq_length(dataloader, seq_field):
    orig_shape = dataloader.output_schema.inputs[seq_field].shape
    shape = [s for s in orig_shape if s is not None]
    shape = [s for s in shape if s != 4]
    if len(shape) != 1:
        raise Exception("DNA sequence output shape not well defined! %s"%str(orig_shape))
    return shape[0]



def predict_snvs(model,
                 vcf_fpath,
                 dataloader,
                 batch_size,
                 dataloader_arguments = None,
                 model_out_annotation = None,
                 evaluation_function = ism,
                 debug=False,
                 evaluation_function_kwargs=None,
                 out_vcf_fpath = None,
                 use_dataloader_example_data = False):
    # if 'intervals_file' not in model_handle.preproc.get_avail_arguments():
    #    raise Exception("Preprocessor does not support DNA regions as input.")
    # seq_pp_outputs = model_handle.preproc.get_output_label_by_type("dna")
    # if len(seq_pp_outputs)==0:
    #    raise Exception("Preprocessor does not generate DNA sequences.")
    #
    seq_fields = get_seq_fields(model)
    seq_lengths = set([get_seq_length(dataloader, seq_field) for seq_field in seq_fields])
    if len(seq_lengths) > 1:
        raise Exception("DNA sequence output shapes must agree for fields: %s"%str(seq_fields))
    else:
        seq_length = list(seq_lengths)[0]
    regions = _vcf_to_regions(vcf_fpath, seq_length)
    temp_bed3_file = tempfile.mktemp()  # file path of the temp file
    _bed3(regions, temp_bed3_file)
    # Assemble the paths for executing the dataloader
    if dataloader_arguments is None:
        dataloader_arguments = {}
    # Copy the missing arguments from the example arguments.
    if use_dataloader_example_data:
        for k in dataloader.example_kwargs:
            if k not in dataloader_arguments:
                dataloader_arguments[k] = dataloader.example_kwargs[k]
    # Where do I have to put my bed file in the command?
    exec_files_bed_keys = get_dl_bed_fields(dataloader)
    for k in exec_files_bed_keys:
        dataloader_arguments[k] = temp_bed3_file
    #
    # Get model output annotation:
    if model_out_annotation is None:
        if isinstance(model.schema.targets, dict):
            raise Exception("Variant effect prediction with dict(array) model output not implemented!")
            #model_out_annotation = np.array(list(model.schema.targets.keys()))
        elif isinstance(model.schema.targets, list):
            raise Exception("Variant effect prediction with list(array) model output not implemented!")
            #model_out_annotation = np.array([x.name for x in model.schema.targets])
        else:
            # TODO - all targets need to have the keys defined
            if model.schema.targets.column_labels is not None:
                model_out_annotation = np.array(model.schema.targets.column_labels)


    if model_out_annotation is None:
        model_out_annotation = np.array([str(i) for i in range(model.schema.targets.shape[0])])
    #
    res = []
    #
    dataloader(**dataloader_arguments)
    #
    it = dataloader(**dataloader_arguments).batch_iter(batch_size=batch_size)
    # test that all predictions go through
    for i, batch in enumerate(tqdm(it)):
        eval_kwargs = _generate_seq_sets(seq_fields, dataloader, batch, regions)
        if evaluation_function_kwargs is not None:
            assert isinstance(evaluation_function_kwargs, dict)
            for k in evaluation_function_kwargs:
                eval_kwargs[k] = evaluation_function_kwargs[k]
        eval_kwargs["out_annotation_all_outputs"] = model_out_annotation
        if debug:
            for k in ["ref", "ref_rc", "alt", "alt_rc"]:
                print(k)
                print(model.predict_on_batch(eval_kwargs[k]))
                print("".join(["-"] * 80))
        res_here = evaluation_function(model, **eval_kwargs)
        for k in res_here:
            res_here[k].index = eval_kwargs["line_id"]
        res.append(res_here)
    #
    res_concatenated = {}
    for batch in res:
        for k in batch:
            if k not in res_concatenated:
                res_concatenated[k] = batch[k]
            else:
                res_concatenated[k] = pd.concat(res_concatenated[k], batch[k])
    # actually annotate the VCF:
    if out_vcf_fpath is not None:
        annotate_vcf(vcf_fpath, out_vcf_fpath, res_concatenated)
    #
    try:
        os.unlink(temp_bed3_file)
    except:
        pass
    return res_concatenated


def annotate_vcf(in_vcf_fpath, out_vcf_fpath, predictions, id_delim = ":"):
    # Use the ranges object to match predictions with the vcf
    # Add original index to the ranges object
    # Sort predictions according to the vcf
    # Annotate vcf object
    def _generate_info_field(id, num, info_type, desc, source, version):
        return  vcf.parser._Info(id, num,
                                 info_type, desc,
                                 source, version)
    import vcf
    vcf_reader = vcf.Reader(open(in_vcf_fpath, 'r'))
    column_labels = None
    # Generate the info tag for the VEP
    for k in predictions:
        info_tag = "KPVEP_%s"%k.upper()
        col_labels_here = predictions[k].columns.tolist()
        # Make sure that the column are consistent across different prediciton methods
        if column_labels is None:
            column_labels = col_labels_here
        else:
            if not np.all(np.array(column_labels) == np.array(col_labels_here)):
                raise Exception("Prediction columns are not identical for methods %s and %s"%(predictions.keys()[0], k))
        # Add the tag to the vcf file
        vcf_reader.infos[info_tag] = _generate_info_field(info_tag, None, 'String',
                    "%s SNV effect prediction. Prediction from model outputs: %s"%(k.upper(), "|".join(column_labels)),
                    None, None)
    vcf_writer = vcf.Writer(open(out_vcf_fpath, 'w'), vcf_reader)
    # Write VCF records to the output file
    for record in vcf_reader:
        # Assemble line id as before for bed file generation
        line_id = record.CHROM + id_delim + str(record.POS) + id_delim + str(record.REF) + id_delim + str(record.ALT[0])
        for k in predictions:
            # In case there is a pediction for this line, annotate the vcf...
            if line_id in predictions[k].index:
                info_tag = "KPVEP_%s" % k.upper()
                preds = predictions[k].loc[line_id,:].tolist()
                record.INFO[info_tag] = "|".join([str(pred) for pred in preds])
        vcf_writer.write_record(record)
    vcf_writer.close()
