from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
import copy
import tempfile
from tqdm import tqdm
import itertools
import os
import re
from collections import OrderedDict

import warnings
from kipoi.components import PostProcType
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Reshape_dna(object):
    def __init__(self, in_shape):
        in_shape = np.array(in_shape)
        none_pos = np.in1d(in_shape, None)
        if np.any(none_pos) and (np.where(none_pos)[0][0] != 0):
            raise Exception("Unexpected 'None' shape in other dimension than the first!")
        else:
            in_shape = in_shape[~none_pos]
        #
        self.in_shape = in_shape
        dummy_dimensions = in_shape == 1
        #
        # Imitate np.squeeze() in self.to_standard()
        in_shape = in_shape[in_shape != 1]
        #
        if in_shape.shape[0] != 2:
            raise Exception("Invalid DNA shape definition definition!")
        #
        one_hot_dim = in_shape == 4
        if one_hot_dim.sum() != 1:
            raise Exception("Could not find 1-hot encoded dimension!")
        #
        seq_len_dim = (in_shape != 1) & (~one_hot_dim)
        if seq_len_dim.sum() != 1:
            raise Exception("Could not find sequence length dimension!")
        #
        dummy_dimensions = np.where(dummy_dimensions)[0]
        one_hot_dim = np.where(one_hot_dim)[0][0]
        seq_len_dim = np.where(seq_len_dim)[0][0]
        #
        self.dummy_dimensions = dummy_dimensions
        self.one_hot_dim = one_hot_dim
        self.seq_len_dim = seq_len_dim
        self.seq_len = in_shape[seq_len_dim]
        self.reshape_needed = True
        if (len(dummy_dimensions) == 0) and (one_hot_dim == 1) and (seq_len_dim == 0):
            self.reshape_needed = False

    def get_seq_len(self):
        return self.seq_len

    def to_standard(self, in_array):
        """
        :param in_array: has to have the sequence samples in the 0th dimension
        :return:
        """
        #
        #  is there an actual sequence sample axis?
        additional_axis = len(in_array.shape) - len(self.in_shape)
        if (additional_axis != 1) or (in_array.shape[1:] != tuple(self.in_shape)):
            raise Exception("Expecting the 0th dimension to be the sequence samples or general array mismatch!")
        #
        if not self.reshape_needed:
            return in_array
        squeezed = in_array
        #
        # Iterative removal of dummy dimensions has to start from highest dimension
        for d in sorted(self.dummy_dimensions)[::-1]:
            squeezed = np.squeeze(squeezed, axis = d+additional_axis)
        # check that the shape is now as expected:
        one_hot_dim_here = additional_axis + self.one_hot_dim
        seq_len_dim_here = additional_axis + self.seq_len_dim
        if squeezed.shape[one_hot_dim_here] != 4:
            raise Exception("Input array does not follow the input definition!")
        #
        if squeezed.shape[seq_len_dim_here] != self.seq_len:
            raise Exception("Input array sequence length does not follow the input definition!")
        #
        if self.one_hot_dim != 1:
            assert (self.seq_len_dim == 1) # Anything else would be weird...
            squeezed = squeezed.swapaxes(one_hot_dim_here, seq_len_dim_here)
        return squeezed

    def from_standard(self, in_array):
        if not self.reshape_needed:
            return in_array
        #
        assumed_additional_axis = 1
        #
        if in_array.shape[1:] != (self.seq_len, 4):
            raise Exception("Input array doesn't agree with standard format (n_samples, seq_len, 4)!")
        #
        one_hot_dim_here = assumed_additional_axis + self.one_hot_dim
        seq_len_dim_here = assumed_additional_axis + self.seq_len_dim
        if self.one_hot_dim != 1:
            in_array = in_array.swapaxes(one_hot_dim_here, seq_len_dim_here)
        #
        if len(self.dummy_dimensions) != 0:
            for d in self.dummy_dimensions:
                in_array = np.expand_dims(in_array,d+assumed_additional_axis)
        return in_array


def _get_seq_len(input_data):
    if isinstance(input_data, (list, tuple)):
        return input_data[0].shape
    elif isinstance(input_data, dict):
        for k in input_data:
            return input_data[k].shape
    elif isinstance(input_data, np.ndarray):
        return input_data.shape
    else:
        raise ValueError("Input can only be of type: list, dict or np.ndarray")


class Pred_analysis(object):
    def __call__(self, ref, ref_rc, alt, alt_rc):
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
    def __call__(self, ref, ref_rc, alt, alt_rc):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))
        diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
        return self.rc_merging(diffs, diffs_rc)

class Diff(Rc_merging_pred_analysis):
    def __call__(self, ref, ref_rc, alt, alt_rc):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        diffs = preds["alt"] - preds["ref"]
        diffs_rc = preds["alt_rc"] - preds["ref_rc"]
        return self.rc_merging(diffs, diffs_rc)

class DeepSEA_effect(Rc_merging_pred_analysis):
    def __call__(self, ref, ref_rc, alt, alt_rc):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logit_diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))
        logit_diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
        diffs = preds["alt"] - preds["ref"]
        diffs_rc = preds["alt_rc"] - preds["ref_rc"]
        return self.rc_merging(np.abs(logit_diffs) * np.abs(diffs), np.abs(logit_diffs_rc) * np.abs(diffs_rc))


class Output_reshaper(object):
    def __init__(self, model_target_schema, group_delim = "."):
        self.model_target_schema = model_target_schema
        self.standard_dict_order =None # This one is used to always produce the same order of outputs for a dict
        # extract output labels correctly.
        if isinstance(model_target_schema, dict):
            anno = {}
            # Reproducible dict output order:
            self.standard_dict_order = list(model_target_schema.keys())
            if not isinstance(model_target_schema, OrderedDict):
                self.standard_dict_order = sorted(self.standard_dict_order)
            for k in model_target_schema:
                group_name = str(k)
                out_group = model_target_schema[k]
                if out_group.name is not None:
                    group_name = out_group.name
                group_labels = [group_name + group_delim + label for label in self.get_column_names(out_group)]
                anno[k] = np.array(group_labels)
        elif isinstance(model_target_schema, list):
            anno = []
            for i, out_group in enumerate(model_target_schema):
                group_name = str(i)
                if out_group.name is not None:
                    group_name = out_group.name
                group_labels = [group_name + group_delim + label for label in self.get_column_names(out_group)]
                anno.append(np.array(group_labels))
        else:
            anno = self.get_column_names(model_target_schema)
        self.anno = anno

    def get_flat_labels(self):
        if isinstance(self.anno, dict):
            labels = []
            for k in self.standard_dict_order:
                labels.append(self.anno[k])
            flat_labels = np.concatenate(labels, axis=0)
        elif isinstance(self.anno, list):
            flat_labels = np.concatenate(self.anno, axis=0)
        else:
            flat_labels = self.anno
        return flat_labels

    def flatten(self, ds):
        if isinstance(ds, dict):
            if not isinstance(self.anno, dict):
                raise Exception("Error in model output defintion: Model definition is"
                                "of type %s but predictions are of type %s!"%(str(type(ds)), str(type(self.anno))))
            outputs = []
            labels = []
            for k in self.standard_dict_order:
                assert(ds[k].shape[1] == self.anno[k].shape[0])
                outputs.append(ds[k])
                labels.append(self.anno[k])
            flat = np.concatenate(outputs, axis=1)
            flat_labels = np.concatenate(labels, axis=0)
        elif isinstance(ds, list):
            if not isinstance(self.anno, list):
                raise Exception("Error in model output defintion: Model definition is"
                                "of type %s but predictions are of type %s!"%(str(type(ds)), str(type(self.anno))))
            assert len(ds) == len(self.anno)
            flat = np.concatenate(ds, axis=1)
            flat_labels = np.concatenate(self.anno, axis=0)
        else:
            flat = ds
            flat_labels = self.anno
        assert flat.shape[1] == flat_labels.shape[0]
        return flat, flat_labels

    @staticmethod
    def get_column_names(arrayschema_obj):
        if arrayschema_obj.column_labels is not None:
            ret = np.array(arrayschema_obj.column_labels)
        else:
            res_shape = [dim for dim in arrayschema_obj.shape if dim is not None]
            if len(res_shape) > 1:
                raise NotImplementedError("Don't know how to deal with multi-dimensional model target %s"%str(arrayschema_obj))
            #if res_shape[0] == 1:
            #    ret = np.array([""])
            #else:
            ret = np.arange(res_shape[0]).astype(np.str)
        return ret

def analyse_model_preds(model, ref, ref_rc, alt, alt_rc, mutation_positions, diff_types,
        output_reshaper, output_filter =None, **kwargs):
    seqs = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
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
    vcf = vcf.loc[(vcf["ref"].str.len() == 1) &(vcf["alt"].str.len() == 1),:]
    vcf["chrom"] = "chr" + vcf["chrom"].astype(str).str.lstrip("chr")
    seq_length_half = int(seq_length / 2)
    l_offset = seq_length_half
    r_offset = seq_length_half - 1 + seq_length % 2
    ids = vcf["chrom"] + id_delim + vcf["pos"].astype(str) + id_delim + vcf["ref"] + id_delim +\
        vcf["alt"].apply(lambda x: x.split(",")[0])
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


def _process_sequence_set(input_set, preproc_conv, allele, s_dir, array_trafo= None):
    # make sure the sequence objects have the correct length (acording to the ranges specifications)
    if array_trafo is not None:
        input_set = array_trafo.to_standard(input_set)
    assert input_set.shape[1] == \
        (preproc_conv["end"] - preproc_conv["start"] + 1).values[0]
    # Modify bases according to allele
    _modify_bases(input_set, preproc_conv["pp_line"].values,
                  preproc_conv["varpos_rel"].values,
                  preproc_conv[allele].values, preproc_conv["strand"].values == "-")
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


def _get_seq_fields(model):
    seq_dict = None
    for pp_obj in model.postprocessing:
        if pp_obj.type == PostProcType.VAR_EFFECT_PREDICTION:
            seq_dict = pp_obj.args
            break
    if seq_dict is None:
        raise Exception("Model does not support var_effect_prediction")
    # TODO: Is there a non-hardcoding way of selecting the seuqence labels?
    return seq_dict['seq_input']


def _get_dl_bed_fields(dataloader):
    seq_dict = None
    for pp_obj in dataloader.postprocessing:
        if pp_obj.type == PostProcType.VAR_EFFECT_PREDICTION:
            seq_dict = pp_obj.args
            break
    if seq_dict is None:
        raise Exception("Dataloader does not support any postprocessing")
    # TODO: Is there a non-hardcoding way of selecting the seuqence labels?
    return seq_dict['bed_input']


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

def _get_seq_shape(dataloader, seq_field):
    if isinstance(dataloader.output_schema.inputs, dict):
        orig_shape = dataloader.output_schema.inputs[seq_field].shape
    elif isinstance(dataloader.output_schema.inputs, list):
        orig_shape = [x.shape for x in dataloader.output_schema.inputs if x.name == seq_field][0]
    else:
        orig_shape = dataloader.output_schema.inputs.shape
    return orig_shape


# TODO - how to deal with a pre-specified bed file?

# TODO - speedup this function (see suggestions in the function)

# MAIN function
def predict_snvs(model,
                 vcf_fpath,
                 dataloader,
                 batch_size,
                 num_workers=0,
                 dataloader_args=None,
                 model_out_annotation=None,
                 evaluation_function=analyse_model_preds,
                 debug=False,
                 evaluation_function_kwargs={'diff_types': {'logit': Logit()}},
                 out_vcf_fpath=None,
                 use_dataloader_example_data=False):
    """Predict the effect of SNVs

        Prediction of effects of SNV based on a VCF. If desired the VCF can be stored with the predicted values as
        annotation. For a detailed description of the requirements in the yaml files please take a look at
        kipoi/nbs/variant_effect_prediction.ipynb.

        # Arguments
            model: A kipoi model handle generated by e.g.: kipoi.get_model()
            vcf_fpath: Path of the VCF defining the positions that shall be assessed. Only SNVs will be tested.
            dataloader: Dataloader factory generated by e.g.: kipoi.get_dataloader_factory()
            batch_size: Prediction batch size used for calling the data loader. Each batch will be generated in 4
                mutated states yielding a system RAM consumption of >= 4x batch size.
            num_workers: Number of parallel workers for loading the dataset.
            dataloader_arguments: arguments passed on to the dataloader for sequence generation, arguments
                mentioned in dataloader.yaml > postprocessing > variant_effects > bed_input will be overwritten
                by the methods here.
            model_out_annotation: Columns of the model output can be passed here, otherwise they will be attepted to be
                loaded from model.yaml > schema > targets > column_labels
            evaluation_function: effect evaluation function. Default is ism
            evaluation_function_kwargs: kwargs passed on to the evaluation function.
            out_vcf_fpath: Path for the annotated VCF, which is created from `vcf_fpath` one predicted effect is
                produced for every output (target) column.
            use_dataloader_example_data: Fill out the missing dataloader arguments with the example values given in the
                dataloader.yaml.

        # Returns
            Dictionary which contains a pandas DataFrame containing the calculated values
                for each model output (target) column VCF SNV line
        """

    seq_fields = _get_seq_fields(model)
    seq_shapes = set([_get_seq_shape(dataloader, seq_field) for seq_field in seq_fields])

    if len(seq_shapes) > 1:
        raise Exception("DNA sequence output shapes must agree for fields: %s"%str(seq_fields))

    seq_shape = list(seq_shapes)[0]

    dna_seq_trafo = Reshape_dna(seq_shape)
    seq_length = dna_seq_trafo.get_seq_len()

    regions = _vcf_to_regions(vcf_fpath, seq_length)
    temp_bed3_file = tempfile.mktemp()  # file path of the temp file
    _bed3(regions, temp_bed3_file)

    # Assemble the paths for executing the dataloader
    if dataloader_args is None:
        dataloader_args = {}

    # Copy the missing arguments from the example arguments.
    if use_dataloader_example_data:
        for k in dataloader.example_kwargs:
            if k not in dataloader_args:
                dataloader_args[k] = dataloader.example_kwargs[k]

    # Where do I have to put my bed file in the command?
    exec_files_bed_keys = _get_dl_bed_fields(dataloader)
    for k in exec_files_bed_keys:
        dataloader_args[k] = temp_bed3_file

    # Get model output annotation:
    if model_out_annotation is None:
        if isinstance(model.schema.targets, dict):
            raise Exception("Variant effect prediction with dict(array) model output not implemented!")
            # model_out_annotation = np.array(list(model.schema.targets.keys()))
        elif isinstance(model.schema.targets, list):
            #raise Exception("Variant effect prediction with list(array) model output not implemented!")
            model_out_annotation = np.array([x.name for x in model.schema.targets])
        else:
            # TODO - all targets need to have the keys defined
            if model.schema.targets.column_labels is not None:
                model_out_annotation = np.array(model.schema.targets.column_labels)

    if model_out_annotation is None:
        model_out_annotation = np.array([str(i) for i in range(model.schema.targets.shape[0])])

    out_reshaper = Output_reshaper(model.schema.targets)

    res = []

    it = dataloader(**dataloader_args).batch_iter(batch_size=batch_size,
                                                  num_workers=num_workers)

    # pre-process regions
    regions = _prepare_regions(regions)

    # test that all predictions go through
    keys = set()
    for i, batch in enumerate(tqdm(it)):
        # For debugging
        # if i >= 10:
        #     break
        # becomes noticable for large vcf's. Is there a way to avoid it? (i.e. to exploit the iterative nature of dataloading)
        eval_kwargs = _generate_seq_sets(seq_fields, dataloader, batch, regions, array_trafo = dna_seq_trafo)
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
        res_here = evaluation_function(model, output_reshaper = out_reshaper, **eval_kwargs)
        for k in res_here:
            keys.add(k)
            res_here[k].index = eval_kwargs["line_id"]
        res.append(res_here)

    res_concatenated = {}
    for k in keys:
        res_concatenated[k] = pd.concat([batch[k]
                                         for batch in res
                                         if k in batch])

    # actually annotate the VCF:
    if out_vcf_fpath is not None:
        if (model.info.name is None) or (model.info.name == ""):
            model_name = model.info.doc[:15] + ":" + model.info.version
        else:
            model_name = model.info.name + ":" + str(model.info.version)
        # TODO - write out the vcf lines already while processing the dataloader. this will save memory space
        # TODO - takes long for large vcf's
        _annotate_vcf(vcf_fpath, out_vcf_fpath, res_concatenated, model_name = model_name)

    try:
        os.unlink(temp_bed3_file)
    except:
        pass
    return res_concatenated


def prep_str(s):
    #https://stackoverflow.com/questions/1007481/how-do-i-replace-whitespaces-with-underscore-and-vice-versa
    # Remove all non-word characters (everything except numbers and letters)
    #s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"[^\w\.\:\s]+", '', s)
    #
    # Replace all runs of whitespace with a single underscore
    s = re.sub(r"\s+", '_', s)
    #
    return s


def concat_columns(df, sep="|"):
    """Concatenate all columns of a dataframe into a pd.Series
    """
    for i in range(df.shape[1]):
        vec = df.iloc[:, i].astype(str)
        if i == 0:
            column = vec
        else:
            column = column.str.cat(vec, sep=sep)
    return column


def _annotate_vcf(in_vcf_fpath, out_vcf_fpath, predictions, id_delim=":", model_name=None):
    # Use the ranges object to match predictions with the vcf
    # Add original index to the ranges object
    # Sort predictions according to the vcf
    # Annotate vcf object
    def _generate_info_field(id, num, info_type, desc, source, version):
        return vcf.parser._Info(id, num,
                                info_type, desc,
                                source, version)
    import vcf
    vcf_reader = vcf.Reader(open(in_vcf_fpath, 'r'))
    column_labels = None
    # Generate the info tag for the VEP
    info_tag_prefix = "KPVEP"
    if (model_name is not None) or(model_name != ""):
        info_tag_prefix += "_%s" % prep_str(model_name)

    # setup the header
    # predictions_concat = {}
    for k in predictions:
        info_tag = info_tag_prefix+"_%s" % k.upper()
        N = len(predictions[k])
        col_labels_here = predictions[k].columns.tolist()
        # Make sure that the column are consistent across different prediciton methods
        if column_labels is None:
            column_labels = col_labels_here
        else:
            if not np.all(np.array(column_labels) == np.array(col_labels_here)):
                raise Exception("Prediction columns are not identical for methods %s and %s" % (predictions.keys()[0], k))
        # Add the tag to the vcf file
        vcf_reader.infos[info_tag] = _generate_info_field(info_tag, None, 'String',
                                                          "%s SNV effect prediction. Prediction from model outputs: %s" % (k.upper(), "|".join(column_labels)),
                                                          None, None)
        # setup the right format
        # predictions_concat[k] = concat_columns(predictions[k], "|")
    vcf_writer = vcf.Writer(open(out_vcf_fpath, 'w'), vcf_reader)

    logger.info("Writing the vcf file to: {0}".format(out_vcf_fpath))
    for record in tqdm(vcf_reader, total=N):
        # Assemble line id as before for bed file generation
        line_id = "chr" + str(record.CHROM).lstrip("chr") + id_delim + str(record.POS) + id_delim + str(record.REF) + id_delim + str(record.ALT[0])
        for k in predictions:
            # In case there is a pediction for this line, annotate the vcf...
            if line_id in predictions[k].index:
                info_tag = info_tag_prefix+"_{0}".format(k.upper())
                # record.INFO[info_tag] = predictions_concat[k][line_id]
                preds = predictions[k].loc[line_id, :]
                # preds = preds.astype(str)
                record.INFO[info_tag] = "|".join([str(pred) for pred in preds])
        vcf_writer.write_record(record)
    vcf_writer.close()
    logger.info("Writing done!")
