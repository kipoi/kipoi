from __future__ import absolute_import
from __future__ import print_function

from abc import abstractmethod

import numpy as np
from collections import OrderedDict
import re
import kipoi
import logging

import kipoi
from kipoi.postprocessing.variant_effects import _modify_bases, _modify_single_string_base, rc_str

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def ensure_tabixed_vcf(input_fn, is_sorted=False, force_tabix=True):
    import pybedtools
    import pysam
    pbh = pybedtools.BedTool(input_fn)
    fn = input_fn
    if not pbh._tabixed():
        # pybedtools bug.
        fn = pbh.bgzip(in_place=True, force=force_tabix)
        pysam.tabix_index(fn, force=force_tabix, preset="vcf")
        #tbxd = pbh.tabix(is_sorted=is_sorted, force=force_tabix)
        #fn = tbxd.fn
    return fn

def prep_str(s):
    # https://stackoverflow.com/questions/1007481/how-do-i-replace-whitespaces-with-underscore-and-vice-versa
    # Remove all non-word characters (everything except numbers and letters)
    # s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"[^\w\.\:\s]+", '', s)
    #
    # Replace all runs of whitespace with a single underscore
    s = re.sub(r"\s+", '_', s)
    #
    return s


def select_from_dl_batch(obj, rows, nrows_expected=None):
    def subset(in_obj):
        if nrows_expected is not None:
            if not in_obj.shape[0] == nrows_expected:
                raise Exception("Error selecting: Expected the first dimension to have %d rows!" % nrows_expected)
        return in_obj[rows, ...]

    if isinstance(obj, dict):
        out_obj = {}
        if isinstance(obj, OrderedDict):
            out_obj = OrderedDict()
        for k in obj:
            out_obj[k] = subset(obj[k])

    elif isinstance(obj, list):
        out_obj = [subset(el) for el in obj]
    else:
        out_obj = subset(obj)

    return out_obj


class ReshapeDna(object):
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

        # Set default that even for a single sample a batch axis is prepended
        self.single_sample_no_batch_axis = False

    def get_seq_len(self):
        return self.seq_len

    def to_standard(self, in_array):
        """
        :param in_array: has to have the sequence samples in the 0th dimension
        :return:
        """
        #
        # if there is no "batch" axis prepended to the array then add it, but remember
        # that as a rule for `from_standard`
        if (len(in_array.shape) - len(self.in_shape)) == 0:
            in_array = in_array[None, ...]
            logger.warn("Prepending missing batch axis to input shape %s." % str(self.in_shape))
            self.single_sample_no_batch_axis = True

        # if there is a "batch" axis prepended to the array then remember that as a rule
        elif in_array.shape[0] == 1:
            self.single_sample_no_batch_axis = False

        #  is there an actual sequence sample axis?
        additional_axis = len(in_array.shape) - len(self.in_shape)
        if (additional_axis != 1) or (in_array.shape[1:] != tuple(self.in_shape)):
            raise Exception("General array mismatch! Given: %s Expecting: %s" % (str(in_array.shape),
                                                                                 "([N]," + str(self.in_shape)[1:]))
        #
        if not self.reshape_needed:
            return in_array
        squeezed = in_array
        #
        # Iterative removal of dummy dimensions has to start from highest dimension
        for d in sorted(self.dummy_dimensions)[::-1]:
            squeezed = np.squeeze(squeezed, axis=d + additional_axis)
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
            assert (self.seq_len_dim == 1)  # Anything else would be weird...
            squeezed = squeezed.swapaxes(one_hot_dim_here, seq_len_dim_here)
        return squeezed

    def from_standard(self, in_array):
        if not self.reshape_needed:
            if (in_array.shape[0] == 1) and self.single_sample_no_batch_axis:
                in_array = in_array[0, ...]
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
                in_array = np.expand_dims(in_array, d + assumed_additional_axis)
        # If single sample and the convention seems to require no prepended batch axis then remove it.
        if (in_array.shape[0] == 1) and self.single_sample_no_batch_axis:
            in_array = in_array[0, ...]
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


class OutputReshaper(object):
    def __init__(self, model_target_schema, group_delim="."):
        self.model_target_schema = model_target_schema
        self.standard_dict_order = None  # This one is used to always produce the same order of outputs for a dict
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

    @staticmethod
    def ensure_2dim(arr):
        # These are fixes to when the model output definition is not strictly fulfilled by the model
        if len(arr.shape) == 1:
            arr = arr[:, None]
        elif (len(arr.shape) == 3) and (arr.shape[2] == 1):
            arr = arr[..., 0]
        return arr

    def flatten(self, ds):
        if isinstance(ds, dict):
            if not isinstance(self.anno, dict):
                raise Exception("Error in model output defintion: Model definition is"
                                "of type %s but predictions are of type %s!" % (str(type(ds)), str(type(self.anno))))
            outputs = []
            labels = []
            for k in self.standard_dict_order:
                arr = self.ensure_2dim(ds[k])
                assert(arr.shape[1] == self.anno[k].shape[0])
                outputs.append(arr)
                labels.append(self.anno[k])
            flat = np.concatenate(outputs, axis=1)
            flat_labels = np.concatenate(labels, axis=0)
        elif isinstance(ds, list):
            if not isinstance(self.anno, list):
                raise Exception("Error in model output defintion: Model definition is"
                                "of type %s but predictions are of type %s!" % (str(type(ds)), str(type(self.anno))))
            assert len(ds) == len(self.anno)
            ds = [self.ensure_2dim(el) for el in ds]
            flat = np.concatenate(ds, axis=1)
            flat_labels = np.concatenate(self.anno, axis=0)
        else:
            flat = self.ensure_2dim(ds)
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
                raise NotImplementedError("Don't know how to deal with multi-dimensional model target %s" % str(arrayschema_obj))
            # if res_shape[0] == 1:
            #    ret = np.array([""])
            # else:
            ret = np.arange(res_shape[0]).astype(np.str)
        return ret


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

# TODO: generalise so that also FORMAT, FILTER and sample identifiers are supported...
def convert_record(input_record, pyvcf_reader):
    """
    Convert a cyvcf2 record into a pyvcf record. The source files should at least be similar in terms of INFO tags.
    FILTER and FORMAT tags might not be handeled correctly at the moment!
    """
    import vcf

    def revert_to_info(info_obj):
        out_str_elms = []
        for el in list(info_obj):
            out_str_elms.append("{0}={1}".format(*el))
        if len(out_str_elms) > 0:
            return pyvcf_reader._parse_info(";".join(out_str_elms))
        else:
            return {}
    #
    info_tag = revert_to_info(input_record.INFO)
    alt = pyvcf_reader._map(pyvcf_reader._parse_alt, input_record.ALT)
    return vcf.model._Record(input_record.CHROM, input_record.POS, input_record.ID,
                             input_record.REF, alt, input_record.QUAL, input_record.FILTER,
                             info_tag, input_record.FORMAT, {})


def default_vcf_id_gen(vcf_record, id_delim=":"):
    return str(vcf_record.CHROM) + id_delim + str(vcf_record.POS) + id_delim + str(vcf_record.REF) + id_delim + str(vcf_record.ALT)


class RegionGenerator(object):
    def __init__(self, model_info_extractor):
        self.seq_length = None
        self.centered_l_offset = None
        self.centered_r_offset = None
        self.model_info_extractor = model_info_extractor

    @abstractmethod
    def __call__(self, variant):
        """single variant instance yielded by vcf_iter
        """
        pass


class SnvCenteredRg(RegionGenerator):
    def __init__(self, model_info_extractor):
        super(SnvCenteredRg, self).__init__(model_info_extractor)
        self.seq_length = model_info_extractor.get_seq_len()
        seq_length_half = int(self.seq_length / 2)
        self.centered_l_offset = seq_length_half - 1
        self.centered_r_offset = seq_length_half + self.seq_length % 2
        #self.centered_l_offset = seq_length_half
        #self.centered_r_offset = seq_length_half + self.seq_length % 2 -1

    def __call__(self, variant_record):
        """single variant instance yielded by vcf_iter
        """
        return {"chrom": [variant_record.CHROM],
                "start": [variant_record.POS - self.centered_l_offset],
                "end": [variant_record.POS + self.centered_r_offset],
                }


class SnvPosRestrictedRg(RegionGenerator):
    def __init__(self, model_info_extractor, pybed_def):
        super(SnvPosRestrictedRg, self).__init__(model_info_extractor)
        self.tabixed = pybed_def.tabix(in_place=False)
        self.seq_length = model_info_extractor.get_seq_len()
        seq_length_half = int(self.seq_length / 2)
        self.centered_l_offset = seq_length_half - 1
        self.centered_r_offset = seq_length_half + self.seq_length % 2

    def __call__(self, variant_record):
        """single variant instance yielded by vcf_iter
        """
        overlap = self.tabixed.tabix_intervals("%s:%d-%d" % (variant_record.CHROM, variant_record.POS, variant_record.POS + 1))
        chroms = []
        starts = []
        ends = []
        for interval in overlap:
            i_s = interval.start + 1
            i_e = interval.end
            if len(interval) < self.seq_length:
                continue

            if len(interval) != self.seq_length:
                centered_se = np.array([(variant_record.POS - self.centered_l_offset), (variant_record.POS + self.centered_r_offset)])
                start_missing = centered_se[0] - i_s  # >=0 if ok
                end_missing = i_e - centered_se[1]  # >=0 if ok
                if start_missing < 0:
                    centered_se -= start_missing  # shift right
                elif end_missing < 0:
                    centered_se += end_missing  # shift left
                assert centered_se[1] - centered_se[0] + 1 == self.seq_length
                assert (i_s <= centered_se[0]) and (i_e >= centered_se[1])
                i_s, i_e = centered_se.tolist()

            chroms.append(variant_record.CHROM)
            starts.append(i_s)
            ends.append(i_e)
        return {"chrom": chroms, "start": starts, "end": ends}


class ModelInfoExtractor(object):
    def __init__(self, model_obj, dataloader_obj):
        self.model = model_obj
        self.dataloader = dataloader_obj
        self.seq_fields = _get_seq_fields(model_obj)
        # Here we really have to go and collect all the possible different input DNA sequences and prepare the correct
        # transformation to standard

        # Collect the different sequence inputs and the corresponfing ranges objects:
        self.seq_input_metadata = {}
        self.seq_input_mutator = {}
        self.seq_input_array_trafo = {}
        for seq_field in self.seq_fields:
            special_type = _get_specialtype(dataloader_obj, seq_field)

            if special_type is None:
                logger.warn("special_type of sequence field '%s' is not set,"
                            "assuming 1-hot encoded DNA sequence." % str(seq_field))

            if (special_type is None) or (special_type == kipoi.components.ArraySpecialType.DNASeq):
                dna_seq_trafo = ReshapeDna(_get_seq_shape(dataloader_obj, seq_field))
                self.seq_input_mutator[seq_field] = OneHotSequenceMutator(dna_seq_trafo)
                self.seq_input_array_trafo[seq_field] = dna_seq_trafo

            if special_type == kipoi.components.ArraySpecialType.DNAStringSeq:
                dna_seq_trafo = ReshapeDnaString(_get_seq_shape(dataloader_obj, seq_field))
                self.seq_input_mutator[seq_field] = DNAStringSequenceMutator(dna_seq_trafo)
                self.seq_input_array_trafo[seq_field] = dna_seq_trafo

            self.seq_input_metadata[seq_field] = _get_metadata_name(dataloader_obj, seq_field)

        # If then where do I have to put my bed file in the command?
        self.exec_files_bed_keys = _get_dl_bed_fields(dataloader_obj)

        self.requires_region_definition = False
        # If there is a field for putting the a postprocessing bed file, then generate the bed file.
        if (self.exec_files_bed_keys is not None) and (len(self.exec_files_bed_keys) != 0):
            self.requires_region_definition = True

        self.seq_length = None
        if self.requires_region_definition:
            # seems to require a bed file definition, so try to assign a sequence length:
            seq_lens = [self.seq_input_array_trafo[seq_field].get_seq_len() for seq_field in self.seq_input_array_trafo]
            seq_len = list(set([el for el in seq_lens if el is not None]))
            if len(seq_len) == 0:
                raise Exception("dataloader.yaml defines postprocessing > args > bed_input, but in model.yaml none of "
                                "the postprocessing > args > seq_input entries defines a sequence length within their "
                                "shape.")
            elif len(seq_len) > 1:
                raise Exception("dataloader.yaml defines postprocessing > args > bed_input, but in model.yaml sequence"
                                "lengths differ in the postprocessing > args > seq_input entries which is inferred "
                                "from the shapes.")
            self.seq_length = seq_len[0]

        self.model_out_annotation = None

        # Get model output annotation:
        if self.model_out_annotation is None:
            if isinstance(model_obj.schema.targets, dict):
                raise Exception("Variant effect prediction with dict(array) model output not implemented!")
            elif isinstance(model_obj.schema.targets, list):
                self.model_out_annotation = np.array([x.name for x in model_obj.schema.targets])
            else:
                if model_obj.schema.targets.column_labels is not None:
                    self.model_out_annotation = np.array(model_obj.schema.targets.column_labels)

        # If no model model output annotation defined,
        if self.model_out_annotation is None:
            self.model_out_annotation = np.array([str(i) for i in range(model_obj.schema.targets.shape[0])])

        # Check if model supports simple rc-testing of input sequences:
        self.use_seq_only_rc = _get_model_use_seq_only_rc(model_obj)

    def get_mutatable_inputs(self):
        return list(self.seq_input_mutator.keys())

    def get_seq_mutator(self, seq_field):
        return self.seq_input_mutator[seq_field]

    def get_seq_metadata(self, seq_field):
        return self.seq_input_metadata[seq_field]

    def get_all_metadata_fields(self):
        return list(set(self.seq_input_metadata.values()))

    def get_seq_len(self):
        return self.seq_length

    def requires_region_definition(self):
        return self.requires_region_definition

    def get_exec_files_bed_keys(self):
        if self.requires_region_definition:
            return self.exec_files_bed_keys

    def get_model_out_annotation(self):
        return self.model_out_annotation


def _get_metadata_name(dataloader, seq_key):
    if isinstance(dataloader.output_schema.inputs, dict):
        ranges_slots = dataloader.output_schema.inputs[seq_key].associated_metadata
    elif isinstance(dataloader.output_schema.inputs, list):
        ranges_slots = [x.associated_metadata for x in dataloader.output_schema.inputs if x.name == seq_key][0]
    else:
        ranges_slots = dataloader.output_schema.inputs.associated_metadata
    # check the ranges slots
    if len(ranges_slots) != 1:
        raise ValueError(
            "Exactly one metadata ranges field must defined for a sequence that has to be used for effect precition.")
    return ranges_slots[0]


def _get_specialtype(dataloader, seq_field):
    if isinstance(dataloader.output_schema.inputs, dict):
        seq_obj = dataloader.output_schema.inputs[seq_field]
    elif isinstance(dataloader.output_schema.inputs, list):
        seq_obj = [x for x in dataloader.output_schema.inputs if x.name == seq_field][0]
    else:
        seq_obj = dataloader.output_schema.inputs
    if hasattr(seq_obj, "special_type"):
        return seq_obj.special_type
    else:
        return None


def _get_seq_fields(model):
    if model.postprocessing.variant_effects is None:
        raise Exception("Model does not support var_effect_prediction")
    else:
        return model.postprocessing.variant_effects.seq_input


def _get_model_use_seq_only_rc(model):
    if model.postprocessing.variant_effects is None:
        return False
    else:
        return model.postprocessing.variant_effects.use_rc


def _get_seq_shape(dataloader, seq_field):
    if isinstance(dataloader.output_schema.inputs, dict):
        orig_shape = dataloader.output_schema.inputs[seq_field].shape
    elif isinstance(dataloader.output_schema.inputs, list):
        orig_shape = [x.shape for x in dataloader.output_schema.inputs if x.name == seq_field][0]
    else:
        orig_shape = dataloader.output_schema.inputs.shape
    return orig_shape


def _get_dl_bed_fields(dataloader):
    if dataloader.postprocessing.variant_effects is None:
        return None
    else:
        return dataloader.postprocessing.variant_effects.bed_input


class ReshapeDnaString(object):
    def __init__(self, input_shape):
        if len(input_shape) == 0:
            self.format_style = "string"
            self.seq_len = None
        elif len(input_shape) == 1 and (input_shape[0] == 1):
            self.format_style = "string_in_vect"
            self.seq_len = None
        elif len(input_shape) == 1 and (input_shape[0] > 1):
            self.format_style = "string_as_vect"
            self.seq_len = input_shape[0]
        else:
            raise Exception("String output definition not recognized in array string converter!")
        self.single_sample_no_batch_axis = False
        self.input_shape = input_shape

    def get_seq_len(self):
        return self.seq_len

    def to_standard(self, arr):
        if len(arr.shape) == len(self.input_shape):
            arr = arr[None, ...]
            logger.warn("Prepending missing batch axis to input shape %s." % str(self.input_shape))
            self.single_sample_no_batch_axis = True
        elif arr.shape[0] == 1:
            self.single_sample_no_batch_axis = False
        if self.format_style == "string":
            return [str(el) for el in arr]
        elif self.format_style == "string_in_vect":
            return [str(el[0]) for el in arr]
        elif self.format_style == "string_as_vect":
            return ["".join(el.tolist()) for el in arr]

    def from_standard(self, arr):
        if self.format_style == "string":
            arr = np.array(arr)
        elif self.format_style == "string_in_vect":
            arr = np.array(arr)[:, None]
        elif self.format_style == "string_as_vect":
            arr = np.array([list(el) for el in arr])
        if (arr.shape[0] == 1) and self.single_sample_no_batch_axis:
            arr = arr[0, ...]
        return arr


class SequenceMutator(object):
    def __call__(self, input_set, preproc_conv, allele, s_dir):
        """
        Process sequence object `input_set` according to information given in the `preproc_conv` dataframe of which
        the column with name set in argument `allele` is used. `s_dir` defines the output sequence direction: 'fwd'
        or 'rc'. The DNA sequence will then be mutated accordingly.
        """
        raise NotImplementedError("This functionality has to be implemented in the specific subclasses")


class OneHotSequenceMutator(SequenceMutator):
    def __init__(self, array_trafo=None):
        # from the model info object guess
        self.array_trafo = array_trafo

    def __call__(self, input_set, preproc_conv, allele, s_dir):
        """
        Process sequence object `input_set` according to information given in the `preproc_conv` dataframe of which
        the column with name set in argument `allele` is used. `s_dir` defines the output sequence direction: 'fwd'
        or 'rc'. The DNA sequence will then be mutated accordingly. `array_trafo` objects are used to bring non-
        (N,seq_len, 4) arrays into the that shape and convert it back.
        """
        # make sure the sequence objects have the correct length (acording to the ranges specifications)
        if self.array_trafo is not None:
            input_set = self.array_trafo.to_standard(input_set)
        # only keep lines that
        preproc_conv_mutate = preproc_conv.query("do_mutate")
        exp_seq_lens = (preproc_conv_mutate["end"] - preproc_conv_mutate["start"] + 1).unique()
        assert exp_seq_lens.shape[0] == 1
        assert input_set.shape[1] == exp_seq_lens[0]
        assert preproc_conv_mutate["strand"].isin(["+", "-", "*", "."]).all()
        # Modify bases according to allele
        get_warnings = allele == "ref"
        ref_warnings = _modify_bases(seq_obj=input_set,
                                     lines=preproc_conv_mutate["pp_line"].values,
                                     pos=preproc_conv_mutate["varpos_rel"].values.astype(np.int),
                                     base=preproc_conv_mutate[allele].str.upper().values,
                                     is_rc=preproc_conv_mutate["strand"].values == "-",
                                     return_ref_warning=get_warnings)
        for ppl in ref_warnings:
            pcl = np.where(preproc_conv["pp_line"].values == ppl)[0][0]
            vstr = "".join([['A', 'C', 'G', 'T', 'N'][x.argmax() if (x.sum() != 0) and
                                                      np.all(np.in1d(x, np.arange(0, 4))) else 4] for x in input_set[pcl, ...]])
            logger.warn("Variant reference allele is not the allele present in sequence for:\n%s\n"
                        "Sequence:\n%s" % (str(preproc_conv.iloc[pcl]), vstr))
        # generate reverse complement if needed
        if s_dir == "rc":
            input_set = input_set[:, ::-1, ::-1]
        if self.array_trafo is not None:
            input_set = self.array_trafo.from_standard(input_set)
        return input_set


class DNAStringSequenceMutator(SequenceMutator):
    def __init__(self, array_trafo=None):
        self.array_trafo = array_trafo

    def __call__(self, input_set, preproc_conv, allele, s_dir):
        """
        Process sequence object `input_set` according to information given in the `preproc_conv` dataframe of which
        the column with name set in argument `allele` is used. `s_dir` defines the output sequence direction: 'fwd'
        or 'rc'. The DNA sequence will then be mutated accordingly.

        The input sequence object must be according to the definition in the dataloader.
        """
        # input_set has to be <list(<str>)> which is achieved by the `array_trafo`.
        if self.array_trafo is not None:
            input_set = self.array_trafo.to_standard(input_set)
        seq_lens = np.unique([len(el) for el in input_set])
        exp_seq_lens = (preproc_conv["end"] - preproc_conv["start"] + 1).unique()
        exp_seq_lens = exp_seq_lens[~np.isnan(exp_seq_lens)]
        assert seq_lens.shape[0] == 1
        assert exp_seq_lens.shape[0] == 1
        assert seq_lens[0] == exp_seq_lens[0]
        strands = preproc_conv.query("do_mutate")["strand"]
        assert strands.isin(["+", "-", "*", "."]).all()
        # Modify bases according to allele.
        output_set = []
        for pcl, l in enumerate(preproc_conv["pp_line"].values):
            if preproc_conv["do_mutate"].values[pcl]:
                output_set.append(_modify_single_string_base(input_set[l],
                                                             pos=int(preproc_conv["varpos_rel"].values[pcl]),
                                                             base=preproc_conv[allele].values[pcl],
                                                             is_rc=preproc_conv["strand"].values[pcl] == "-"))
                if allele == "ref":
                    is_rc = preproc_conv["strand"].values[pcl] == "-"
                    base = preproc_conv[allele].values[pcl]
                    vstr = input_set[l]
                    if is_rc:
                        vstr = rc_str(input_set[l])
                    if vstr[int(preproc_conv["varpos_rel"].values[pcl])].upper() != base.upper():
                        logger.warn("Variant reference allele is not the allele present in sequence for:\n%s\n"
                                    "Sequence:\n%s" % (str(preproc_conv.iloc[pcl]), str(input_set[l])))
            else:
                output_set.append(input_set[l])
        # subset to the lines that have been identified
        if len(output_set) != preproc_conv.shape[0]:
            raise Exception("Mismatch between requested and generated DNA sequences.")
        # generate reverse complement if needed
        if s_dir == "rc":
            output_set = [rc_str(el) for el in output_set]
        if self.array_trafo is not None:
            output_set = self.array_trafo.from_standard(output_set)
        return output_set
