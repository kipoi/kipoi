from __future__ import absolute_import
from __future__ import print_function

import abc
import copy
import logging
import re
from abc import abstractmethod
from collections import OrderedDict
from .dna_reshapers import ReshapeDnaString, ReshapeDna
from .mutators import OneHotSequenceMutator, DNAStringSequenceMutator

import numpy as np

import kipoi

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
    s = re.sub(r"[^\w\.\:\s/]+", '_', s)
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
    __metaclass__ = abc.ABCMeta
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
        if self.seq_length is None:
            raise Exception("SnvCenteredRg cannot be used when model input sequence length is not defined!")
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
        if self.seq_length is None:
            raise Exception("SnvPosRestrictedRg cannot be used when model input sequence length is not defined!")
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

        self.seq_length = None # None means either not requires_region_definition or undefined sequence length
        if self.requires_region_definition:
            # seems to require a bed file definition, so try to assign a sequence length:
            seq_lens = [self.seq_input_array_trafo[seq_field].get_seq_len() for seq_field in self.seq_input_array_trafo]
            seq_len = list(set([el for el in seq_lens]))
            seq_len_noNone = [el for el in seq_len if el is not None]
            if len(seq_len) == 0:
                raise Exception("dataloader.yaml defines postprocessing > args > bed_input, but in model.yaml none of "
                                "the postprocessing > args > seq_input entries defines a sequence length within their "
                                "shape.")
            elif len(seq_len_noNone) > 1:
                raise Exception("dataloader.yaml defines postprocessing > args > bed_input, but in model.yaml sequence"
                                "lengths differ in the postprocessing > args > seq_input entries which is inferred "
                                "from the shapes.")
            if seq_len_noNone != seq_len:
                self.seq_length = None
            else:
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


class VariantLocalisation(object):
    def __init__(self):
        self.obj_keys = ["pp_line", "varpos_rel", "ref", "alt", "start", "end", "id", "do_mutate", "strand"]
        self.dummy_initialisable_keys = ["varpos_rel", "ref", "alt", "start", "end", "id", "strand"]
        self.data = {k: [] for k in self.obj_keys}

    def append_multi(self, seq_key, ranges_input_obj, vcf_records, process_lines, process_ids, process_seq_fields):
        import six
        strand_avail = False
        strand_default = "."
        if ("strand" in ranges_input_obj) and (isinstance(ranges_input_obj["strand"], list) or
                                                   isinstance(ranges_input_obj["strand"], np.ndarray)):
            strand_avail = True

        # If the strand is a single string value rather than a list or numpy array than use that as a
        # default for everything
        if ("strand" in ranges_input_obj) and isinstance(ranges_input_obj["strand"], six.string_types):
            strand_default = ranges_input_obj["strand"]

        # Iterate over all variants
        for i, record in enumerate(vcf_records):
            assert not record.is_indel  # Catch indels, that needs a slightly modified processing
            ranges_input_i = process_lines[i]
            # Initialise the new values as missing values, and as skip for processing
            new_vals = {k: np.nan for k in self.dummy_initialisable_keys}
            new_vals["do_mutate"] = False
            new_vals["pp_line"] = i
            new_vals["id"] = str(process_ids[i])
            new_vals["strand"] = strand_default
            # If the corresponding sequence key should be modified, then calculate the relative variant position
            if seq_key in process_seq_fields[i]:
                pre_new_vals = {}
                pre_new_vals["start"] = ranges_input_obj["start"][ranges_input_i] + 1
                pre_new_vals["end"] = ranges_input_obj["end"][ranges_input_i]
                pre_new_vals["varpos_rel"] = int(record.POS) - pre_new_vals["start"]
                # Check if variant position is valid
                if not ((pre_new_vals["varpos_rel"] < 0) or
                            (pre_new_vals["varpos_rel"] > (pre_new_vals["end"] - pre_new_vals["start"] + 1))):

                    # If variant lies in the region then actually mutate it with the first alternative allele
                    pre_new_vals["do_mutate"] = True
                    pre_new_vals["ref"] = str(record.REF)
                    pre_new_vals["alt"] = str(record.ALT[0])

                    if strand_avail:
                        pre_new_vals["strand"] = ranges_input_obj["strand"][ranges_input_i]

                    # overwrite the nans with actual data now that
                    for k in pre_new_vals:
                        new_vals[k] = pre_new_vals[k]

            for k in new_vals:
                self.data[k].append(new_vals[k])

    def subset_to_mutate(self):
        sel_mutate = [i for i, dm in enumerate(self.data["do_mutate"]) if dm]
        data_subset = {k: [self.data[k][i] for i in sel_mutate] for k in self.data}
        new_obj = self.__class__()
        new_obj.data = data_subset
        return new_obj

    def get_seq_lens(self):
        lens = np.array([end - start +1 for start, end in zip(self.data["start"], self.data["end"])])
        return lens

    def strand_vals_valid(self):
        return all([el in ["+", "-", "*", "."] for el in self.data["strand"]])

    def get(self, item, trafo = None):
        vals = self.data.__getitem__(item)
        if trafo is not None:
            vals = [trafo(el) for el in vals]
        return np.array(vals)

    def __getitem__(self, item):
        return self.get(item)

    def num_entries(self):
        return len(self.data["pp_line"])

    def to_df(self):
        import pandas as pd
        return pd.DataFrame(self.data)




