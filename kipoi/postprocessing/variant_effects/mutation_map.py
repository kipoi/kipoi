import copy
import itertools
import logging
import os
import tempfile

import numpy as np
import pandas as pd
from tqdm import tqdm
from kipoi.utils import cd

from kipoi.postprocessing.variant_effects.utils import select_from_dl_batch, OutputReshaper, default_vcf_id_gen, \
    ModelInfoExtractor, BedWriter, VariantLocalisation
from kipoi.postprocessing.variant_effects.utils.plot import seqlogo_heatmap
from kipoi.postprocessing.variant_effects.utils.scoring_fns import Logit
from kipoi.postprocessing.variant_effects import BedOverlappingRg, SnvCenteredRg, ensure_tabixed_vcf
from .snv_predict import SampleCounter, get_genomicranges_line, merge_intervals, get_variants_in_regions_search_vcf, \
    get_variants_in_regions_sequential_vcf, analyse_model_preds, _overlap_vcf_region

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _generate_records_for_all_regions(regions, ref_seq):
    """
    Generate records for all positions covered by the region
    """
    from vcf.model import _Record, _Substitution
    assert isinstance(regions["chr"], list) or isinstance(regions["chr"], np.ndarray)
    assert isinstance(ref_seq, list)
    assert len(regions["chr"]) == len(ref_seq)
    contained_regions = []
    vcf_records = []
    for i, ref_seq_here in enumerate(ref_seq):
        chrom, start, end = regions["chr"][i], regions["start"][i] + 1, regions["end"][i]
        for pos, ref in zip(range(start, end + 1), ref_seq_here.upper()):
            qual = 0
            filt = []
            info = {}
            fmt = None
            sample_indexes = None
            for alt in ["A", "C", "G", "T"]:
                # skip REF/REF variants - they should always be 0 anyways.
                if ref == alt:
                    continue
                ID = ":".join([chrom, str(pos), ref, alt])
                record = _Record(chrom, pos, ID, ref, [_Substitution(alt)], qual, filt, info, fmt,
                                 sample_indexes)
                vcf_records.append(record)
                contained_regions.append(i)
    #
    return vcf_records, contained_regions


def get_variants_for_all_positions(dl_batch, seq_to_meta, ref_sequences, process_lines_preselection=None):
    """
    Function that generates VCF records for all positions in the input sequence(s). This is done by merging the regions.
    When regions are party overlapping then a variant will be tagged with all sequence-fields that participated in the
    merged region, hence not all input regions might be affected by the variant.
    """
    vcf_records = []  # list of vcf records to use
    process_lines = []  # sample id within batch
    process_seq_fields = []  # sequence fields that should be mutated
    #
    meta_to_seq = {v: [k for k in seq_to_meta if seq_to_meta[k] == v] for v in seq_to_meta.values()}
    all_meta_fields = list(set(seq_to_meta.values()))
    #
    if process_lines_preselection is None:
        num_samples_in_batch = len(dl_batch['metadata'][all_meta_fields[0]]["chr"])
        process_lines_preselection = range(num_samples_in_batch)
    #
    # If we should search for the overlapping VCF lines - for every sample collect all region objects
    # under the assumption that all generated sequences have the same number of samples in a batch:
    for line_id in process_lines_preselection:
        # check is there is more than one metadata_field that is used:

        if len(all_meta_fields) > 1:
            # one region per meta_field
            regions_by_meta = {k: get_genomicranges_line(dl_batch['metadata'][k], line_id)
                               for k in all_meta_fields}
            sequence = {k: ref_sequences[k][line_id] for k in ref_sequences}
            # regions_unif: union across all regions. meta_field_unif_r: meta_fields, has the length of regions_unif
            regions_unif, meta_field_unif_r = merge_intervals(regions_by_meta)
            # get list of merged reference sequences as defined by regions_unif.
            merged_seq = merged_intervals_seq(regions_by_meta, sequence, regions_unif, meta_field_unif_r)
        else:
            # Only one meta_field and only one line hence:
            meta_field_unif_r = [all_meta_fields]
            # get respective reference sequence for the DL batch:
            merged_seq = [ref_sequences[all_meta_fields[0]][line_id]]
            # Only one region:
            regions_unif = get_genomicranges_line(dl_batch['metadata'][all_meta_fields[0]], line_id)
        #
        vcf_records_here, process_lines_rel = _generate_records_for_all_regions(regions_unif, merged_seq)
        #
        for rec, sub_line_id in zip(vcf_records_here, process_lines_rel):
            vcf_records.append(rec)
            process_lines.append(line_id)
            metas = []
            for f in meta_field_unif_r[sub_line_id]:
                metas += meta_to_seq[f]
            process_seq_fields.append(metas)
    return vcf_records, process_lines, process_seq_fields


def merged_intervals_seq(ranges_dict, sequence, regions_unif, meta_field_unif_r):
    """
    ranges_dict: Dict of Genomic ranges objects for all the different metadata slots
    sequence: Sequences stored in a dictionary with metadata slots as keys
    regions_unif: A genomic ranges object of the unified ranges
    meta_field_unif_r: Metadata slot names that are associated with a certain region in `regions_unif`
    seq_to_meta: sequence slot keys to their associated metadata slot key
    """

    all_joint_seqs = []
    for reg_i, mf in enumerate(meta_field_unif_r):
        reg_start = regions_unif["start"][reg_i]
        reg_len = regions_unif["end"][reg_i] - reg_start
        joint_seq = np.empty(reg_len, dtype=str)
        joint_seq[:] = ""
        for mf_here in mf:
            rel_start, rel_end = ranges_dict[mf_here]["start"][0] - reg_start, ranges_dict[mf_here]["end"][
                0] - reg_start
            # When generating the merged sequence make sure the overlapping parts of the sequence match up!
            if np.any(joint_seq[rel_start:rel_end] != ""):
                assert all([a == b for a, b in zip(joint_seq[rel_start:rel_end], sequence[mf_here]) if a != ""])
            joint_seq[rel_start:rel_end] = list(sequence[mf_here])
        all_joint_seqs.append("".join(joint_seq.tolist()))
    return all_joint_seqs


def _overlap_bedtools_region(bedtools_obj, regions):
    """
    Overlap a vcf with regions generated by the dataloader
    The region definition is assumed to be 0-based hence it is converted to 1-based for tabix overlaps!
    Returns VCF records
    """
    assert isinstance(regions["chr"], list) or isinstance(regions["chr"], np.ndarray)
    contained_regions = []
    bed_regions = []
    for i in range(len(regions["chr"])):
        chrom, start, end = regions["chr"][i], regions["start"][i] + 1, regions["end"][i]
        region_str = "{0}:{1}-{2}".format(chrom, start, end)
        bf_regions = bedtools_obj.tabix_intervals(region_str)
        for region in bf_regions:
            bed_regions.append(region)
            contained_regions.append(i)
    #
    return bed_regions, contained_regions


def compress_genomicranges_list(input_list):
    """Convert list of genomicranges objects to a single genomicranges object."""
    # TODO - directly use kipoi.metadata.GenomicRanges.collate(input_list) instead of this function
    assert isinstance(input_list, list)
    out_regions = {k: [] for k in ["chr", "start", "end", "strand"]}
    [[out_regions[k].append(v[k][0]) for k in out_regions] for v in input_list]
    return out_regions


def get_overlapping_bed_regions(dl_batch, seq_to_meta, bedtools_obj):
    """
    Function that overlaps metadata ranges with a bed file.
    Regions are not merged prior to overlap with bedtools_obj!

    Arguments:
        dl_batch: batch coming from the dataloader
        seq_to_meta: dictionary that converts model input names to its associated metadata field.
        bedtools_obj: Tabixed bedtools object of the regions that should be investigated.
    """
    bed_regions = []  # list of vcf records to use
    process_lines = []  # sample id within batch
    process_seq_fields = []  # sequence fields that should be mutated
    #
    meta_to_seq = {v: [k for k in seq_to_meta if seq_to_meta[k] == v] for v in seq_to_meta.values()}
    all_meta_fields = list(set(seq_to_meta.values()))
    #
    num_samples_in_batch = len(dl_batch['metadata'][all_meta_fields[0]]["chr"])
    #
    # If we should search for the overlapping VCF lines - for every sample collect all region objects
    # under the assumption that all generated sequences have the same number of samples in a batch:
    for line_id in range(num_samples_in_batch):
        # check is there is more than one metadata_field that is used:
        if len(all_meta_fields) > 1:
            # As opposed to the VCF handling here don't merge intervals... If two metadata fields are overlapping this
            # then they will be tested independently.
            regions_unif_list = [get_genomicranges_line(dl_batch['metadata'][mf], line_id) for mf in all_meta_fields]
            regions_unif = compress_genomicranges_list(regions_unif_list)
            meta_field_unif_r = [[v] for v in all_meta_fields]
        else:
            # Only one meta_field and only one line hence:
            meta_field_unif_r = [all_meta_fields]
            # Only one region:
            regions_unif = get_genomicranges_line(dl_batch['metadata'][all_meta_fields[0]], line_id)
        #
        bed_regions_here, process_lines_rel = _overlap_bedtools_region(bedtools_obj, regions_unif)
        #
        for reg, sub_line_id in zip(bed_regions_here, process_lines_rel):
            bed_regions.append(reg)
            process_lines.append(line_id)
            metas = []
            for f in meta_field_unif_r[sub_line_id]:
                metas += meta_to_seq[f]
            process_seq_fields.append(metas)
    return bed_regions, process_lines, process_seq_fields


def _generate_seq_sets_mutmap_iter(dl_ouput_schema, dl_batch, seq_to_mut, seq_to_meta,
                                   sample_counter, ref_sequences, bedtools_obj=None, vcf_fh=None,
                                   vcf_id_generator_fn=None, vcf_search_regions=False, generate_rc=True,
                                   batch_size=32):
    all_meta_fields = list(set(seq_to_meta.values()))

    num_samples_in_batch = len(dl_batch['metadata'][all_meta_fields[0]]["chr"])

    metadata_ids = sample_counter.get_ids(num_samples_in_batch)

    if "_id" in dl_batch['metadata']:
        metadata_ids = dl_batch['metadata']['id']
        assert num_samples_in_batch == len(metadata_ids)

    # now get the right region from the vcf:
    # list of vcf records to use: vcf_records
    process_ids = None  # id from genomic ranges metadata: process_lines
    # sample id within batch: process_lines
    # sequence fields that should be mutated: process_seq_fields

    query_bed_regions = None
    query_vcf_records = None
    if vcf_fh is not None:
        # This is from the variant effect prediction function - use this to annotate the output in the end...
        if vcf_search_regions:
            query_vcf_records, query_process_lines, query_process_seq_fields = \
                get_variants_in_regions_search_vcf(dl_batch, seq_to_meta, vcf_fh)
        else:
            # vcf_search_regions == False means: rely completely on the variant id
            # so for every sample assert that all metadata ranges ids agree and then find the entry.
            query_vcf_records, query_process_lines, query_process_seq_fields, query_process_ids = \
                get_variants_in_regions_sequential_vcf(dl_batch, seq_to_meta, vcf_fh, vcf_id_generator_fn)
    elif bedtools_obj is not None:
        query_bed_regions, query_process_lines, query_process_seq_fields = \
            get_overlapping_bed_regions(dl_batch, seq_to_meta, bedtools_obj)
    else:
        # No restrictions are given so process all input lines
        query_process_lines = list(range(num_samples_in_batch))

    # Now generate fake variants for all bases on all positions
    # only the "query_process_lines" selected above should be considered!
    vcf_records, process_lines, process_seq_fields = \
        get_variants_for_all_positions(dl_batch, seq_to_meta, ref_sequences, query_process_lines)

    # short-cut if no sequences are left
    if len(process_lines) == 0:
        raise StopIteration

    if process_ids is None:
        process_ids = []
        for line_id in process_lines:
            process_ids.append(metadata_ids[line_id])

    # Generate a batched output
    real_batch_size = batch_size // 2
    if generate_rc:
        real_batch_size = real_batch_size // 2

    if real_batch_size == 0:
        logger.warn("Batch size too small, resetting it to %d." % (2 + int(generate_rc) * 2))
        real_batch_size = 1

    n_batches = len(process_lines) // real_batch_size
    if len(process_lines) % real_batch_size != 0:
        n_batches += 1

    for batch_i in range(n_batches):
        bs, be = batch_i * real_batch_size, min(((batch_i + 1) * real_batch_size), len(process_lines))
        vcf_records_batch = vcf_records[bs:be]
        process_lines_batch = process_lines[bs:be]
        process_seq_fields_batch = process_seq_fields[bs:be]
        process_ids_batch = process_ids[bs:be]

        # Generate 4 copies of the input set. subset datapoints if needed.
        input_set = {}
        seq_dirs = ["fwd"]
        if generate_rc:
            seq_dirs = ["fwd", "rc"]
        for s_dir, allele in itertools.product(seq_dirs, ["ref", "alt"]):
            k = "%s_%s" % (s_dir, allele)
            ds = dl_batch['inputs']
            all_lines = list(range(num_samples_in_batch))
            if process_lines_batch != all_lines:
                # subset or rearrange elements
                ds = select_from_dl_batch(dl_batch['inputs'], process_lines_batch, num_samples_in_batch)
            input_set[k] = copy.deepcopy(ds)

        # input_set matrices now are in the order required for mutation

        all_mut_seq_keys = list(set(itertools.chain.from_iterable(process_seq_fields_batch)))

        # Start from the sequence inputs mentioned in the model.yaml
        for seq_key in all_mut_seq_keys:
            ranges_input_obj = dl_batch['metadata'][seq_to_meta[seq_key]]
            vl = VariantLocalisation()
            vl.append_multi(seq_key, ranges_input_obj, vcf_records_batch,
                            process_lines_batch, process_ids_batch, process_seq_fields_batch)

            # for the individual sequence input key get the correct sequence mutator callable
            dna_mutator = seq_to_mut[seq_key]

            # Actually modify sequences according to annotation
            # two for loops
            for s_dir, allele in itertools.product(seq_dirs, ["ref", "alt"]):
                k = "%s_%s" % (s_dir, allele)
                if isinstance(dl_ouput_schema.inputs, dict):
                    if seq_key not in input_set[k]:
                        raise Exception("Sequence field %s is missing in DataLoader output!" % seq_key)
                    input_set[k][seq_key] = dna_mutator(input_set[k][seq_key], vl, allele, s_dir)
                elif isinstance(dl_ouput_schema.inputs, list):
                    modified_set = []
                    for seq_el, input_schema_el in zip(input_set[k], dl_ouput_schema.inputs):
                        if input_schema_el.name == seq_key:
                            modified_set.append(dna_mutator(seq_el, vl, allele, s_dir))
                        else:
                            modified_set.append(seq_el)
                    input_set[k] = modified_set
                else:
                    input_set[k] = dna_mutator(input_set[k], vl, allele, s_dir)

        #
        # Reformat so that effect prediction function will get its required inputs
        pred_set = {"ref": input_set["fwd_ref"], "alt": input_set["fwd_alt"]}
        if generate_rc:
            pred_set["ref_rc"] = input_set["rc_ref"]
            pred_set["alt_rc"] = input_set["rc_alt"]
        # pred_set["line_id"] = np.array(process_ids_batch).astype(str) # the process_id does not make sense here
        pred_set["line_id"] = np.array([rec.ID for rec in vcf_records_batch])  # use the id of the variant
        pred_set["vcf_records"] = vcf_records_batch
        pred_set["batch_iters"] = list(range(bs, be))
        pred_set["process_line"] = process_lines_batch
        pred_set["query_bed_regions"] = query_bed_regions
        pred_set["query_vcf_records"] = query_vcf_records
        pred_set["query_process_lines"] = query_process_lines
        yield pred_set


def get_ref_seq_from_seq_set(input_set, seq_to_meta, seq_to_str_converter, dl_ouput_schema_inputs):
    meta_to_seq = {v: [k for k in seq_to_meta if seq_to_meta[k] == v] for v in seq_to_meta.values()}
    all_meta_fields = list(set(seq_to_meta.values()))
    str_seqs = {}

    for meta_field in all_meta_fields:
        seq_key = meta_to_seq[meta_field][0]
        in_set = input_set["inputs"]
        if isinstance(dl_ouput_schema_inputs, dict):
            if seq_key not in in_set:
                raise Exception("Sequence field %s is missing in DataLoader output!" % seq_key)
            seq_data = in_set[seq_key]
        elif isinstance(dl_ouput_schema_inputs, list):
            seq_data = in_set[dl_ouput_schema_inputs.index(seq_key)]
        else:
            seq_data = in_set
        strand_info = input_set["metadata"][meta_field]["strand"]
        if isinstance(strand_info, list):
            strand_info = np.array(strand_info)

        if not isinstance(strand_info, np.ndarray):
            logger.warn("Strand in dataloader batch not defined properly for metadata field: %s. "
                        "Assuming all sequences to be in forward direction!" % meta_field)
            is_rc = np.array([False] * input_set["metadata"][meta_field]["start"].shape[0])
        else:
            is_rc = strand_info == "-"

        str_seqs[meta_field] = seq_to_str_converter[seq_key].to_str(seq_data, is_rc=is_rc)
    return str_seqs


# Structure of mutation_map
# For every DL batch entry across all batches:
#   For every metadata class (seq key):
#       Get the DNA sequence as a String
#       Get the genomic region of the metadata
#       Get all variants that overlap the region and reset to relative coordinates
#       Generate the mapping variants versus effects matrix
#       For every scoring method
#           For every model output
#               Extract the effect score


class MutationMapDataMerger(object):
    def __init__(self, seq_to_meta):
        self.predictions = []
        self.pred_sets = []
        self.ref_seqs = []
        self.batch_metadata_list = []
        self.seq_to_meta = seq_to_meta
        self.meta_to_seq = {v: [k for k in seq_to_meta if seq_to_meta[k] == v] for v in seq_to_meta.values()}
        self.mutation_map = None

    def append(self, predictions, pred_set, ref_seq, batch_metadata):
        # append a new prediction to the set.
        # It is necessary that every batch that is appended is complete in terms of dataloader batches!
        self.predictions.append(predictions)
        self.pred_sets.append(pred_set)
        self.ref_seqs.append(ref_seq)
        self.batch_metadata_list.append(batch_metadata)

    def get_merged_data(self):
        # In general it doesn't make much sense to concatenate the batches. The output shall be as one entry per process_line in pred_sets
        # predictions are stored as dictionary (keys = scoring function) of pandas DFs where each column is a model output.
        # pred_sets contain ["line_id", "vcf_records", "process_line", "query_vcf_records", "query_process_lines"]
        #
        # For every line from the DL
        # mutation_map = [process_line across batches: { model input key: { scoring function key: { Model output label: Object}}}]
        if self.mutation_map is not None:
            return self.mutation_map
        mutation_map = []
        for predictions, pred_set, ref_seq, batch_metadata in zip(self.predictions, self.pred_sets,
                                                                  self.ref_seqs, self.batch_metadata_list):
            for process_line in np.unique(pred_set["process_line"]):
                predictions_rowfilt = pred_set["process_line"] == process_line
                query_vcf_recs = None
                if pred_set["query_vcf_records"] is not None:
                    query_vcf_recs = [rec for rec, pl in zip(pred_set["query_vcf_records"],
                                                             pred_set["query_process_lines"]) if pl == process_line]
                pl_mut_maps = {}
                for metadata_key in ref_seq:
                    metadata_subset = get_genomicranges_line(batch_metadata[metadata_key], process_line)
                    subset_keys = ["chr", "start", "end", "strand"]
                    if not (isinstance(metadata_subset["strand"], list) or
                                isinstance(metadata_subset["strand"], np.ndarray)):
                        subset_keys = ["chr", "start", "end"]
                    metadata_subset_dict = {k: metadata_subset[k][0] for k in subset_keys}
                    if "strand" not in metadata_subset_dict:
                        metadata_subset_dict["strand"] = "*"
                    metadata_chrom = metadata_subset["chr"][0]
                    metadata_start = metadata_subset["start"][0]
                    metadata_end = metadata_subset["end"][0]
                    metadata_seqlen = metadata_end - metadata_start
                    ref_seq_here = ref_seq[metadata_key][process_line]
                    query_vcf_recs_rel = {k: [] for k in ["chr", "pos", "ref", "alt", "id", "varpos_rel"]}
                    if query_vcf_recs is not None:
                        for rec in query_vcf_recs:
                            query_vcf_recs_rel["chr"].append(rec.CHROM)
                            query_vcf_recs_rel["pos"].append(rec.POS)
                            query_vcf_recs_rel["ref"].append(rec.REF)
                            query_vcf_recs_rel["alt"].append(rec.ALT)
                            query_vcf_recs_rel["id"].append(rec.ID)
                            # pos is 1-based, metadata_start is 0 based and the relative position is 0-based, hence -1.
                            query_vcf_recs_rel["varpos_rel"].append(int(rec.POS) - metadata_start - 1)
                    # check that number of variants matches the number of bases *3
                    assert predictions_rowfilt.sum() == metadata_seqlen * 3
                    # generate the ids to get the right order of predictions:
                    correct_order_ids = []
                    # REF/REF was not generated, so those can be appended to the results
                    ref_ref_var_ids = []
                    for pos, ref in zip(range(metadata_start + 1, metadata_end + 1), ref_seq_here.upper()):
                        for alt in ["A", "C", "G", "T"]:
                            ID = ":".join([metadata_chrom, str(pos), ref, alt])
                            correct_order_ids.append(ID)
                            if ref == alt:
                                ref_ref_var_ids.append(ID)
                    metadata_mutmap = {}
                    for scoring_fn in predictions:
                        assert predictions[scoring_fn].shape[0] == predictions_rowfilt.shape[0]
                        metadata_mutmap[scoring_fn] = {}
                        # check if output labels are not unique
                        if len(set(predictions[scoring_fn].columns)) != predictions[scoring_fn].shape[1]:
                            predictions[scoring_fn].columns = ["%s_%d" % (d, i) for i, d in
                                                               enumerate(predictions[scoring_fn].columns)]
                            logger.warn("Model output labels are not unique! appending the column number.")
                        for model_output in predictions[scoring_fn]:
                            mutmap_dict = {}
                            preds = predictions[scoring_fn][model_output].loc[predictions_rowfilt]
                            # append predictions for ref/ref variants:
                            ref_ref_preds = pd.Series(0, name=preds.name, index=ref_ref_var_ids)
                            preds = pd.concat([preds, ref_ref_preds])
                            preds = preds.loc[correct_order_ids].values
                            mutmap_dict["mutation_map"] = np.reshape(preds, (metadata_seqlen, 4)).T
                            mutmap_dict["ovlp_var"] = query_vcf_recs_rel
                            mutmap_dict["ref_seq"] = ref_seq_here
                            mutmap_dict["metadata_region"] = metadata_subset_dict
                            metadata_mutmap[scoring_fn][model_output] = mutmap_dict
                    for seq_key in self.meta_to_seq[metadata_key]:
                        pl_mut_maps[seq_key] = metadata_mutmap
                mutation_map.append(pl_mut_maps)
        self.mutation_map = mutation_map
        return mutation_map

    def save_to_file(self, fname):
        # Best is to use a hdf5 file
        from kipoi.postprocessing.variant_effects.utils.generic import write_hdf5
        return write_hdf5(fname, self.get_merged_data())

    def to_plotter(self):
        return MutationMapPlotter(mutation_map=self.get_merged_data())


class MutationMapPlotter(object):
    def __init__(self, mutation_map=None, fname=None):
        if mutation_map is None and fname is None:
            raise Exception("Either mutation_map or fname for a mutation_map file has to be given")
        if mutation_map is not None:
            self.mutation_map = mutation_map
        elif fname is not None:
            from kipoi.postprocessing.variant_effects.utils.generic import read_hdf5
            self.mutation_map = read_hdf5(fname)

    def save_to_file(self, fname):
        # Best is to use a hdf5 file
        from kipoi.postprocessing.variant_effects.utils.generic import write_hdf5
        return write_hdf5(fname, self.mutation_map)

    def plot_mutmap(self, dl_entry, model_seq_key, scoring_key, model_output, ax=None, show_letter_scale=False,
                    cmap=None, limit_region=None, limit_region_genomic=None, annotation_vcf=None,
                    annotation_variants=None, ignore_stored_var_annotation=False, rc_plot=False,
                    minimum_letter_height=None):
        """
        Generate a mutation map plot
        
        
        dl_entry: index of the region generated by the dataloader. (Integer) 
        model_seq_key: Model input sequence key as defined in model.yaml. e.g. `seq`
        scoring_key: Key given for the scoring method. E.g.: `diff`
        model_output: Model output name for which the mutation map should be plotted
        ax: matplotlib axis object
        show_letter_scale: Display letter scale for seqlogo
        cmap: Colourmap for the heatmap
        limit_region: Limit the plotted region: Tuple of (x_start, x_end) where both x_* are integer in [0,sequence_length)
        limit_region_genomic: Like limit_region but a tuple of genomic positions. Values outside the queried
            regions are ignored. Region definition has to 0-based!
        annotation_vcf: VCF used for additional variant annotation in the plot. Only SNVs will be used.
        annotation_variants: dictionary with key: `chr`, `pos`, `id`, `ref`, `alt` and values are lists of strings,
            except for `pos` which is a list of integers of 1-based variant positions.
        ignore_stored_var_annotation: Ignore annotations that have been stored with the mutation map on generation.
        rc_plot: Reverse-complement plotting
        minimum_letter_height: Require a minimum height of the reference base. proportion of maximum letter height.
            e.g. 0.2
        """

        def append_to_ovlp_var(mm_obj, ovlp_var, pos, id, ref, alt):
            seq_len = mm_obj["end"] - mm_obj["start"]
            varpos_rel = pos - mm_obj["start"] - 1  # variant position is 1-based.
            if (varpos_rel > 0) and (varpos_rel < seq_len):
                ovlp_var['varpos_rel'].append(varpos_rel)
                ovlp_var['id'].append(id)
                ovlp_var['ref'].append(ref)
                ovlp_var['alt'].append(alt)

        from kipoi.external.concise.seqplotting_deps import encodeDNA
        import matplotlib.pyplot as plt
        if cmap is None:
            cmap = plt.cm.bwr
        mm_obj = self.mutation_map[dl_entry][model_seq_key][scoring_key][model_output]

        if (limit_region_genomic is not None) and isinstance(limit_region_genomic, tuple):
            mr_start = mm_obj["metadata_region"]["start"]
            mr_end = mm_obj["metadata_region"]["end"]
            if any([(el < mr_start) or (el > mr_end) for el in list(limit_region_genomic)]):
                raise Exception("`limit_region_genomic` has to lie within: %s" % str([mr_start, mr_end]))
            limit_region = (limit_region_genomic[0] - mr_start, limit_region_genomic[1] - mr_start,)

        if (limit_region is None) or ((limit_region is not None) and (not isinstance(limit_region, tuple))):
            limit_region = (0, mm_obj['mutation_map'].shape[1])

        # subset to defined subset
        mm_matrix = mm_obj['mutation_map'][:, limit_region[0]:limit_region[1]]
        ref_dna_str = mm_obj["ref_seq"]
        if hasattr(mm_obj["ref_seq"], "decode"):
            ref_dna_str = mm_obj["ref_seq"].decode('UTF-8')
        ref_dna_str = ref_dna_str[limit_region[0]:limit_region[1]]
        metadata_region = copy.deepcopy(mm_obj["metadata_region"])
        metadata_region["end"] = metadata_region["start"] + limit_region[1]
        metadata_region["start"] = metadata_region["start"] + limit_region[0]

        if ignore_stored_var_annotation or mm_obj['ovlp_var'] is None:
            ovlp_var = {"varpos_rel": [], "id": [], "ref": [], "alt": []}
        else:
            ovlp_var = copy.deepcopy(mm_obj['ovlp_var'])
            # correct the pre-computed variant overlap:
            ovlp_var["varpos_rel"] = [el - limit_region[0] for el in ovlp_var["varpos_rel"]]

        if (annotation_variants is not None) and isinstance(annotation_variants, dict):
            if not all([k in annotation_variants for k in ["chr", "pos", "id", "ref", "alt"]]):
                raise Exception('`annotation_variants` has to be a dictionary with keys ["chr", '
                                '"pos", "id", "ref", "alt"]')
            num_entries = list(set([len(v) for v in annotation_variants.values()]))
            if len(num_entries) != 1:
                raise Exception('All entried in `annotation_variants` have to have the same length!')
            for i in range(num_entries[0]):
                if metadata_region["chr"].lstrip("chr") == str(annotation_variants["chr"][i]).lstrip("chr"):
                    append_to_ovlp_var(metadata_region, ovlp_var, annotation_variants["pos"][i],
                                       annotation_variants["id"][i], annotation_variants["ref"][i],
                                       [annotation_variants["alt"][i]])

        if annotation_vcf is not None:
            import cyvcf2
            if not os.path.exists(annotation_vcf):
                raise Exception('`annotation_vcf` path doesn\'t exist!')
            # ensure tabixed
            vcf_path = ensure_tabixed_vcf(annotation_vcf)
            vcf_fh = cyvcf2.VCF(vcf_path, "r")
            # overlap records with region
            reg = {k: [v] for k, v in metadata_region.items()}
            vcf_records, contained_regions = _overlap_vcf_region(vcf_fh, reg)
            for rec in vcf_records:
                append_to_ovlp_var(metadata_region, ovlp_var, rec.POS, str(rec.ID), str(rec.REF),
                                   [str(el) for el in rec.ALT])

        # RC if necessary
        if rc_plot:
            mm_matrix = mm_matrix[::-1, ::-1]
            from .utils.mutators import rc_str
            ref_dna_str = rc_str(ref_dna_str)
            seq_len = mm_matrix.shape[1]
            ovlp_var["varpos_rel"] = [seq_len - el - 1 for el in ovlp_var["varpos_rel"]]
            ovlp_var["ref"] = rc_str(ovlp_var["ref"])
            ovlp_var["alt"] = [rc_str(el) for el in ovlp_var["alt"]]

        # Derive letter heights from the mutation scores.
        onehot_refseq = encodeDNA([str(ref_dna_str).upper()])[0, ...]
        mm_non_na = mm_matrix.copy()
        nans = np.isnan(mm_non_na)
        if np.any(nans):
            logger.warn(
                "There were %d missing values in the mutation map which are reset to 0 for plotting!" % nans.sum())
            mm_non_na[nans] = 0
        letter_heights = onehot_refseq * np.abs(mm_non_na.mean(axis=0))[:, None]

        if minimum_letter_height is not None:
            if (minimum_letter_height > 1) or (minimum_letter_height < 0):
                raise Exception("minimum_letter_height has to be a float within [0,1]")
            max_h = letter_heights.max()
            letter_heights = letter_heights * (
            1 - minimum_letter_height) + onehot_refseq * minimum_letter_height * max_h

        return seqlogo_heatmap(letter_heights, mm_non_na, ovlp_var, vocab="DNA", ax=ax,
                               show_letter_scale=show_letter_scale, cmap=cmap, limit_region=None)


class MutationMap(object):
    def __init__(self, model, dataloader, dataloader_args=None, use_dataloader_example_data=False):
        """Generate mutation map

            Prediction of effects of every base at every position of datalaoder input sequences. The regions for which
            the effect scores will be calculated are primarily defined by the dataloader input. 

            # Arguments
                model: A kipoi model handle generated by e.g.: kipoi.get_model()
                dataloader: Dataloader factory generated by e.g.: kipoi.get_dataloader_factory()
                dataloader_args: arguments passed on to the dataloader for sequence generation, arguments
                    mentioned in dataloader.yaml > postprocessing > variant_effects > bed_input will be overwritten
                    by the methods here.
                use_dataloader_example_data: Fill out the missing dataloader arguments with the example values given in 
                    the dataloader.yaml.
            """
        self.model = model
        self.dataloader = dataloader
        self.model_info_extractor = ModelInfoExtractor(model_obj=model, dataloader_obj=dataloader)
        self.exec_files_bed_keys = self.model_info_extractor.get_exec_files_bed_keys()
        if dataloader_args is None:
            self.dataloader_args = {}
        else:
            self.dataloader_args = dataloader_args

        # Copy the missing arguments from the example arguments.
        if use_dataloader_example_data:
            for k in self.dataloader.example_kwargs:
                if k not in self.dataloader_args:
                    self.dataloader_args[k] = self.dataloader.example_kwargs[k]

    def _setup_dataloader_kwargs(self,
                                 vcf_fpath,
                                 bed_fpath,
                                 vcf_to_region,
                                 bed_to_region,
                                 vcf_id_generator_fn):
        """
        Generate the dataloader kwargs. If e.g. the vcf_fpath should only be used for annotation, but not for
        region generation then set vcf_to_region to None.
        """

        import cyvcf2
        from pybedtools import BedTool
        import copy
        # If then where do I have to put my bed file in the command?
        temp_bed3_file = None
        vcf_search_regions = True

        dataloader_args = copy.deepcopy(self.dataloader_args)

        if (vcf_to_region is not None) and (vcf_fpath is not None) and (bed_to_region is not None) and \
                (bed_fpath is not None):
            logger.warn("`vcf_to_region` and `bed_to_region` are both non-None so regions will only be generated "
                        "based on the VCF! Please use either `vcf_to_region` or `bed_to_region` selectively.")

        # If there is a field in the datalaoder arguments for putting the a postprocessing bed file,
        # then generate the bed file.
        if self.exec_files_bed_keys is not None:
            temp_bed3_file = tempfile.mktemp()  # file path of the temp file
            if (vcf_to_region is not None) and (vcf_fpath is not None):
                logger.warn("Using VCF file %s to define the dataloader intervals." % vcf_fpath)
                vcf_search_regions = False
                vcf_fh = cyvcf2.VCF(vcf_fpath, "r")
                with BedWriter(temp_bed3_file) as ofh:
                    for record in vcf_fh:
                        if not record.is_indel:
                            region = vcf_to_region(record)
                            id = vcf_id_generator_fn(record)
                            for chrom, start, end in zip(region["chrom"], region["start"], region["end"]):
                                ofh.append_interval(chrom=chrom, start=start, end=end, id=id)
                vcf_fh.close()
            elif (bed_to_region is not None) and (bed_fpath is not None):
                logger.warn("Using bed file %s to define the dataloader intervals." % bed_fpath)
                bedtools_obj = BedTool(bed_fpath)
                with BedWriter(temp_bed3_file) as ofh:
                    for bed_entry in bedtools_obj:
                        # get all the input regions for a given bed entry
                        in_regions = bed_to_region(bed_entry)
                        for i in range(len(in_regions["chrom"])):
                            ofh.append_interval(**{k: v[i] for k, v in in_regions.items()})
            else:
                logger.warn("Keeping bed file regions defined in `dataloader_args`.")
        else:
            if vcf_to_region is not None:
                logger.warn("`vcf_to_region` will be ignored as it was set, but the dataloader does not define "
                            "a bed_input in dataloader.yaml: "
                            "postprocessing > variant_effects > bed_input.")
            if bed_to_region is not None:
                logger.warn("`bed_to_region` will be ignored as it was set, but the dataloader does not define "
                            "a bed_input in dataloader.yaml: "
                            "postprocessing > variant_effects > bed_input.")

        # If there was a field for dumping the region definition bed file, then use it.
        if (self.exec_files_bed_keys is not None) and ((vcf_search_regions is not None) or (bed_to_region is not None)):
            for k in self.exec_files_bed_keys:
                dataloader_args[k] = temp_bed3_file

        return dataloader_args, temp_bed3_file, vcf_search_regions

    def _generate_mutation_map(self,
                               vcf_fpath=None,
                               bed_fpath=None,
                               batch_size=32,
                               num_workers=0,
                               vcf_to_region=None,
                               bed_to_region=None,
                               vcf_id_generator_fn=default_vcf_id_gen,
                               evaluation_function=analyse_model_preds,
                               evaluation_function_kwargs={'diff_types': {'logit': Logit()}}):
        """Generate mutation map
    
            Prediction of effects of every base at every position of datalaoder input sequences. The regions for which
            the effect scores will be calculated are primarily defined by the dataloader input. Alternatively
            dataloader `bed` file inputs can be overwritten either by sequences centered on variants (see arguments
            `vcf_fpath` and `vcf_to_region`) or bed files (see arguments `bed_fpath` and `bed_to_region`). Effect
            scores are returned as `MutationMapPlotter` object which can be saved to an hdf5 file and used for
            plotting. It is important to mention that the order of the scored sequences is the order in which the
            dataloader has produced data input - intersected with `vcf_fpath` or `bed_fpath` if given.
            Only `vcf_fpath` or `bed_fpath` can be set at once, but neither is required.
    
            # Arguments
                vcf_fpath: If defined only genomic regions overlapping the variants in this VCF will be evaluated.
                    Variants defined here will be highlighted in mutation map plots. Only SNVs will be used. If
                    vcf_to_region is defined and the dataloader accepts bed file input then the dataloader bed input
                    file will be overwritten with regions based on variant positions of this VCF.
                bed_fpath: If defined only genomic regions overlapping regions in this bed file will be evaluated. If
                    bed_to_region is defined and the dataloader accepts bed file input then the dataloader bed input
                    file will be overwritten with regions based this (`bed_fpath`) bed file.
                batch_size: Prediction batch size used for calling the data loader. Each batch will be generated in 4
                    mutated states yielding a system RAM consumption of >= 4x batch size.
                num_workers: Number of parallel workers for loading the dataset.
                vcf_to_region: Callable that generates a regions compatible with dataloader/model from a cyvcf2 record
                bed_to_region: Callable that generates a regions compatible with dataloader/model from a bed entry
                vcf_id_generator_fn: Callable that generates a unique ID from a cyvcf2 record, has to be defined if
                    `vcf_fpath` is set.
                evaluation_function: effect evaluation function. Default is `analyse_model_preds`, which will get
                    arguments defined in `evaluation_function_kwargs`
                evaluation_function_kwargs: kwargs passed on to `evaluation_function`.
    
            # Returns
                A `MutationMapPlotter` object containing variant scores.
            """
        import cyvcf2
        from pybedtools import BedTool
        with cd(self.dataloader.source_dir):
            if (bed_fpath is not None) and (vcf_fpath is not None):
                raise Exception("Can't use both `bed_fpath` and `vcf_fpath`.")

            if (vcf_fpath is not None) and (vcf_id_generator_fn is None):
                raise Exception("If `vcf_fpath` is set then also `vcf_id_generator_fn` has to be defined!")

            dataloader_args, temp_bed3_file, vcf_search_regions = self._setup_dataloader_kwargs(vcf_fpath,
                                                                                                bed_fpath,
                                                                                                vcf_to_region,
                                                                                                bed_to_region,
                                                                                                vcf_id_generator_fn)

            model_out_annotation = self.model_info_extractor.get_model_out_annotation()

            out_reshaper = OutputReshaper(self.model.schema.targets)

            seq_to_mut = self.model_info_extractor.seq_input_mutator
            seq_to_meta = self.model_info_extractor.seq_input_metadata
            seq_to_str_converter = self.model_info_extractor.seq_to_str_converter

            # Open vcf again
            vcf_fh = None
            bed_obj = None
            if vcf_fpath is not None:
                vcf_fh = cyvcf2.VCF(vcf_fpath, "r")
            if bed_fpath is not None:
                bed_obj = BedTool(bed_fpath).tabix()

            # pre-process regions
            keys = set()  # what is that?

            sample_counter = SampleCounter()

            mmdm = MutationMapDataMerger(seq_to_meta)

            # TODO - ignore the un-used params?
            it = self.dataloader(**dataloader_args).batch_iter(batch_size=batch_size,
                                                               num_workers=num_workers)
            for i, batch in enumerate(tqdm(it)):

                # get reference sequence for every line in the batch input
                ref_seq_strs = get_ref_seq_from_seq_set(batch, seq_to_meta, seq_to_str_converter,
                                                        self.dataloader.output_schema.inputs)

                eval_kwargs_iter = _generate_seq_sets_mutmap_iter(self.dataloader.output_schema, batch,
                                                                  seq_to_mut=seq_to_mut,
                                                                  seq_to_meta=seq_to_meta,
                                                                  sample_counter=sample_counter,
                                                                  ref_sequences=ref_seq_strs,
                                                                  bedtools_obj=bed_obj,
                                                                  vcf_fh=vcf_fh,
                                                                  vcf_id_generator_fn=vcf_id_generator_fn,
                                                                  vcf_search_regions=vcf_search_regions,
                                                                  generate_rc=self.model_info_extractor.use_seq_only_rc,
                                                                  batch_size=batch_size)

                dl_batch_res = []
                # Keep the following metadata entries from the from the lines
                eval_kwargs_noseq = {k: [] for k in ["line_id", "vcf_records", "process_line"]}
                query_vcf_records = None
                query_process_lines = None

                for eval_kwargs in tqdm(eval_kwargs_iter):
                    if eval_kwargs is None:
                        # No generated datapoint overlapped any VCF region
                        continue

                    if evaluation_function_kwargs is not None:
                        assert isinstance(evaluation_function_kwargs, dict)
                        for k in evaluation_function_kwargs:
                            eval_kwargs[k] = evaluation_function_kwargs[k]

                    eval_kwargs["out_annotation_all_outputs"] = model_out_annotation
                    res_here = evaluation_function(self.model, output_reshaper=out_reshaper, **eval_kwargs)
                    for k in res_here:
                        keys.add(k)
                        res_here[k].index = eval_kwargs["line_id"]

                    # save predictions
                    dl_batch_res.append(res_here)

                    # save metadata for creating mutation maps
                    [eval_kwargs_noseq[k].extend(eval_kwargs[k]) for k in eval_kwargs_noseq]
                    query_vcf_records = eval_kwargs["query_vcf_records"]
                    query_process_lines = eval_kwargs["query_process_lines"]

                # query vcf entries have to appended at the end of the batch again
                eval_kwargs_noseq["query_vcf_records"] = query_vcf_records
                eval_kwargs_noseq["query_process_lines"] = query_process_lines
                # Concatenate results over batches
                dl_batch_res_concatenated = {}
                for k in keys:
                    dl_batch_res_concatenated[k] = pd.concat(
                        [inner_batch[k] for inner_batch in dl_batch_res if k in inner_batch])

                # Append results and inputs to mutation map
                mmdm.append(dl_batch_res_concatenated, eval_kwargs_noseq, ref_seq_strs, batch["metadata"])

            if vcf_fh is not None:
                vcf_fh.close()

            try:
                if temp_bed3_file is not None:
                    os.unlink(temp_bed3_file)
            except:
                pass

            return mmdm.to_plotter()

    def query_region(self,
                     chrom,
                     start,
                     end,
                     model_seq_length=None,
                     evaluation_function_kwargs={'diff_types': {'logit': Logit()}},
                     **kwargs):
        """Generate mutation map

        Prediction of effects of every base at every position of datalaoder input sequences. The regions for which
        the effect scores will be calculated are primarily defined by the dataloader input. If the dataloader accepts
        `bed` file inputs then this file will be overwritten with regions defined here of length `model_seq_length` or 
        the model input sequence length. If that is not available all datalaoder-generated regions that overlap the
        region defined here will be investigated. Effect scores are returned as `MutationMapPlotter` object which can
        be saved to  an hdf5 file and used for plotting. It is important to mention that the order of the scored
        sequences is the order in which the dataloader has produced data input - intersected with the region defined
        here.

        # Arguments
            chrom: Chrosome of region of interest. Assembly is defined by the dataload arguments.
            start: Start of region of interest. Assembly is defined by the dataload arguments.
            end: End of region of interest. Assembly is defined by the dataload arguments.
            model_seq_length: Optional argument of model sequence length to use if model accepts variable input
            sequence length. 
            evaluation_function_kwargs: kwargs passed on to `evaluation_function`.

        # Returns
            A `MutationMapPlotter` object containing variant scores.
        """
        from pybedtools import BedTool
        bed_to_region = None
        bed_region = BedTool("\t".join(["chr" + chrom.lstrip("chr"), str(start), str(end)]), from_string=True)
        if (self.exec_files_bed_keys is not None):
            bed_to_region = BedOverlappingRg(self.model_info_extractor, seq_length=model_seq_length)
        mmdm = self._generate_mutation_map(bed_fpath=bed_region.fn,
                                           vcf_to_region=None,
                                           bed_to_region=bed_to_region,
                                           evaluation_function_kwargs=evaluation_function_kwargs,
                                           **kwargs)
        return mmdm

    def query_bed(self,
                  bed_fpath,
                  model_seq_length=None,
                  evaluation_function_kwargs={'diff_types': {'logit': Logit()}},
                  **kwargs):
        """Generate mutation map

        Prediction of effects of every base at every position of datalaoder input sequences. The regions for which
        the effect scores will be calculated are primarily defined by the dataloader input. If the dataloader accepts
        `bed` file inputs then this file will be overwritten with regions defined in `bed_fpath` of length
        `model_seq_length` or the model input sequence length. If that is not available all datalaoder-generated
        regions that overlap the region defined here will be investigated. Effect scores are returned as
        `MutationMapPlotter` object which can be saved to  an hdf5 file and used for plotting. It is important to
        mention that the order of the scored sequences is the  order in which the dataloader has produced data
        input - intersected with `bed_fpath`. 

        # Arguments
            bed_fpath: Only genomic regions overlapping regions in this bed file will be evaluated. If
                the dataloader accepts bed file input then the dataloader bed input file will be overwritten with
                regions based this (`bed_fpath`) bed file. Assembly is defined by the dataload arguments.
            model_seq_length: Optional argument of model sequence length to use if model accepts variable input
                sequence length. 
            evaluation_function_kwargs: kwargs passed on to `evaluation_function`.

        # Returns
            A `MutationMapPlotter` object containing variant scores.
        """
        bed_to_region = None
        if (self.exec_files_bed_keys is not None):
            bed_to_region = BedOverlappingRg(self.model_info_extractor, seq_length=model_seq_length)
        mmdm = self._generate_mutation_map(vcf_fpath=None,
                                           bed_fpath=bed_fpath,
                                           vcf_to_region=None,
                                           bed_to_region=bed_to_region,
                                           vcf_id_generator_fn=default_vcf_id_gen,
                                           evaluation_function_kwargs=evaluation_function_kwargs,
                                           **kwargs)
        return mmdm

    def query_vcf(self,
                  vcf_fpath,
                  model_seq_length=None,
                  evaluation_function_kwargs={'diff_types': {'logit': Logit()}},
                  var_centered_regions=True,
                  **kwargs):
        """Generate mutation map

        Prediction of effects of every base at every position of datalaoder input sequences. The regions for which
        the effect scores will be calculated are primarily defined by the dataloader input. If the dataloader accepts
        `bed` file inputs then this file will be overwritten with regions generaten from the SNVs in `vcf_fpath`in a
        variant-centered fashion. Sequence length is defined by `model_seq_length` or the model input sequence length.
        If the datalaoder does not have a `bed` file input all datalaoder-generated regions that overlap SNVs
        here will be investigated. Effect scores are returned as `MutationMapPlotter` object which can be saved to 
        an hdf5 file and used for plotting. It is important to mention that the order of the scored sequences is the 
        order in which the dataloader has produced data input - intersected with `vcf_fpath`. 

        # Arguments
            vcf_fpath: Only genomic regions overlapping the variants in this VCF will be evaluated.
                Variants defined here will be highlighted in mutation map plots. Only SNVs will be used. If
                vcf_to_region is defined and the dataloader accepts bed file input then the dataloader bed input
                file will be overwritten with regions based on variant positions of this VCF.
            model_seq_length: Optional argument of model sequence length to use if model accepts variable input
                sequence length.
            var_centered_regions: Generate variant-centered regions if the model accepts that. If a custom
                `vcf_to_region` should be used then this can be set explicitly in the kwargs.
            evaluation_function_kwargs: kwargs passed on to `evaluation_function`.

        # Returns
            A `MutationMapPlotter` object containing variant scores.
        """
        vcf_to_region = None
        if var_centered_regions and (self.exec_files_bed_keys is not None):
            vcf_to_region = SnvCenteredRg(self.model_info_extractor, seq_length=model_seq_length)
        if "vcf_to_region" in kwargs:
            vcf_to_region = kwargs["vcf_to_region"]
        mmdm = self._generate_mutation_map(vcf_fpath=vcf_fpath,
                                           bed_fpath=None,
                                           vcf_to_region=vcf_to_region,
                                           bed_to_region=None,
                                           vcf_id_generator_fn=default_vcf_id_gen,
                                           evaluation_function=analyse_model_preds,
                                           evaluation_function_kwargs=evaluation_function_kwargs,
                                           **kwargs)
        return mmdm


def _generate_mutation_map(model,
                           dataloader,
                           vcf_fpath=None,
                           bed_fpath=None,
                           batch_size=32,
                           num_workers=0,
                           dataloader_args=None,
                           vcf_to_region=None,
                           bed_to_region=None,
                           vcf_id_generator_fn=default_vcf_id_gen,
                           evaluation_function=analyse_model_preds,
                           evaluation_function_kwargs={'diff_types': {'logit': Logit()}},
                           use_dataloader_example_data=False,
                           ):
    """Generate mutation map

    Prediction of effects of every base at every position of datalaoder input sequences. The regions for which
    the effect scores will be calculated are primarily defined by the dataloader input. Alternatively
    dataloader `bed` file inputs can be overwritten either by sequences centered on variants (see arguments
    `vcf_fpath` and `vcf_to_region`) or bed files (see arguments `bed_fpath` and `bed_to_region`). Effect
    scores are returned as `MutationMapPlotter` object which can be saved to an hdf5 file and used for
    plotting. It is important to mention that the order of the scored sequences is the order in which the
    dataloader has produced data input - intersected with `vcf_fpath` or `bed_fpath` if given.
    Only `vcf_fpath` or `bed_fpath` can be set at once, but neither is required.

    # Arguments
        model: A kipoi model handle generated by e.g.: kipoi.get_model()
        dataloader: Dataloader factory generated by e.g.: kipoi.get_dataloader_factory()
        vcf_fpath: If defined only genomic regions overlapping the variants in this VCF will be evaluated.
            Variants defined here will be highlighted in mutation map plots. Only SNVs will be used. If
            vcf_to_region is defined and the dataloader accepts bed file input then the dataloader bed input
            file will be overwritten with regions based on variant positions of this VCF.
        bed_fpath: If defined only genomic regions overlapping regions in this bed file will be evaluated. If
            bed_to_region is defined and the dataloader accepts bed file input then the dataloader bed input
            file will be overwritten with regions based this (`bed_fpath`) bed file.
        batch_size: Prediction batch size used for calling the data loader. Each batch will be generated in 4
            mutated states yielding a system RAM consumption of >= 4x batch size.
        num_workers: Number of parallel workers for loading the dataset.
        dataloader_args: arguments passed on to the dataloader for sequence generation, arguments
            mentioned in dataloader.yaml > postprocessing > variant_effects > bed_input will be overwritten
            by the methods here.
        vcf_to_region: Callable that generates a regions compatible with dataloader/model from a cyvcf2 record
        bed_to_region: Callable that generates a regions compatible with dataloader/model from a bed entry
        vcf_id_generator_fn: Callable that generates a unique ID from a cyvcf2 record, has to be defined if
            `vcf_fpath` is set.
        evaluation_function: effect evaluation function. Default is `analyse_model_preds`, which will get
            arguments defined in `evaluation_function_kwargs`
        evaluation_function_kwargs: kwargs passed on to `evaluation_function`.
        use_dataloader_example_data: Fill out the missing dataloader arguments with the example values given in the
            dataloader.yaml.

    # Returns
        A `MutationMapPlotter` object containing variant scores.
    """
    mm = MutationMap(model, dataloader, dataloader_args, use_dataloader_example_data)
    return mm._generate_mutation_map(vcf_fpath=vcf_fpath,
                                     bed_fpath=bed_fpath,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     vcf_to_region=vcf_to_region,
                                     bed_to_region=bed_to_region,
                                     vcf_id_generator_fn=vcf_id_generator_fn,
                                     evaluation_function=evaluation_function,
                                     evaluation_function_kwargs=evaluation_function_kwargs)
