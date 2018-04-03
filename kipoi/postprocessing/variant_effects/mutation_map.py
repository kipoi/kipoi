import copy
import itertools
import logging
import os
import tempfile
import h5py

import numpy as np
import pandas as pd
import six
from tqdm import tqdm
from vcf.model import _Record, _Substitution
import matplotlib.pyplot as plt

from kipoi.postprocessing.variant_effects.utils.plot import seqlogo_heatmap
from kipoi.postprocessing.variant_effects.utils.scoring_fns import Logit
from kipoi.postprocessing.variant_effects.utils import select_from_dl_batch, OutputReshaper, default_vcf_id_gen, \
    ModelInfoExtractor, BedWriter, VariantLocalisation
from .snv_predict import SampleCounter, get_genomicranges_line, merge_intervals, get_variants_in_regions_search_vcf,\
    get_variants_in_regions_sequential_vcf, analyse_model_preds

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())



def _generate_records_for_all_regions(regions, ref_seq):
    """
    Generate records for all positions covered by the region
    """
    assert isinstance(regions["chr"], list) or isinstance(regions["chr"], np.ndarray)
    assert isinstance(ref_seq, list)
    assert len(regions["chr"]) == len(ref_seq)
    contained_regions = []
    vcf_records = []
    for i, ref_seq_here in enumerate(ref_seq):
        chrom, start, end = regions["chr"][i], regions["start"][i] + 1, regions["end"][i]
        for pos, ref in zip(range(start, end+1), ref_seq_here):
            qual = 0
            filt = []
            info = {}
            fmt = None
            sample_indexes = None
            for alt in ["A", "C", "G", "T"]:
                ID = ":".join([chrom, str(pos), ref.upper(), alt])
                record = _Record(chrom, pos, ID, ref.upper(), [_Substitution(alt)], qual, filt, info, fmt,
                                 sample_indexes)
                vcf_records.append(record)
                contained_regions.append(i)
    #
    return vcf_records, contained_regions



def get_variants_for_all_positions(dl_batch, seq_to_meta, ref_sequences, process_lines_preselection = None):
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
    # ranges_dict: Dict of Genomic ranges objects for all the different metadata slots
    # sequence: Sequences stored in a dictionary with metadata slots as keys 
    # regions_unif: A genomic ranges object of the unified ranges
    # meta_field_unif_r: Metadata slot names that are associated with a certain region in `regions_unif`
    # seq_to_meta: sequence slot keys to their associated metadata slot key 
    """

    all_joint_seqs = []
    for reg_i, mf in enumerate(meta_field_unif_r):
        reg_start = regions_unif["start"][reg_i]
        reg_len = regions_unif["end"][reg_i] - reg_start
        joint_seq = np.empty(reg_len, dtype=str)
        joint_seq[:] = ""
        for mf_here in mf:
            rel_start, rel_end = ranges_dict[mf_here]["start"][0] - reg_start, ranges_dict[mf_here]["end"][0] - reg_start
            # When generating the merged sequence make sure the overlapping parts of the sequence match up!
            if np.any(joint_seq[rel_start:rel_end] != ""):
                assert all([a == b for a, b in zip(joint_seq[rel_start:rel_end], sequence[mf_here]) if a != ""])
            joint_seq[rel_start:rel_end] = list(sequence[mf_here])
        all_joint_seqs.append("".join(joint_seq.tolist()))
    return all_joint_seqs





def _generate_seq_sets_mutmap_iter(dl_ouput_schema, dl_batch, vcf_fh, vcf_id_generator_fn, seq_to_mut, seq_to_meta,
                       sample_counter, ref_sequences, vcf_search_regions=False, generate_rc=True, batch_size = 32):
    """
        Perform in-silico mutagenesis on what the dataloader has returned.  

        This function has to convert the DNA regions in the model input according to ref, alt, fwd, rc and
        return a dictionary of which the keys are compliant with evaluation_function arguments

        DataLoaders that implement fwd and rc sequence output *__at once__* are not treated in any special way.

        Arguments:
        `dataloader`: dataloader object
        `dl_batch`: model input as generated by the datalaoder
        `vcf_fh`: cyvcf2 file handle
        `vcf_id_generator_fn`: function that generates ids for VCF records
        `seq_to_mut`: dictionary that contains DNAMutator classes with seq_fields as keys
        `seq_to_meta`: dictionary that contains Metadata key names with seq_fields as keys
        `vcf_search_regions`: if `False` assume that the regions are labelled and only test variants/region combinations for
        which the label fits. If `True` find all variants overlapping with all regions and test all.
        `generate_rc`: generate also reverse complement sequences. Only makes sense if supported by model.
        """

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

    # This is from the variant effect prediction function - use this to annotate the output in the end...
    if vcf_search_regions:
        query_vcf_records, query_process_lines, query_process_seq_fields = \
            get_variants_in_regions_search_vcf(dl_batch,seq_to_meta, vcf_fh)
    else:
        # vcf_search_regions == False means: rely completely on the variant id
        # so for every sample assert that all metadata ranges ids agree and then find the entry.
        query_vcf_records, query_process_lines, query_process_seq_fields, query_process_ids = \
            get_variants_in_regions_sequential_vcf(dl_batch, seq_to_meta, vcf_fh, vcf_id_generator_fn)

    # Now generate fake variants for all bases on all positions
    # only the "query_process_lines" selected above should be considered!
    vcf_records, process_lines, process_seq_fields =\
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
        logger.warn("Batch size too small, resetting it to %d."%(2+int(generate_rc)*2))
        real_batch_size = 1

    n_batches = len(process_lines)//real_batch_size
    if len(process_lines) % real_batch_size != 0:
        n_batches += 1


    for batch_i in range(n_batches):
        bs, be = batch_i*real_batch_size, min(((batch_i+1)*real_batch_size), len(process_lines))
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
        #pred_set["line_id"] = np.array(process_ids_batch).astype(str) # the process_id does not make sense here
        pred_set["line_id"] = np.array([rec.ID for rec in vcf_records_batch]) # use the id of the variant
        pred_set["vcf_records"] = vcf_records_batch
        pred_set["batch_iters"] = list(range(bs, be))
        pred_set["process_line"] = process_lines_batch
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
                        "Assuming all sequences to be in forward direction!"%meta_field)
            is_rc = np.array([False]*input_set["metadata"][meta_field]["start"].shape[0])
        else:
            is_rc = strand_info == "-"

        str_seqs[meta_field] = seq_to_str_converter[seq_key].to_str(seq_data, is_rc = is_rc)
    return str_seqs



### Clear definition of what is actually needed:
# For every DL batch entry across all batches:
#   For every metadata class (seq key!):
#       Get the DNA sequence as a String -> OK
#       Get the genomic region of the metadata -> OK
#       Get all variants that overlap the region and reset to relative coordinates -> Done
#       Generate the mapping variants versus effects matrix -> Done
#       For every socring method
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
                query_vcf_recs = [rec for rec, pl in zip(pred_set["query_vcf_records"],
                                                         pred_set["query_process_lines"]) if pl == process_line]
                pl_mut_maps = {}
                for metadata_key in ref_seq:
                    metadata_subset = get_genomicranges_line(batch_metadata[metadata_key], process_line)
                    subset_keys = ["chr", "start", "end", "strand"]
                    if not (isinstance(metadata_subset["strand"], list)
                            or isinstance(metadata_subset["strand"], np.ndarray)):
                        subset_keys = ["chr", "start", "end"]
                    metadata_subset_dict = {k: metadata_subset[k][0] for k in subset_keys}
                    metadata_chrom = metadata_subset["chr"][0]
                    metadata_start = metadata_subset["start"][0]
                    metadata_end = metadata_subset["end"][0]
                    metadata_seqlen = metadata_end-metadata_start
                    ref_seq_here = ref_seq[metadata_key][process_line]
                    query_vcf_recs_rel = {k: [] for k in ["chr", "pos", "ref", "alt", "id", "varpos_rel"]}
                    for rec in query_vcf_recs:
                        query_vcf_recs_rel["chr"].append(rec.CHROM)
                        query_vcf_recs_rel["pos"].append(rec.POS)
                        query_vcf_recs_rel["ref"].append(rec.REF)
                        query_vcf_recs_rel["alt"].append(rec.ALT)
                        query_vcf_recs_rel["id"].append(rec.ID)
                        query_vcf_recs_rel["varpos_rel"].append(int(rec.POS) - metadata_start)
                    # check that number of variants matches the number of bases *4
                    assert predictions_rowfilt.sum() == metadata_seqlen *4
                    # generate the ids to get the right order of predictions:
                    correct_order_ids = []
                    for pos, ref in zip(range(metadata_start+1, metadata_end+1), ref_seq_here):
                        for alt in ["A", "C", "G", "T"]:
                            ID = ":".join([metadata_chrom, str(pos), ref.upper(), alt])
                            correct_order_ids.append(ID)
                    metadata_mutmap = {}
                    for scoring_fn in predictions:
                        assert predictions[scoring_fn].shape[0] == predictions_rowfilt.shape[0]
                        metadata_mutmap[scoring_fn] = {}
                        # check if output labels are not unique
                        if len(set(predictions[scoring_fn].columns)) != predictions[scoring_fn].shape[1]:
                            predictions[scoring_fn].columns = ["%s_%d"%(d, i) for i, d in enumerate(predictions[scoring_fn].columns)]
                            logger.warn("Model output labels are not unique! appending the column number.")
                        for model_output in predictions[scoring_fn]:
                            mutmap_dict = {}
                            preds = predictions[scoring_fn][model_output].loc[predictions_rowfilt].loc[correct_order_ids].values
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
        res = self.get_merged_data()
        res = {"_list_%d"%i:v for i, v in enumerate(res)}
        ofh = h5py.File(fname, "w")
        recursive_h5_mutmap_writer(res, ofh)
        ofh.close()


    def from_file(self, fname):
        fh = h5py.File(fname, "r")
        recovered = recursive_h5_mutmap_reader(fh)
        recovered = [recovered["_list_%d"%i] for i in range(len(recovered))]
        self.mutation_map = recovered
        fh.close()



class MutationMapDrawer(object):
    def __init__(self, mutation_map=None, fname=None):
        if mutation_map is None and fname is None:
            raise Exception("Either mutation_map or fname for a mutation_map file has to be given")
        if mutation_map is not None:
            self.mutation_map = mutation_map
        elif fname is not None:
            with h5py.File(fname, "r") as ifh:
                mutation_map = recursive_h5_mutmap_reader(ifh)
                mutation_map = [mutation_map["_list_%d" % i] for i in range(len(mutation_map))]
                self.mutation_map = mutation_map


    def draw_mutmap(self, dl_entry, model_seq_key, scoring_key, model_output, ax=None, show_letter_scale = False,
                    cmap = plt.cm.bwr, limit_region = None):
        from .utils.seqplotting_deps import encodeDNA
        mm_obj = self.mutation_map[dl_entry][model_seq_key][scoring_key][model_output]

        # Derive letter heights from the mutation scores.
        letter_heights = encodeDNA([mm_obj["ref_seq"]])[0, ...]
        # letter_heights = letter_heights * np.abs(mm_obj['mutation_map'].sum(axis=0))[:,None]
        letter_heights = letter_heights * np.abs(mm_obj['mutation_map'].mean(axis=0))[:, None]

        return seqlogo_heatmap(letter_heights, mm_obj['mutation_map'], mm_obj['ovlp_var'], vocab="DNA", ax = ax,
                               show_letter_scale = show_letter_scale, cmap=cmap, limit_region =limit_region)




def recursive_h5_mutmap_writer(objs, handle, path =""):
    for key in objs.keys():
        if isinstance(objs[key], dict):
            g = handle.create_group(key)
            recursive_h5_mutmap_writer(objs[key], g, path =path + "/" + key)
        else:
            if isinstance(objs[key], list) or isinstance(objs[key], np.ndarray):
                el = np.array(objs[key])
                if "U" in el.dtype.str:
                    el = el.astype("S")
                handle.create_dataset(name=path + "/" + key, data=el, chunks=True, compression='gzip')
            else:
                el = objs[key]
                if isinstance(el, six.string_types):
                    el = str(el)
                handle.create_dataset(name=path + "/" + key, data=el)

def recursive_h5_mutmap_reader(handle):
    objs = {}
    for key in handle.keys():
        if isinstance(handle[key], h5py.Group):
            objs[key] = recursive_h5_mutmap_reader(handle[key])
        else:
            if isinstance(handle[key], h5py.Dataset):
                el = handle[key].value
                if isinstance(el, np.ndarray):
                    if "S" in el.dtype.str:
                        el = el.astype(str)
                objs[key] = el
    return objs







def mutation_map(model,
                 dataloader,
                 vcf_fpath,
                 batch_size,
                 num_workers=0,
                 dataloader_args=None,
                 vcf_to_region=None,
                 vcf_id_generator_fn=default_vcf_id_gen,
                 evaluation_function=analyse_model_preds,
                 evaluation_function_kwargs={'diff_types': {'logit': Logit()}},
                 use_dataloader_example_data=False,
                 ):
    """Predict the effect of SNVs

            Prediction of effects of SNV based on a VCF. If desired the VCF can be stored with the predicted values as
            annotation. For a detailed description of the requirements in the yaml files please take a look at
            kipoi/nbs/variant_effect_prediction.ipynb.

            # Arguments
                model: A kipoi model handle generated by e.g.: kipoi.get_model()
                dataloader: Dataloader factory generated by e.g.: kipoi.get_dataloader_factory()
                vcf_fpath: Path of the VCF defining the positions that shall be assessed. Only SNVs will be tested.
                batch_size: Prediction batch size used for calling the data loader. Each batch will be generated in 4
                    mutated states yielding a system RAM consumption of >= 4x batch size.
                num_workers: Number of parallel workers for loading the dataset.
                dataloader_args: arguments passed on to the dataloader for sequence generation, arguments
                    mentioned in dataloader.yaml > postprocessing > variant_effects > bed_input will be overwritten
                    by the methods here.
                vcf_to_region: Callable that generates a region compatible with dataloader/model from a cyvcf2 record
                vcf_id_generator_fn: Callable that generates a unique ID from a cyvcf2 record
                evaluation_function: effect evaluation function. Default is `analyse_model_preds`, which will get
                    arguments defined in `evaluation_function_kwargs`
                evaluation_function_kwargs: kwargs passed on to `evaluation_function`.
                sync_pred_writer: Single writer or list of writer objects like instances of `VcfWriter`. This object
                    will be called after effect prediction of a batch is done.
                use_dataloader_example_data: Fill out the missing dataloader arguments with the example values given in the
                    dataloader.yaml.
                return_predictions: Return all variant effect predictions as a dictionary. Setting this to False will
                    help maintain a low memory profile and is faster as it avoids concatenating batches after prediction.
                generated_seq_writer: Single writer or list of writer objects like instances of `SyncHdf5SeqWriter`.
                    This object will be called after the DNA sequence sets have been generated. If this parameter is
                    not None, no prediction will be performed and only DNA sequence will be written!! This is relevant
                    if you want to use the `predict_snvs` to generate appropriate input DNA sequences for your model.

            # Returns
                If return_predictions: Dictionary which contains a pandas DataFrame containing the calculated values
                    for each model output (target) column VCF SNV line. If return_predictions == False, returns None.
            """
    import cyvcf2
    model_info_extractor = ModelInfoExtractor(model_obj=model, dataloader_obj=dataloader)

    # If then where do I have to put my bed file in the command?

    exec_files_bed_keys = model_info_extractor.get_exec_files_bed_keys()
    temp_bed3_file = None

    vcf_search_regions = True

    # If there is a field for putting the a postprocessing bed file, then generate the bed file.
    if exec_files_bed_keys is not None:
        if vcf_to_region is not None:
            vcf_search_regions = False

            temp_bed3_file = tempfile.mktemp()  # file path of the temp file

            vcf_fh = cyvcf2.VCF(vcf_fpath, "r")

            with BedWriter(temp_bed3_file) as ofh:
                for record in vcf_fh:
                    if not record.is_indel:
                        region = vcf_to_region(record)
                        id = vcf_id_generator_fn(record)
                        for chrom, start, end in zip(region["chrom"], region["start"], region["end"]):
                            ofh.append_interval(chrom=chrom, start=start, end=end, id=id)

            vcf_fh.close()
    else:
        if vcf_to_region is not None:
            logger.warn("`vcf_to_region` will be ignored as it was set, but the dataloader does not define "
                        "a bed_input in dataloader.yaml: "
                        "postprocessing > variant_effects > bed_input.")
    # Assemble the paths for executing the dataloader
    if dataloader_args is None:
        dataloader_args = {}

    # Copy the missing arguments from the example arguments.
    if use_dataloader_example_data:
        for k in dataloader.example_kwargs:
            if k not in dataloader_args:
                dataloader_args[k] = dataloader.example_kwargs[k]

    # If there was a field for dumping the region definition bed file, then use it.
    if (exec_files_bed_keys is not None) and (not vcf_search_regions):
        for k in exec_files_bed_keys:
            dataloader_args[k] = temp_bed3_file

    model_out_annotation = model_info_extractor.get_model_out_annotation()

    out_reshaper = OutputReshaper(model.schema.targets)


    it = dataloader(**dataloader_args).batch_iter(batch_size=batch_size,
                                                  num_workers=num_workers)

    seq_to_mut = model_info_extractor.seq_input_mutator
    seq_to_meta = model_info_extractor.seq_input_metadata
    seq_to_str_converter = model_info_extractor.seq_to_str_converter

    # Open vcf again
    vcf_fh = cyvcf2.VCF(vcf_fpath, "r")

    # pre-process regions
    keys = set()  # what is that?

    sample_counter = SampleCounter()

    mmdm = MutationMapDataMerger(seq_to_meta)


    for i, batch in enumerate(tqdm(it)):

        # get reference sequence for every line in the batch input
        ref_seq_strs = get_ref_seq_from_seq_set(batch, seq_to_meta, seq_to_str_converter,
                                                dataloader.output_schema.inputs)

        eval_kwargs_iter = _generate_seq_sets_mutmap_iter(dataloader.output_schema, batch, vcf_fh, vcf_id_generator_fn,
                                         seq_to_mut=seq_to_mut, seq_to_meta=seq_to_meta,
                                         sample_counter=sample_counter, vcf_search_regions=vcf_search_regions,
                                         generate_rc=model_info_extractor.use_seq_only_rc,
                                         batch_size=batch_size, ref_sequences=ref_seq_strs)
        dl_batch_res = []
        # Keep the following metadata entries from the from the lines
        eval_kwargs_noseq = {k:[] for k in ["line_id", "vcf_records", "process_line"]}
        query_vcf_records = None
        query_process_lines = None

        for eval_kwargs in eval_kwargs_iter:
            if eval_kwargs is None:
                # No generated datapoint overlapped any VCF region
                continue

            if evaluation_function_kwargs is not None:
                assert isinstance(evaluation_function_kwargs, dict)
                for k in evaluation_function_kwargs:
                    eval_kwargs[k] = evaluation_function_kwargs[k]

            eval_kwargs["out_annotation_all_outputs"] = model_out_annotation

            res_here = evaluation_function(model, output_reshaper=out_reshaper, **eval_kwargs)
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
            dl_batch_res_concatenated[k] = pd.concat([inner_batch[k] for inner_batch in dl_batch_res if k in inner_batch])

        # Append results and inputs to mutation map
        mmdm.append(dl_batch_res_concatenated, eval_kwargs_noseq, ref_seq_strs, batch["metadata"])
        #res.extend(dl_batch_res)
        # now all the results from the dl-batch could be concatenated and summarised with the corresponding: pred_set["query_vcf_records"] and pred_set["query_process_lines"]


    vcf_fh.close()

    try:
        if temp_bed3_file is not None:
            os.unlink(temp_bed3_file)
    except:
        pass


    return mmdm


