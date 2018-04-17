import sys
import copy
import warnings
import pytest
from kipoi.pipeline import install_model_requirements
from kipoi.utils import cd
import kipoi
from kipoi.postprocessing.variant_effects import Logit, Diff, DeepSEA_effect, analyse_model_preds
import cyvcf2
from kipoi.metadata import GenomicRanges
import pandas as pd
import numpy as np
from kipoi.postprocessing.variant_effects.utils import OneHotSeqExtractor, StrSeqExtractor
import h5py
import os

warnings.filterwarnings('ignore')
from kipoi.postprocessing.variant_effects import mutation_map as mm
from kipoi.postprocessing.variant_effects import snv_predict as sp


INSTALL_REQ = False


class DummyModelInfo(object):
    def __init__(self, seq_length):
        self.seq_length = seq_length
    def get_seq_len(self):
        return self.seq_length

class dummy_container(object):
    pass

class Dummy_internval:
    def __init__(self):
        self.storage = {"chrom": [], "start": [], "end": [], "id": []}
    def append_interval(self, **kwargs):
        for k in kwargs:
            self.storage[k].append(kwargs[k])

def compare_rec(a,b):
    if isinstance(a, dict):
        for k in a:
            compare_rec(a[k],b[k])
    elif isinstance(a, list):
        [compare_rec(a_1, b_1) for a_1, b_1 in zip(a,b)]
    elif isinstance(a, np.ndarray):
        assert np.all(a == b)
    else:
        assert a==b

def test_generate_records_for_all_regions():
    regions = {"chr": ["chr22"] * 2, "start": [21541589, 30630701], "end": [21541953, 30631065], "strand": ["*"] * 2}
    seqs = ["A"*364, "C"*364]
    vcf_records, contained_regions = mm._generate_records_for_all_regions(regions, ref_seq=seqs)
    for i, [start, end] in enumerate(zip(regions["start"], regions["end"])):
        ret_sel = np.array(contained_regions)==i
        assert ret_sel.sum() == (end-start)*3
        for i2 in np.where(ret_sel)[0]:
            assert  (vcf_records[i2].POS >= start) and (vcf_records[i2].POS <= end)


def test__overlap_bedtools_region():
    from pybedtools import BedTool
    bobj = BedTool("chr22 10 20" , from_string=True).tabix()
    starts = np.array([10, 15, 20])
    ends = starts+10
    ins = [True, True, False]
    for start, end, isin in zip(starts, ends, ins):
        regions = dict(start=[start], end=[end], chr = ["chr22"])
        bed_regions, contained_regions = kipoi.postprocessing.variant_effects.mutation_map._overlap_bedtools_region(bobj, regions)
        if isin:
            assert len(bed_regions) == 1
            assert contained_regions == [0]
        else:
            assert len(bed_regions) == 0
            assert len(bed_regions) == 0


def test_get_overlapping_bed_regions():
    from pybedtools import BedTool
    bobj = BedTool("chr22 21541588 21541689\nchr22 30630220 30630702", from_string=True).tabix()
    ints1 = {"chr": ["chr22"] * 2, "start": [21541589, 30630701], "end": [21541953, 36702138], "strand": ["*"] * 2}
    ints2 = {"chr": ["chr22"] * 2, "start": [30630219, 30630220], "end": [30630222, 30630222], "strand": ["*"] * 2}
    model_input = {"metadata": {"gr_a": ints1, "gr_b": ints1, "gr_c": ints2}}
    seq_to_meta = {"seq_a": "gr_a", "seq_a2": "gr_a", "seq_b": "gr_b", "seq_c": "gr_c"}
    bed_entries, process_lines, process_seq_fields = kipoi.postprocessing.variant_effects.mutation_map.get_overlapping_bed_regions(model_input, seq_to_meta, bobj)
    assert process_lines == [0, 0, 0, 1]
    expected = [['seq_b'], ['seq_a', 'seq_a2'], ['seq_c'], ['seq_c']]
    for i, ind in enumerate(process_lines):
        expected_subset = [el for ind2, el in zip(process_lines, expected) if ind2 == ind]
        assert any([set(process_seq_fields[i]) == set(el2) for el2 in expected_subset])

def test_get_variants_for_all_positions():
    ints1 = {"chr": ["chr22"] * 2, "start": [21541589, 30630701], "end": [21541953, 30631065], "strand": ["*"] * 2}
    ints2 = {"chr": ["chr22"] * 2, "start": [30630219, 30630820], "end": [30630222, 30630822], "strand": ["*"] * 2}
    seqs1 = ["A" * 364, "C" * 364]
    seqs2 = ["A" * 3, "C" * 2]
    model_input = {"metadata": {"gr_a": ints1, "gr_b": ints1, "gr_c": ints2}}
    ref_seqs = {"gr_a": seqs1, "gr_b": seqs1, "gr_c": seqs2}
    seq_to_meta = {"seq_a": "gr_a", "seq_a2": "gr_a", "seq_b": "gr_b", "seq_c": "gr_c"}
    vcf_records, process_lines, process_seq_fields = kipoi.postprocessing.variant_effects.mutation_map.get_variants_for_all_positions(
        model_input, seq_to_meta, ref_seqs)
    # Check that every position has been ticked.
    ints1_ticked = [[], []]
    ints2_ticked = [[], []]
    for vcf_rec, pl, psf in zip(vcf_records, process_lines, process_seq_fields):
        if "seq_c" in psf:
            ints2_ticked[pl].append(vcf_rec.POS)
        if "seq_a" in psf:
            assert all([el in psf for el in ["seq_b", "seq_a2"]])
            ints1_ticked[pl].append(vcf_rec.POS)
    # check that all variants are gerenated for the given regions
    for i, [start, end] in enumerate(zip(ints1["start"], ints1["end"])):
        assert np.all(np.in1d(np.arange(start + 1, end + 1), ints1_ticked[i]))
    # check that all variants are gerenated for the given regions
    for i, [start, end] in enumerate(zip(ints2["start"], ints2["end"])):
        assert np.all(np.in1d(np.arange(start + 1, end + 1), ints2_ticked[i]))



def _write_regions_from_vcf(vcf_iter, vcf_id_generator_fn, int_write_fn, region_generator):
    for record in vcf_iter:
        if not record.is_indel:
            region = region_generator(record)
            id = vcf_id_generator_fn(record)
            for chrom, start, end in zip(region["chrom"], region["start"], region["end"]):
                int_write_fn(chrom=chrom, start=start, end=end, id=id)

### greatly simplify this one:
# Check that batching works
# check that query_vcf_records and query_process_lines is always the same for all batches and that it is complete
def test__generate_seq_sets_mutmap_iter():
    from pybedtools import BedTool
    model_dir = "examples/rbp/"
    vcf_sub_path = "example_files/variants.vcf"
    vcf_path = model_dir + vcf_sub_path
    vcf_path = kipoi.postprocessing.variant_effects.ensure_tabixed_vcf(vcf_path)
    # for any given input type: list, dict and np.array return 4 identical sets, except for mutated bases on one position
    seq_len = 101
    model_info_extractor = DummyModelInfo(seq_len)
    for num_seqs in [1, 5]:
        empty_seq_input = np.zeros((num_seqs, seq_len, 4))
        empty_seq_input[:, :, 0] = 1  # All As
        empty_other_input = np.zeros((num_seqs, seq_len, 4)) - 10
        #
        relv_seq_keys = ["seq"]
        #
        vcf_fh = cyvcf2.VCF(vcf_path)
        regions = Dummy_internval()
        #
        model_info_extractor.seq_length = seq_len
        region_generator = kipoi.postprocessing.variant_effects.utils.generic.SnvCenteredRg(model_info_extractor)
        _write_regions_from_vcf(vcf_fh, kipoi.postprocessing.variant_effects.utils.generic.default_vcf_id_gen,
                                regions.append_interval, region_generator)
        #
        vcf_fh.close()
        annotated_regions = pd.DataFrame(regions.storage).iloc[:num_seqs, :]
        #
        gr_meta = {
            "ranges": GenomicRanges(annotated_regions["chrom"].values, annotated_regions["start"].values - 1,
                                    annotated_regions["end"].values,
                                    annotated_regions["id"].values)}
        #
        dict_meta = {
            "ranges": {"chr": annotated_regions["chrom"].values, "start": annotated_regions["start"].values - 1,
                       "end": annotated_regions["end"].values,
                       "id": annotated_regions["id"].values}}
        #
        ref_seqs = {
            "ranges": ["A" * (annotated_regions["end"].values[i] - annotated_regions["start"].values[i]) for i in
                       range(annotated_regions.shape[0])]}
        #
        meta_data_options = [gr_meta, dict_meta]
        #
        seq_to_mut = {"seq": kipoi.postprocessing.variant_effects.utils.generic.OneHotSequenceMutator()}
        seq_to_meta = {"seq": "ranges"}
        n_qseq = annotated_regions.shape[0]
        for batch_size in [4, 8]:
            for meta_data in meta_data_options:
                for vcf_search_regions in [False, True]:
                    # Test the dict case:
                    dataloader = dummy_container()
                    dataloader.output_schema = dummy_container()
                    seq_container = dummy_container()
                    seq_container.associated_metadata = ["ranges"]
                    dataloader.output_schema.inputs = {"seq": seq_container, "other_input": None}
                    inputs = {"seq": copy.deepcopy(empty_seq_input[:n_qseq, ...]),
                              "other_input": copy.deepcopy(empty_other_input[:n_qseq, ...])}
                    inputs_2nd_copy = copy.deepcopy(inputs)
                    #
                    model_input = {"inputs": inputs, "metadata": meta_data}
                    vcf_fh = cyvcf2.VCF(vcf_path, "r")
                    # relv_seq_keys, dataloader, model_input, vcf_fh, vcf_id_generator_fn, array_trafo=None
                    sample_counter = sp.SampleCounter()
                    eval_kwargs_iter = mm._generate_seq_sets_mutmap_iter(dataloader.output_schema, model_input,
                                                                      seq_to_mut=seq_to_mut,
                                                                      seq_to_meta=seq_to_meta,
                                                                      sample_counter=sample_counter,
                                                                      ref_sequences=ref_seqs,
                                                                      vcf_fh=vcf_fh,
                                                                      vcf_id_generator_fn=kipoi.postprocessing.variant_effects.utils.generic.default_vcf_id_gen,
                                                                      vcf_search_regions=vcf_search_regions,
                                                                      generate_rc=True,
                                                                      batch_size=batch_size)
                    for ss_batch in eval_kwargs_iter:
                        assert (len(ss_batch['vcf_records']) == batch_size // 4)
                        assert (len(ss_batch['query_vcf_records']) == num_seqs)
                        req_cols = ['alt', 'ref_rc', 'ref', 'alt_rc']
                        assert np.all(np.in1d(req_cols, list(ss_batch.keys())))
                        for k in req_cols:
                            for k2 in inputs:
                                assert (k2 in ss_batch[k])
                                if k2 not in relv_seq_keys:
                                    assert np.all(ss_batch[k][k2] == inputs_2nd_copy[k2][ss_batch['process_line'], ...])
                                else:
                                    # Assuming modification of matrices works as desired - see its own unit test
                                    # Assuming 1-hot coding with background as 0
                                    if k.endswith("fwd"):
                                        assert np.sum(ss_batch[k][k2] != inputs_2nd_copy[k2][
                                            ss_batch['process_line'], ...]) == 2 * n_qseq
                        #
                        for k in ["ref", "alt"]:
                            for k2 in relv_seq_keys:
                                assert np.all(ss_batch[k][k2] == ss_batch[k + "_rc"][k2][:, ::-1, ::-1])
                    vcf_fh.close()
        ## now just also test whether things work for using bed file and using neither bed nor bed file inputs
        dataloader = dummy_container()
        dataloader.output_schema = dummy_container()
        seq_container = dummy_container()
        seq_container.associated_metadata = ["ranges"]
        dataloader.output_schema.inputs = {"seq": seq_container, "other_input": None}
        inputs = {"seq": copy.deepcopy(empty_seq_input[:n_qseq, ...]),
                  "other_input": copy.deepcopy(empty_other_input[:n_qseq, ...])}
        inputs_2nd_copy = copy.deepcopy(inputs)
        #
        model_input = {"inputs": inputs, "metadata": gr_meta}
        vcf_fh = cyvcf2.VCF(vcf_path, "r")
        # relv_seq_keys, dataloader, model_input, vcf_fh, vcf_id_generator_fn, array_trafo=None
        sample_counter = sp.SampleCounter()
        batch_size = 4
        eval_kwargs_iter = mm._generate_seq_sets_mutmap_iter(dataloader.output_schema, model_input,
                                                             seq_to_mut=seq_to_mut,
                                                             seq_to_meta=seq_to_meta,
                                                             sample_counter=sample_counter,
                                                             ref_sequences=ref_seqs,
                                                             generate_rc=True,
                                                             batch_size=batch_size)
        for ss_batch in eval_kwargs_iter:
            assert (len(ss_batch['vcf_records']) == batch_size // 4)
            assert ss_batch['query_vcf_records'] is None
            req_cols = ['alt', 'ref_rc', 'ref', 'alt_rc']
            assert np.all(np.in1d(req_cols, list(ss_batch.keys())))
        # using bed input
        bed_obj = BedTool("chr22 %d %d" % (annotated_regions["start"].values[0] - 1, annotated_regions["end"].values[0]),
                          from_string=True).tabix()
        eval_kwargs_iter = mm._generate_seq_sets_mutmap_iter(dataloader.output_schema, model_input,
                                                             seq_to_mut=seq_to_mut,
                                                             seq_to_meta=seq_to_meta,
                                                             bedtools_obj=bed_obj,
                                                             sample_counter=sample_counter,
                                                             ref_sequences=ref_seqs,
                                                             generate_rc=True,
                                                             batch_size=batch_size)
        for ss_batch in eval_kwargs_iter:
            assert (len(ss_batch['vcf_records']) == batch_size // 4)
            assert ss_batch['query_vcf_records'] is None
            assert all([el == 0 for el in ss_batch["process_line"]])


def test_OneHotSeqExtractor():
    BASES = ["A", "C", "G", "T"]
    def onehot(seq):
        X = np.zeros((len(seq), len(BASES)))
        for i, char in enumerate(seq):
            X[i, BASES.index(char.upper())] = 1
        return X
    seqs = ["ACTGTCTATA"]*2
    one_hot = np.array([onehot(seq) for seq in seqs])
    one_hot[1, ...] = one_hot[1,::-1, ::-1]
    extr = OneHotSeqExtractor()
    seqs_conv = extr.to_str(one_hot, is_rc=[False, True])
    assert (seqs_conv == seqs)

def test_StrSeqExtractor():
    seqs = ["ACTGTCTATA"]*2
    extr = StrSeqExtractor()
    seqs_conv = extr.to_str(["ACTGTCTATA", "TATAGACAGT"], is_rc=[False, True])
    assert (seqs_conv == seqs)

def test_merged_intervals_seq():
    regions_unif = {"chr": ["chr1"] * 2, "start": [0, 20], "end": [6, 24], "strand": ["+"] * 2}
    ranges_dict = {"ranges1": {"chr": ["chr1"], "start": [0], "end": [4], "strand": ["+"]},
                   "ranges2": {"chr": ["chr1"], "start": [2], "end": [6], "strand": ["+"]},
                   "ranges3": {"chr": ["chr1"], "start": [20], "end": [24], "strand": ["+"]}}
    sequence = {"ranges1": "ACGT", "ranges2": "GTGA", "ranges3": "ACGT"}
    expected_out = ["ACGTGA", "ACGT"]
    sequence_bad = {"ranges1": "ACGT", "ranges2": "GAGA", "ranges3": "ACGT"}
    meta_field_unif_r = [["ranges1", "ranges2"], ["ranges3"]]
    out = mm.merged_intervals_seq(ranges_dict, sequence, regions_unif, meta_field_unif_r)
    assert expected_out == out
    with pytest.raises(Exception):
        mm.merged_intervals_seq(ranges_dict, sequence_bad, regions_unif, meta_field_unif_r)





def test_MutationMapDataMerger():
    if sys.version_info[0] == 2:
        pytest.skip("Skip")
    model_dir = "examples/rbp/"
    vcf_sub_path = "example_files/variants.vcf"
    vcf_path = model_dir + vcf_sub_path
    vcf_path = kipoi.postprocessing.variant_effects.ensure_tabixed_vcf(vcf_path)
    seq_len = 10
    model_info_extractor = DummyModelInfo(seq_len)
    model_info_extractor.seq_length = seq_len
    region_generator = kipoi.postprocessing.variant_effects.utils.generic.SnvCenteredRg(model_info_extractor)
    vcf_fh = cyvcf2.VCF(vcf_path)
    regions = Dummy_internval()
    _write_regions_from_vcf(vcf_fh, kipoi.postprocessing.variant_effects.utils.generic.default_vcf_id_gen,
                            regions.append_interval, region_generator)
    #
    vcf_fh.close()
    annotated_regions = pd.DataFrame(regions.storage)
    num_seqs = annotated_regions.shape[0]
    query_process_lines = list(range(num_seqs))
    vcf_fh = cyvcf2.VCF(vcf_path)
    query_vcf_records = [rec for rec in vcf_fh if
                         kipoi.postprocessing.variant_effects.utils.generic.default_vcf_id_gen(rec)
                         in annotated_regions["id"].tolist()]
    #
    gr_meta = {
        "ranges": GenomicRanges(annotated_regions["chrom"].values, annotated_regions["start"].values - 1,
                                annotated_regions["end"].values,
                                annotated_regions["id"].values,
                                ["*"]*num_seqs)}
    #
    rseq = ["A" * (annotated_regions["end"].values[i] - annotated_regions["start"].values[i]+1) for i in
            range(annotated_regions.shape[0])]
    ref_seqs = {
        "ranges": rseq}
    #
    seq_to_meta = {"seq": "ranges"}
    # "query_vcf_records", "query_process_lines"
    pred_proto_idxs = []
    process_line = []
    for i in range(annotated_regions.shape[0]):
        for pos, ref in zip(range(annotated_regions["start"].values[i], annotated_regions["end"].values[i] + 1),
                            rseq[i]):
            for alt in ["A", "C", "G", "T"]:
                if ref == alt:
                    continue
                ID = ":".join(["chr22", str(pos), ref.upper(), alt])
                pred_proto_idxs.append(ID)
                process_line += [i]

    model_outputs = ["out1", "out2"]
    pred_proto = pd.DataFrame(np.zeros((num_seqs * seq_len * 3, 2)), columns=["out1", "out2"], index=pred_proto_idxs)
    #
    predictions = {"DIFF": pred_proto, "PRED2": pred_proto}
    #
    pred_set = {"query_process_lines": query_process_lines, "query_vcf_records": query_vcf_records,
                "process_line": process_line}
    mmdm = mm.MutationMapDataMerger(seq_to_meta)
    mmdm.append(predictions, pred_set, ref_seqs, gr_meta)
    merged_data = mmdm.get_merged_data()
    assert len(merged_data) == num_seqs
    for i, md in enumerate(merged_data):
        for k in md:
            assert k in list(seq_to_meta.keys())
            for scr in md[k]:
                assert scr in list(predictions.keys())
                for model_output in md[k][scr]:
                    mm_obj = md[k][scr][model_output]
                    assert model_output in model_outputs
                    exp_entries = ['ovlp_var', 'mutation_map', 'ref_seq', 'metadata_region']
                    assert all([k in exp_entries for k in mm_obj])
                    assert len(mm_obj) == len(exp_entries)
                    # This only works when als ref/ref mutations are taken into account
                    #retval = np.reshape(pred_proto[model_output].loc[np.array(process_line)==i], (seq_len, 4)).T
                    #assert np.all(mm_obj['mutation_map'] == retval)
                    assert mm_obj['ref_seq'] == rseq[i]
                    assert mm_obj['ovlp_var']['varpos_rel'][0] == seq_len//2 -1 
                    assert all([mm_obj['metadata_region'][k] == gr_meta["ranges"][k][i]
                                for k in mm_obj['metadata_region']])



def test_mutation_map():
    if sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    # Take the rbp model
    model_dir = "examples/rbp/"
    if INSTALL_REQ:
        install_model_requirements(model_dir, "dir", and_dataloaders=True)

    model = kipoi.get_model(model_dir, source="dir")
    # The preprocessor
    Dataloader = kipoi.get_dataloader_factory(model_dir, source="dir")
    #
    dataloader_arguments = {
        "fasta_file": "example_files/hg38_chr22.fa",
        "preproc_transformer": "dataloader_files/encodeSplines.pkl",
        "gtf_file": "example_files/gencode_v25_chr22.gtf.pkl.gz",
    }
    #
    # Run the actual predictions
    vcf_path = "example_files/first_variant.vcf"
    #
    model_info = kipoi.postprocessing.variant_effects.ModelInfoExtractor(model, Dataloader)
    vcf_to_region = kipoi.postprocessing.variant_effects.SnvCenteredRg(model_info)
    mdmm = mm._generate_mutation_map(model, Dataloader, vcf_path, dataloader_args=dataloader_arguments,
                                     evaluation_function=analyse_model_preds, batch_size=32,
                                     vcf_to_region=vcf_to_region,
                                     evaluation_function_kwargs={'diff_types': {'diff': Diff("mean")}})
    with cd(model.source_dir):
        mdmm.save_to_file("example_files/first_variant_mm_totest.hdf5")
        fh = h5py.File("example_files/first_variant_mm.hdf5", "r")
        reference = mm.recursive_h5_mutmap_reader(fh)
        reference = [reference["_list_%d" % i] for i in range(len(reference))]
        fh.close()
        fh = h5py.File("example_files/first_variant_mm.hdf5", "r")
        obs = mm.recursive_h5_mutmap_reader(fh)
        obs = [obs["_list_%d" % i] for i in range(len(obs))]
        fh.close()
        compare_rec(reference[0], obs[0])
        os.unlink("example_files/first_variant_mm_totest.hdf5")
