import kipoi
import kipoi.postprocessing.utils.generic
import kipoi.postprocessing.utils.io
import kipoi.postprocessing.variant_effects as ve
import kipoi.postprocessing.snv_predict as sp
import numpy as np
import pytest
import sys
from kipoi.pipeline import install_model_requirements
import warnings
import filecmp
import config
import os
import copy
from kipoi.utils import cd
import pandas as pd
import tempfile
from kipoi.metadata import GenomicRanges
import kipoi
from kipoi.postprocessing.variant_effects import Logit, Diff, DeepSEA_effect, Rc_merging_pred_analysis, analyse_model_preds, _prepare_regions
import numpy as np
from scipy.special import logit
import cyvcf2
import pybedtools as pb

warnings.filterwarnings('ignore')


from kipoi.components import ArraySchema, ModelSchema
from related import from_yaml
from kipoi.postprocessing.utils.generic import Output_reshaper

CLS = ArraySchema
MS = ModelSchema

RES={}
RES["2darray_NoLab"] = np.zeros((50, 2))
RES["2darray_Lab"] = np.zeros((50, 2))
RES["list1D_NoLab"] = [np.zeros((50, 1)), np.zeros((50, 1))]
RES["list1D_Lab"] = [np.zeros((50, 1)), np.zeros((50, 1))]
RES["listMixed_NoLab"] = [np.zeros((50, 2)), np.zeros((50, 1))]
RES["listMixed_Lab"] = [np.zeros((50, 2)), np.zeros((50, 1))]
RES["dictMixed_NoLab"] = {"A":np.zeros((50, 2)), "B":np.zeros((50, 1))}
RES["dictMixed_Lab"] = {"A":np.zeros((50, 2)), "B":np.zeros((50, 1))}

RES_OUT_SHAPES = {}
RES_OUT_SHAPES["2darray_NoLab"] = 2
RES_OUT_SHAPES["2darray_Lab"] = 2
RES_OUT_SHAPES["list1D_NoLab"] = 2
RES_OUT_SHAPES["list1D_Lab"] = 2
RES_OUT_SHAPES["listMixed_NoLab"] = 3
RES_OUT_SHAPES["listMixed_Lab"] = 3
RES_OUT_SHAPES["dictMixed_NoLab"] = 3
RES_OUT_SHAPES["dictMixed_Lab"] = 3

RES_OUT_LABELS = {'dictMixed_Lab': ['A.blablabla', 'A.blaSecond', 'B.blaThird'],
                  'list1D_Lab': ['A.blablabla', 'B.blaSecond'], 'listMixed_NoLab':
                      ['0.0', '0.1', '1.0'], '2darray_Lab': ['rbp_prb', 'second'],
                  'dictMixed_NoLab': ['B.0', 'A.0', 'A.1'], 'list1D_NoLab': ['0.0', '1.0'],
                  '2darray_NoLab': ['0', '1'], 'listMixed_Lab':
                      ['A.blablabla', 'A.blaSecond', 'B.blaThird']}

YAMLS = {}
YAMLS["2darray_Lab"] = """
inputs:
  A:
    shape: (101, 4)
    doc: abjhdbajd
targets:
    shape: (2, )
    doc: Predicted binding strength
    name: A
    column_labels:
        - rbp_prb
        - second"""

YAMLS["2darray_NoLab"] = """
inputs:
  A:
    shape: (101, 4)
    doc: abjhdbajd
targets:
    shape: (2, )
    doc: Predicted binding strength"""

YAMLS["list1D_NoLab"] = """
inputs:
  A:
    shape: (101, 4)
    doc: abjhdbajd
targets:
  - shape: (1, )
    doc: Predicted binding strength
  - shape: (1, )
    doc: Predicted binding strength
    """
YAMLS["list1D_Lab"] = """
inputs:
  A:
    shape: (101, 4)
    doc: abjhdbajd
targets:
  - shape: (1, )
    name: A 
    doc: Predicted binding strength
    column_labels:
      - blablabla
  - shape: (1, )
    name: B
    doc: Predicted binding strength
    column_labels:
      - blaSecond
    """

YAMLS["listMixed_Lab"] = """
inputs:
  A:
    shape: (101, 4)
    doc: abjhdbajd
targets:
  - shape: (2, )
    name: A
    doc: Predicted binding strength
    column_labels:
      - blablabla
      - blaSecond
  - shape: (1, )
    name: B
    doc: Predicted binding strength
    column_labels:
      - blaThird
    """

YAMLS["listMixed_NoLab"] = """
inputs:
  A:
    shape: (101, 4)
    doc: abjhdbajd
targets:
  - shape: (2, )
    doc: Predicted binding strength
  - shape: (1, )
    doc: Predicted binding strength
    """

YAMLS["dictMixed_Lab"] = """
inputs:
  A:
    shape: (101, 4)
    doc: abjhdbajd
targets:
  A:
    shape: (2, )
    doc: Predicted binding strength
    column_labels:
      - blablabla
      - blaSecond
  B:
    shape: (1, )
    doc: Predicted binding strength
    column_labels:
      - blaThird
    """

YAMLS["dictMixed_NoLab"] = """
inputs:
  A:
    shape: (101, 4)
    doc: abjhdbajd
targets:
  B:
    shape: (1, )
    doc: Predicted binding strength
  A:
    shape: (2, )
    doc: Predicted binding strength
    """


# TODO: We still need a way to get the model output annotation from somewhere...
# TODO: which other arguments should we use for variant effect predictions?
# Only viable model at the moment is rbp, so not offering to test anything else
# INSTALL_REQ = True
INSTALL_REQ = config.install_req

# TODO - check if you are on travis or not regarding the --install-req flag
INSTALL_REQ = True
# INSTALL_REQ = False



class dummy_container(object):
    pass


def test_ism():
    # Here we should have a simple dummy model, at the moment tested in test_var_eff_pred
    pass

def test__annotate_vcf():
    # This is tested in test_var_eff_pred
    pass

def test__get_seq_len():
    assert (kipoi.postprocessing.utils.generic._get_seq_len([np.array([111])]) == (1,))
    assert (kipoi.postprocessing.utils.generic._get_seq_len((np.array([111]))) == (1,))
    assert (kipoi.postprocessing.utils.generic._get_seq_len({"a": np.array([111]), "b": np.array([111])}) == (1,))
    assert (kipoi.postprocessing.utils.generic._get_seq_len(np.array([111])) == (1,))


def compare_vcfs(fpath1, fpath2):
    fh1 = cyvcf2.VCF(fpath1)
    fh2 = cyvcf2.VCF(fpath2)
    for rec1, rec2 in zip(fh1, fh2):
        i1 = dict(rec1.INFO)
        i2 = dict(rec2.INFO)
        for k in i1:
            min_round = min(len(i1[k]) - i1[k].index("."), len(i2[k]) - i2[k].index("."))
            assert np.round(float(i1[k]), min_round) == np.round(float(i2[k]), min_round)
    fh2.close()
    fh1.close()

def test__vcf_to_regions():
    model_dir = "examples/rbp/"
    vcf_path = "example_files/variants.vcf"
    lct = 0
    with open(model_dir + vcf_path, "r") as ifh:
        for l in ifh:
            if not l.startswith("#"):
                if (len(l.split("\t")[3]) == 1) and (len(l.split("\t")[4]) == 1):
                    lct +=1
    for seq_length in [100,101]:
        regions = ve._vcf_to_regions(model_dir + vcf_path, seq_length)
        assert np.all(np.in1d(["line_id", "chrom", "start", "end", "ref", "alt", "varpos"], regions.columns.values))
        # 1-based format?
        assert ((regions["end"] - regions["start"]+1) == seq_length).all()
        assert (regions.shape[0] == lct)


def test__bed3():
    chrom = "chr1"
    start = 1
    end = 3
    regions = pd.DataFrame({"chrom": [chrom], "start": [start], "end": [end]})
    temp_bed3_file = tempfile.mktemp()  # file path of the temp file
    # expects 1-based input
    ve._bed3(regions, temp_bed3_file)
    with open(temp_bed3_file, "r") as ifh:
        for l in ifh:
            assert(l.strip().split("\t") == [chrom, str(start-1), str(end)])
    os.unlink(temp_bed3_file)


def test__generate_seq_sets():
    model_dir = "examples/rbp/"
    vcf_path = "example_files/variants.vcf"
    # for any given input type: list, dict and np.array return 4 identical sets, except for mutated bases on one position
    seq_len = 101
    num_seqs = 5
    empty_seq_input = np.zeros((num_seqs, seq_len, 4))
    empty_other_input = np.zeros((num_seqs, seq_len, 4))-10
    #
    relv_seq_keys = ["seq"]
    annotated_regions = ve._vcf_to_regions(model_dir + vcf_path, seq_len).iloc[:num_seqs,:]
    #
    gr_meta = {
        "ranges": GenomicRanges(annotated_regions["chrom"].values, annotated_regions["start"].values - 1, annotated_regions["end"].values,
                                np.arange(num_seqs))}
    #
    dict_meta = {
        "ranges": {"chr": annotated_regions["chrom"].values, "start": annotated_regions["start"].values - 1, "end": annotated_regions["end"].values,
                                "id":np.arange(num_seqs)}}
    #
    meta_data_options = [gr_meta, dict_meta]

    annotated_regions = _prepare_regions(annotated_regions)

    for meta_data in meta_data_options:
        ## Test the dict case:
        dataloader = dummy_container()
        dataloader.output_schema = dummy_container()
        seq_container = dummy_container()
        seq_container.associated_metadata = ["ranges"]
        dataloader.output_schema.inputs = {"seq": seq_container, "other_input": None}
        inputs = {"seq": copy.deepcopy(empty_seq_input), "other_input": copy.deepcopy(empty_other_input)}
        inputs_2nd_copy = copy.deepcopy(inputs)
        #
        model_input= {"inputs": inputs, "metadata":meta_data}
        ssets = ve._generate_seq_sets(relv_seq_keys, dataloader, model_input, annotated_regions)
        req_cols = ['alt', 'ref_rc', 'ref', 'alt_rc']
        assert np.all(np.in1d(req_cols, list(ssets.keys())))
        for k in req_cols:
            for k2 in inputs:
                assert(k2 in ssets[k])
                if k2 not in relv_seq_keys:
                    assert np.all(ssets[k][k2] == inputs_2nd_copy[k2])
                else:
                    # Assuming modification of matrices works as desired - see its own unit test
                    # Assuming 1-hot coding with background as 0
                    assert np.sum(ssets[k][k2] != inputs_2nd_copy[k2]) == num_seqs
        #
        for k in ["ref", "alt"]:
            for k2 in relv_seq_keys:
                assert np.all(ssets[k][k2] == ssets[k + "_rc"][k2][:,::-1,::-1])




def test__modify_bases():
    # actually modify the bases of a numpy array
    seq_len = 101
    var_pos = np.array([1,2,3,4,5])
    alphabet = np.array(['A', "C", "G", "T"])
    new_base = np.array(['A', "C", "G", "T", "T"])
    lines = np.array([0,4,2,1,3])
    is_rc_vec = np.zeros((5))==1
    is_rc_vec[[1,2]] = True
    for is_rc in [False, True]:
        empty_input = np.zeros((5, seq_len, 4)) - 1
        if is_rc:
            is_rc_vec = ~is_rc_vec
        ve._modify_bases(empty_input, lines, var_pos, new_base, is_rc_vec)
        empty_input[is_rc_vec, ...] = empty_input[is_rc_vec, ::-1, ::-1]
        assert np.all(empty_input.sum(axis=1).sum(axis=1)==((-101)*4 + 4 +1))
        for p,b,l in zip(var_pos, new_base, lines):
            base_sel = alphabet == b
            assert empty_input[l, p, base_sel] == 1
            assert np.all(empty_input[l, p, ~base_sel] == 0)
            assert np.all(empty_input[l, ~np.in1d(np.arange(seq_len), [p]), :] == -1)



def test__get_seq_fields():
    model_dir = "examples/rbp/"
    assert (
    kipoi.postprocessing.utils.generic._get_seq_fields(kipoi.get_model_descr(model_dir, source="dir")) == ['seq'])
    model_dir = "examples/extended_coda/"
    with pytest.raises(Exception):
        kipoi.postprocessing.utils.generic._get_seq_fields(kipoi.get_model_descr(model_dir, source="dir"))


def test__get_dl_bed_fields():
    model_dir = "examples/rbp/"
    assert(
    kipoi.postprocessing.utils.generic._get_dl_bed_fields(kipoi.get_dataloader_descr(model_dir, source="dir")) == ['intervals_file'])
    # This is not valid anymore:
    #model_dir = "examples/extended_coda/"
    #with pytest.raises(Exception):
    #    kipoi.postprocessing.utils.generic._get_dl_bed_fields(kipoi.get_dataloader_descr(model_dir, source="dir"))



def test_dna_reshaper():
    for n_seqs in [1,3,500]:
        for seq_len in [101,1000,1001]:
            for in_shape in [(n_seqs,4,1,1,seq_len), (n_seqs,4,1,seq_len), (n_seqs,seq_len,4)]:
                content = np.arange(n_seqs*seq_len*4)
                start = np.reshape(content, in_shape)
                input_shape = start.shape[1:]
                reshaper_obj = kipoi.postprocessing.utils.generic.Reshape_dna(input_shape)
                reshaped = reshaper_obj.to_standard(start)
                reshaped_2 = reshaper_obj.from_standard(reshaped)
                assert (np.all(start == reshaped_2))
                assert (reshaped.shape[1:] == (seq_len, 4))
                # check the transformed array:
                seq_dim = np.where(np.array(in_shape) == seq_len)[0][0]
                one_hot_dim = np.where(np.array(in_shape) == 4)[0][0]
                swap = seq_dim > one_hot_dim
                # is the transformation performed correctly?
                for n in range(n_seqs):
                    itm = np.squeeze(start[n,...])
                    if swap:
                        itm = np.swapaxes(itm,1,0)
                    assert np.all(itm == reshaped[n,...])
                # make sure it fails if there is spmething wrong:
                for expa in range(len(in_shape)):
                    with pytest.raises(Exception):
                        reshaped = reshaper_obj.to_standard(np.expand_dims(start, expa))


def test_var_eff_pred():
    if sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")
    # Take the rbp model
    model_dir = "examples/rbp/"
    if INSTALL_REQ:
        install_model_requirements(model_dir, "dir", and_dataloaders=True)
    #
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
    vcf_path = "example_files/variants.vcf"
    out_vcf_fpath = "example_files/variants_generated.vcf"
    ref_out_vcf_fpath = "example_files/variants_ref_out.vcf"
    #
    with cd(model.source_dir):
        model_info = kipoi.postprocessing.Model_info_extractor(model, Dataloader)
        writer = kipoi.postprocessing.Vcf_writer(model, vcf_path, out_vcf_fpath)
        vcf_to_region = kipoi.postprocessing.SNV_centered_rg(model_info)
        res = sp.predict_snvs(model_info, vcf_path, dataloader_args=dataloader_arguments,
                              evaluation_function=analyse_model_preds, batch_size=32,
                              vcf_to_region = vcf_to_region,
                              evaluation_function_kwargs={'diff_types': {'diff': Diff("absmax")}},
                              sync_pred_writer=writer)
        writer.close()
        # pass
        #assert filecmp.cmp(out_vcf_fpath, ref_out_vcf_fpath)
        compare_vcfs(out_vcf_fpath, ref_out_vcf_fpath)
        os.unlink(out_vcf_fpath)



def test_var_eff_pred2():
    if sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")
    # Take the rbp model
    model_dir = "examples/rbp/"
    if INSTALL_REQ:
        install_model_requirements(model_dir, "dir", and_dataloaders=True)
    #
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
    vcf_path = "example_files/variants.vcf"
    out_vcf_fpath = "example_files/variants_generated2.vcf"
    ref_out_vcf_fpath = "example_files/variants_ref_out2.vcf"
    restricted_regions_fpath = "example_files/restricted_regions.bed"
    #
    with cd(model.source_dir):
        pbd = pb.BedTool(restricted_regions_fpath)
        model_info = kipoi.postprocessing.Model_info_extractor(model, Dataloader)
        vcf_to_region = kipoi.postprocessing.SNV_pos_restricted_rg(model_info, pbd)
        writer = kipoi.postprocessing.utils.io.Vcf_writer(model, vcf_path, out_vcf_fpath)
        res = sp.predict_snvs(model_info, vcf_path, dataloader_args=dataloader_arguments,
                              evaluation_function=analyse_model_preds,batch_size=32,
                              vcf_to_region = vcf_to_region,
                              evaluation_function_kwargs={'diff_types': {'diff': Diff("absmax")}},
                              sync_pred_writer=writer)
        writer.close()
        # pass
        #assert filecmp.cmp(out_vcf_fpath, ref_out_vcf_fpath)
        compare_vcfs(out_vcf_fpath, ref_out_vcf_fpath)
        os.unlink(out_vcf_fpath)



def test_Rc_merging():
    # test the variant effect calculation routines
    # test the different functions:
    arr_a = np.array([[1,2],[3,4]])
    arr_b = np.array([[2,1],[5,3]])
    for k in ["min", "max", "mean", "median", lambda x,y: x-y]:
        ro = Rc_merging_pred_analysis(k)
        if k == "min":
            assert np.all(ro.rc_merging(arr_a, arr_b) == np.min([arr_a, arr_b], axis=0))
        elif k == "max":
            assert np.all(ro.rc_merging(arr_a, arr_b) == np.max([arr_a, arr_b], axis=0))
        elif k == "mean":
            assert np.all(ro.rc_merging(arr_a, arr_b) == np.mean([arr_a, arr_b], axis=0))
        elif k == "median":
            assert np.all(ro.rc_merging(arr_a, arr_b) == np.median([arr_a, arr_b], axis=0))
        else:
            assert np.all(ro.rc_merging(arr_a, arr_b) == arr_a - arr_b)
    assert np.all(
        Rc_merging_pred_analysis.absmax(arr_a, arr_b * (-1), inplace=False) == np.array([[-2, 2], [-5, 4]]))
    x = Rc_merging_pred_analysis.absmax(arr_a, arr_b * (-1), inplace=True)
    assert np.all(arr_a== np.array([[-2, 2], [-5, 4]]))


def test_enhanced_analysis_effects():
    probs_r = np.array([0.1,0.2,0.3])
    probs_a = np.array([0.2,0.29,0.9])
    counts = np.array([10,23,-2])
    preds_prob = {"ref":probs_a, "ref_rc":probs_r, "alt":probs_a, "alt_rc":probs_a}
    preds_arb = {"ref":probs_a, "ref_rc":probs_r, "alt":counts, "alt_rc":counts}
    assert np.all((Logit()(**preds_prob) == logit(probs_a) - logit(probs_r)))
    assert np.all((Diff()(**preds_prob) == probs_a - probs_r))
    assert np.all(DeepSEA_effect()(**preds_prob) == np.abs(logit(probs_a) - logit(probs_r)) * np.abs(probs_a - probs_r))
    # now with values that contain values outside [0,1].
    with pytest.warns(UserWarning):
        x =(Logit()(**preds_arb))
    #
    with pytest.warns(UserWarning):
        x =(DeepSEA_effect()(**preds_arb))
    #
    assert np.all((Diff()(**preds_arb) == counts - probs_r))



def test_output_reshaper():
    for k1 in RES:
        for k2 in YAMLS:
            if k1 == k2:
                o = Output_reshaper(ModelSchema.from_config(from_yaml(YAMLS[k2])).targets)
                fl, fll = o.flatten(RES[k1])
                assert (fl.shape[1] == RES_OUT_SHAPES[k1])
                assert (RES_OUT_LABELS[k2] == fll.tolist())
            elif (k1.replace("Lab", "NoLab") == k2) or (k1 == k2.replace("Lab", "NoLab")):
                o = Output_reshaper(ModelSchema.from_config(from_yaml(YAMLS[k2])).targets)
                fl, fll = o.flatten(RES[k1])
                assert (fl.shape[1] == RES_OUT_SHAPES[k1])
                assert (RES_OUT_LABELS[k2] == fll.tolist())
            else:
                with pytest.raises(Exception):
                    o = Output_reshaper(ModelSchema.from_config(from_yaml(YAMLS[k2])).targets)
                    fl, fll = o.flatten(RES[k1])



class Dummy_internval:
    def __init__(self):
        self.storage = {"chrom":[], "start":[], "end":[], "id":[]}
    def append_interval(self, **kwargs):
        for k in kwargs:
            self.storage[k].append(kwargs[k])

def _write_regions_from_vcf(vcf_iter, vcf_id_generator_fn, int_write_fn, region_generator):
    for record in vcf_iter:
        if not record.is_indel:
            region = region_generator(record)
            id = vcf_id_generator_fn(record)
            for chrom, start, end in zip(region["chrom"], region["start"], region["end"]):
                int_write_fn(chrom = chrom, start= start, end=end , id=id)

def test__generate_pos_restricted_seqs():
    model_dir = "examples/rbp/"
    vcf_path = model_dir+"example_files/variants.vcf"
    tuples = (([21541490, 21541591], [21541491, 21541591]),
              ([21541390, 21541891], [21541540, 21541640]),
              ([21541570, 21541891], [21541571, 21541671]))
    model = kipoi.get_model(model_dir, source="dir")
    dataloader = kipoi.get_dataloader_factory(model_dir, source="dir")
    model_info_extractor = kipoi.postprocessing.Model_info_extractor(model, dataloader)
    for tpl in tuples:
        vcf_fh = cyvcf2.VCF(vcf_path, "r")
        qbf = pb.BedTool("chr22 %d %d"%tuple(tpl[0]), from_string=True)
        regions = Dummy_internval()
        #sp._generate_pos_restricted_seqs(vcf_fh, sp._default_vcf_id_gen, qbf, regions.append_interval, seq_length)
        region_generator = kipoi.postprocessing.SNV_pos_restricted_rg(model_info_extractor, qbf)
        _write_regions_from_vcf(vcf_fh, kipoi.postprocessing.utils.generic._default_vcf_id_gen, regions.append_interval, region_generator)
        vcf_fh.close()
        regions_df = pd.DataFrame(regions.storage)
        assert regions_df.shape[0] == 1
        assert np.all(regions_df[["start", "end"]].values == tpl[1])



def test__generate_snv_centered_seqs():
    model_dir = "examples/rbp/"
    vcf_path = model_dir+"example_files/variants.vcf"
    model = kipoi.get_model(model_dir, source="dir")
    dataloader = kipoi.get_dataloader_factory(model_dir, source="dir")
    model_info_extractor = kipoi.postprocessing.Model_info_extractor(model, dataloader)
    lct = 0
    hdr = None
    with open(vcf_path, "r") as ifh:
        for l in ifh:
            if not l.startswith("#"):
                if (len(l.split("\t")[3]) == 1) and (len(l.split("\t")[4]) == 1):
                    lct += 1
            elif l[2] != "#":
                hdr = l.lstrip("#").rstrip().split("\t")
    #
    vcf_df = pd.read_csv(vcf_path, sep="\t", comment='#', header=None, usecols=range(len(hdr)))
    vcf_df.columns = hdr
    # Subset the VCF to SNVs:
    vcf_df = vcf_df.loc[(vcf_df["REF"].str.len() == 1) & (vcf_df["ALT"].str.len() == 1), :]
    #
    for seq_length in [100, 101]:
        vcf_fh = cyvcf2.VCF(vcf_path, "r")
        regions = Dummy_internval()
        model_info_extractor.seq_length = seq_length
        region_generator = kipoi.postprocessing.utils.generic.SNV_centered_rg(model_info_extractor)
        _write_regions_from_vcf(vcf_fh, kipoi.postprocessing.utils.generic._default_vcf_id_gen, regions.append_interval, region_generator)
        vcf_fh.close()
        regions_df = pd.DataFrame(regions.storage)
        #
        # 1-based format?
        assert ((regions_df["end"] - regions_df["start"] + 1) == seq_length).all()
        assert (regions_df.shape[0] == lct)
        assert (regions_df["start"].values == vcf_df["POS"] - int(seq_length/2)).all()


def test__generate_seq_sets_v2():
    model_dir = "examples/rbp/"
    vcf_sub_path = "example_files/variants.vcf"
    model = kipoi.get_model(model_dir, source="dir")
    dataloader = kipoi.get_dataloader_factory(model_dir, source="dir")
    model_info_extractor = kipoi.postprocessing.Model_info_extractor(model, dataloader)
    vcf_path = model_dir + vcf_sub_path
    vcf_path = kipoi.postprocessing.ensure_tabixed_vcf(vcf_path)
    # for any given input type: list, dict and np.array return 4 identical sets, except for mutated bases on one position
    seq_len = 101
    model_info_extractor.seq_length = seq_len
    for num_seqs in [1,5]:
        empty_seq_input = np.zeros((num_seqs, seq_len, 4))
        empty_other_input = np.zeros((num_seqs, seq_len, 4)) - 10
        #
        relv_seq_keys = ["seq"]
        #
        vcf_fh = cyvcf2.VCF(vcf_path)
        regions = Dummy_internval()
        #
        model_info_extractor.seq_length = seq_len
        region_generator = kipoi.postprocessing.utils.generic.SNV_centered_rg(model_info_extractor)
        _write_regions_from_vcf(vcf_fh, kipoi.postprocessing.utils.generic._default_vcf_id_gen, regions.append_interval, region_generator)
        #
        vcf_fh.close()
        annotated_regions = pd.DataFrame(regions.storage).iloc[:num_seqs,:]
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
        meta_data_options = [gr_meta, dict_meta]
        #
        for meta_data in meta_data_options:
            for vcf_search_regions in [False, True]:
                ## Test the dict case:
                dataloader = dummy_container()
                dataloader.output_schema = dummy_container()
                seq_container = dummy_container()
                seq_container.associated_metadata = ["ranges"]
                dataloader.output_schema.inputs = {"seq": seq_container, "other_input": None}
                inputs = {"seq": copy.deepcopy(empty_seq_input), "other_input": copy.deepcopy(empty_other_input)}
                inputs_2nd_copy = copy.deepcopy(inputs)
                #
                model_input = {"inputs": inputs, "metadata": meta_data}
                vcf_fh = cyvcf2.VCF(vcf_path, "r")
                #relv_seq_keys, dataloader, model_input, vcf_fh, vcf_id_generator_fn, array_trafo=None
                ssets = sp._generate_seq_sets(relv_seq_keys, dataloader, model_input, vcf_fh,
                                              kipoi.postprocessing.utils.generic._default_vcf_id_gen,
                                              vcf_search_regions=vcf_search_regions)
                vcf_fh.close()
                req_cols = ['alt', 'ref_rc', 'ref', 'alt_rc']
                assert np.all(np.in1d(req_cols, list(ssets.keys())))
                for k in req_cols:
                    for k2 in inputs:
                        assert (k2 in ssets[k])
                        if k2 not in relv_seq_keys:
                            assert np.all(ssets[k][k2] == inputs_2nd_copy[k2])
                        else:
                            # Assuming modification of matrices works as desired - see its own unit test
                            # Assuming 1-hot coding with background as 0
                            assert np.sum(ssets[k][k2] != inputs_2nd_copy[k2]) == num_seqs
                #
                for k in ["ref", "alt"]:
                    for k2 in relv_seq_keys:
                        assert np.all(ssets[k][k2] == ssets[k + "_rc"][k2][:, ::-1, ::-1])
        #
        #
        # ------ Now also test the bed-restricted prediction -------
        restricted_regions_fpath = "example_files/restricted_regions.bed"
        #
        pbd = pb.BedTool(model_dir + restricted_regions_fpath)
        vcf_fh = cyvcf2.VCF(vcf_path, "r")
        regions = Dummy_internval()
        region_generator = kipoi.postprocessing.SNV_pos_restricted_rg(model_info_extractor, pbd)
        _write_regions_from_vcf(vcf_fh, kipoi.postprocessing.utils.generic._default_vcf_id_gen, regions.append_interval, region_generator)
        #sp._generate_pos_restricted_seqs(vcf_fh, sp._default_vcf_id_gen, pbd, regions.append_interval, seq_len)
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
        meta_data_options = [gr_meta, dict_meta]
        #
        n_qseq = annotated_regions.shape[0]
        for meta_data in meta_data_options:
            for vcf_search_regions in [False, True]:
                ## Test the dict case:
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
                ssets = sp._generate_seq_sets(relv_seq_keys, dataloader, model_input, vcf_fh,
                                              kipoi.postprocessing.utils.generic._default_vcf_id_gen,
                                              vcf_search_regions=vcf_search_regions)
                vcf_fh.close()
                req_cols = ['alt', 'ref_rc', 'ref', 'alt_rc']
                assert np.all(np.in1d(req_cols, list(ssets.keys())))
                for k in req_cols:
                    for k2 in inputs:
                        assert (k2 in ssets[k])
                        if k2 not in relv_seq_keys:
                            assert np.all(ssets[k][k2] == inputs_2nd_copy[k2])
                        else:
                            # Assuming modification of matrices works as desired - see its own unit test
                            # Assuming 1-hot coding with background as 0
                            assert np.sum(ssets[k][k2] != inputs_2nd_copy[k2]) == n_qseq
                #
                for k in ["ref", "alt"]:
                    for k2 in relv_seq_keys:
                        assert np.all(ssets[k][k2] == ssets[k + "_rc"][k2][:, ::-1, ::-1])
                #
                # Now also assert that the nuc change has been performed at the correct position:
                # Region: chr22 36702133    36706137
                # Variant within: chr22 36702137    rs1116  C   A   .   .   .
                mut_pos = 36702137 - 36702134 # bed file is 0-based
                assert np.all(ssets["ref"]["seq"][0, mut_pos, :] == np.array([0, 1, 0, 0]))
                assert np.all(ssets["alt"]["seq"][0, mut_pos, :] == np.array([1, 0, 0, 0]))


def test_subsetting():
    for sel in [[0], [1, 2, 3]]:
        for k in RES:
            if "dict" in k:
                ret = sp.select_from_model_inputs(RES[k], sel, 50)
                for k2 in ret:
                    assert ret[k2].shape[0] == len(sel)
                with pytest.raises(Exception):
                    ret = sp.select_from_model_inputs(RES[k], sel, 20)
            elif "list" in k:
                ret = sp.select_from_model_inputs(RES[k], sel, 50)
                assert all([el.shape[0] == len(sel) for el in ret])
                with pytest.raises(Exception):
                    ret = sp.select_from_model_inputs(RES[k], sel, 20)
            else:
                ret = sp.select_from_model_inputs(RES[k], sel, 50)
                assert ret.shape[0] == len(sel)
                with pytest.raises(Exception):
                    ret = sp.select_from_model_inputs(RES[k], sel, 20)

def test_ensure_tabixed_vcf():
    vcf_in_fpath = "examples/rbp/example_files/variants.vcf"
    vcf_path = kipoi.postprocessing.ensure_tabixed_vcf(vcf_in_fpath)
    assert os.path.exists(vcf_path)
    assert vcf_path.endswith(".gz")
    with pytest.raises(Exception):
        # since the file exists, we should now complain
        vcf_path = kipoi.postprocessing.ensure_tabixed_vcf(vcf_in_fpath, force_tabix=False)
    vcf_in_fpath_gz = vcf_in_fpath + ".gz"
    assert  vcf_in_fpath_gz == kipoi.postprocessing.ensure_tabixed_vcf(vcf_in_fpath_gz)


def test__overlap_vcf_region():
    vcf_path = kipoi.postprocessing.ensure_tabixed_vcf("examples/rbp/example_files/variants.vcf")
    vcf_obj = cyvcf2.VCF(vcf_path)
    all_records = [rec for rec in vcf_obj]
    vcf_obj.close()
    vcf_obj = cyvcf2.VCF(vcf_path)
    #
    regions_dict = {"chr": ["chr22"], "start": [21541589],"end": [36702137],"id": [0]}
    regions_gr = GenomicRanges(regions_dict["chr"], regions_dict["start"],
                               regions_dict["end"],regions_dict["id"])
    for regions in [regions_dict, regions_gr]:
        found_vars, overlapping_region = sp._overlap_vcf_region(vcf_obj, regions, exclude_indels=False)
        assert all([str(el1) == str(el2) for el1, el2 in zip(all_records, found_vars)])
        assert len(overlapping_region) == len(found_vars)
        assert all([el == 0 for el in overlapping_region])

    regions_dict = {"chr": ["chr22", "chr22", "chr22"], "start": [21541589, 21541589, 30630220], "end": [36702137, 21541590, 30630222], "id": [0,1,2]}
    regions_gr = GenomicRanges(regions_dict["chr"], regions_dict["start"],
                               regions_dict["end"], regions_dict["id"])
    #
    plus_indel_results = all_records + all_records[:1] + all_records[3:4]
    snv_results = [el for el in plus_indel_results if not el.is_indel]
    #
    ref_lines_indel = [0]*len(all_records) + [1] + [2]
    snv_ref_lines = [el for el, el1 in zip(ref_lines_indel,plus_indel_results) if not el1.is_indel]
    #
    for regions in [regions_dict, regions_gr]:
        for exclude_indels, ref_res, ref_lines in zip([False, True], [plus_indel_results, snv_results], [ref_lines_indel, snv_ref_lines]):
            found_vars, overlapping_region = sp._overlap_vcf_region(vcf_obj, regions, exclude_indels)
            assert all([str(el1) == str(el2) for el1, el2 in zip(ref_res, found_vars) if not el1.is_indel])
            assert overlapping_region == ref_lines






"""
# Take the rbp model
model_dir = "examples/rbp/"
install_model_requirements(model_dir, "dir", and_dataloaders=True)

model = kipoi.get_model(model_dir, source="dir")
# The preprocessor
Dataloader = kipoi.get_dataloader_factory(model_dir, source="dir")

dataloader_arguments = {
    "fasta_file": "example_files/hg38_chr22.fa",
    "preproc_transformer": "dataloader_files/encodeSplines.pkl",
    "gtf_file": "example_files/gencode_v25_chr22.gtf.pkl.gz",
}

# Run the actual predictions
vcf_path = "example_files/variants.vcf"
out_vcf_fpath = "example_files/variants_generated.vcf"
ref_out_vcf_fpath = "example_files/variants_ref_out.vcf"


with cd(model.source_dir):
    res = ve.predict_snvs(model, vcf_path, dataloader_args=dataloader_arguments,
                          evaluation_function=analyse_model_preds,
                       dataloader=Dataloader, batch_size=32,
                       evaluation_function_kwargs={'diff_types':{'diff':Diff("absmax")}},
                       out_vcf_fpath=out_vcf_fpath)
    assert filecmp.cmp(out_vcf_fpath, ref_out_vcf_fpath)
    os.unlink(out_vcf_fpath)
"""
