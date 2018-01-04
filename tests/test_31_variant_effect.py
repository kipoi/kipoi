import kipoi
import kipoi.postprocessing.variant_effects as ve
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

warnings.filterwarnings('ignore')


from kipoi.components import ArraySchema, ModelSchema
from related import from_yaml
from kipoi.postprocessing.variant_effects import Output_reshaper

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
    assert (ve._get_seq_len([np.array([111])]) == (1,))
    assert (ve._get_seq_len((np.array([111]))) == (1,))
    assert (ve._get_seq_len({"a": np.array([111]), "b": np.array([111])}) == (1,))
    assert (ve._get_seq_len(np.array([111])) == (1,))



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
    assert (ve._get_seq_fields(kipoi.get_model_descr(model_dir, source="dir")) == ['seq'])
    model_dir = "examples/extended_coda/"
    with pytest.raises(Exception):
        ve._get_seq_fields(kipoi.get_model_descr(model_dir, source="dir"))


def test__get_dl_bed_fields():
    model_dir = "examples/rbp/"
    assert(ve._get_dl_bed_fields(kipoi.get_dataloader_descr(model_dir, source="dir")) == ['intervals_file'])
    model_dir = "examples/extended_coda/"
    with pytest.raises(Exception):
        ve._get_dl_bed_fields(kipoi.get_dataloader_descr(model_dir, source="dir"))



def test_dna_reshaper():
    for n_seqs in [1,3,500]:
        for seq_len in [101,1000,1001]:
            for in_shape in [(n_seqs,4,1,1,seq_len), (n_seqs,4,1,seq_len), (n_seqs,seq_len,4)]:
                content = np.arange(n_seqs*seq_len*4)
                start = np.reshape(content, in_shape)
                input_shape = start.shape[1:]
                reshaper_obj = ve.Reshape_dna(input_shape)
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
    #
    with cd(model.source_dir):
        res = ve.predict_snvs(model, vcf_path, dataloader_args=dataloader_arguments,
                              evaluation_function=analyse_model_preds,
                              dataloader=Dataloader, batch_size=32,
                              evaluation_function_kwargs={'diff_types': {'ism': Diff("absmax")}},
                              out_vcf_fpath=out_vcf_fpath)
        # pass
        assert filecmp.cmp(out_vcf_fpath, ref_out_vcf_fpath)
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
                       evaluation_function_kwargs={'diff_types':{'ism':Diff("absmax")}},
                       out_vcf_fpath=out_vcf_fpath)
    assert filecmp.cmp(out_vcf_fpath, ref_out_vcf_fpath)
    os.unlink(out_vcf_fpath)
"""
