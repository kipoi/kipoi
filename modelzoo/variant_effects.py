import pandas as pd
import numpy as np
import copy
import tempfile

def _vcf_to_regions(vcf_fpath, seq_length, id_delim=":"):
    colnames = ["chrom", "pos", "id", "ref", "alt"]
    vcf = pd.read_csv(vcf_fpath, sep="\t", comment='#', header=None, usecols= range(len(colnames)))
    vcf.columns = colnames
    vcf["chrom"] = "chr" + vcf["chrom"].str.lstrip("chr")
    seq_length = int(seq_length)
    l_offset = seq_length / 2
    r_offset = seq_length / 2 - 1 + seq_length % 2
    ids = vcf["chrom"] + id_delim + vcf["pos"].astype(np.int) + id_delim + vcf["pos"] + id_delim + vcf["ref"] + id_delim + vcf["alt"]
    regions = pd.DataFrame({"id": ids, "chrom": "chr" + vcf["chrom"].astype(np.str),
                            "start": vcf["pos"] - l_offset, "end": vcf["pos"] + r_offset})
    return regions

def _bed3(regions, fpath):
    regions_0based = copy.deepcopy(regions)
    regions_0based["start"] = regions_0based["start"]-1
    regions_0based[["chrom", "start", "end"]].to_csv(fpath, sep="\t", header=False, index=False)

def _generate_seq_sets(model_input):
    # This function has to convert the DNA regions in the model input according to ref, alt, fwd, rc and
    # return a dictionary of which the keys are compliant with evaluation_function arguments

    # Therefore it requires:
    # - info from the description.yaml where the DNA sequences are and if they are already fwd/rc
    # - what the individual sequence is (id or region etc...)

    raise Exception("Not implemented")
    pass

def predict_variants(model_handle, vcf_fpath, seq_length, evaluation_function, other_files_path = None):
    if 'intervals_file' not in model_handle.preproc.get_avail_arguments():
        raise Exception("Preprocessor does not support DNA regions as input.")
    seq_pp_outputs = model_handle.preproc.get_output_label_by_type("dna")
    if len(seq_pp_outputs)==0:
        raise Exception("Preprocessor does not generate DNA sequences.")
    regions = _vcf_to_regions(vcf_fpath, seq_length)
    region_file = {}
    temp_bed3_file = tempfile.mktemp()[1] # file path of the temp file
    _bed3(regions, temp_bed3_file)
    region_file['intervals_file'] = temp_bed3_file

    res = []
    for batch in model_handle.run_preproc(other_files_path, region_file):
        res.append(evaluation_function(model_handle.get_model_obj(), **_generate_seq_sets(batch)))

    return res


