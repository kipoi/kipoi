import cyvcf2
import os
from kipoi.postprocessing.variant_effects import _get_seq_fields, Reshape_dna, _get_seq_shape, _get_dl_bed_fields, Output_reshaper, _process_sequence_set, analyse_model_preds, Logit, _annotate_vcf
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools, copy


def _generate_seq_sets(relv_seq_keys, dataloader, model_input, vcf_fh, vcf_id_generator_fn, array_trafo=None):
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

    # first establish which VCF region we are talking about...
    ranges_input_objs = {}
    null_key = None
    # every sequence key can have it's own region definition
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
        ranges_input_objs[seq_key] = model_input['metadata'][ranges_slots[0]]
        if null_key is None:
            null_key = seq_key
        else:
            for k in  ["chr", "start", "end", "id"]:
                assert(ranges_input_objs[seq_key][k] == ranges_input_objs[null_key][k])

    # now get the right region from the vcf:
    vcf_records = []
    process_ids = []
    for returned_id in ranges_input_objs[null_key]["id"]:
        for record in vcf_fh:
            id = vcf_id_generator_fn(record)
            if str(id) == str(returned_id):
                vcf_records.append(record)
                process_ids.append(returned_id)
                break
            else:
                # Warn here...
                pass

    # Start from the sequence inputs mentioned in the model.yaml
    for seq_key in relv_seq_keys:
        # extract the metadata output
        ranges_input_obj = ranges_input_objs[seq_key]
        #
        # Assemble variant modification information
        preproc_conv = {"pp_line":[], "varpos_rel":[], "strand":[], "ref":[], "alt":[], "start":[], "end":[], "id":[]}

        for i, record in enumerate(vcf_records):
            assert process_ids[i] == ranges_input_obj["id"][i]
            assert (record.end - record.start) == 1 # Catch indels, that needs a slightly modified processing
            preproc_conv["start"].append(ranges_input_obj["start"][i]+1) # convert bed back to 1-based
            preproc_conv["end"].append(ranges_input_obj["end"][i])
            preproc_conv["varpos_rel"].append(int(record.POS) - preproc_conv["start"][-1])

            preproc_conv["ref"].append(str(record.REF))
            preproc_conv["alt"].append(str(record.ALT[0]))
            preproc_conv["id"].append(str(process_ids[i]))
            preproc_conv["pp_line"].append(i)

        if "strand" in ranges_input_obj:
            preproc_conv["strand"] = ranges_input_obj["strand"]
        else:
            preproc_conv["strand"] = ["*"] * len(ranges_input_obj["chr"])

        preproc_conv_df = pd.DataFrame(preproc_conv)

        if preproc_conv_df.shape[0] != len(ranges_input_obj["id"]):
            raise Exception("Error, id mismatch between generated sequences and VCF.")


        # Actually modify sequences according to annotation
        for s_dir, allele in itertools.product(["fwd", "rc"], ["ref", "alt"]):
            k = "%s_%s" % (s_dir, allele)
            if isinstance(dataloader.output_schema.inputs, dict):
                if seq_key not in input_set[k]:
                    raise Exception("Sequence field %s is missing in DataLoader output!" % seq_key)
                input_set[k][seq_key] = _process_sequence_set(input_set[k][seq_key], preproc_conv_df, allele, s_dir, array_trafo)
            elif isinstance(dataloader.output_schema.inputs, list):
                modified_set = []
                for seq_el, input_schema_el in zip(input_set[k], dataloader.output_schema.inputs):
                    if input_schema_el.name == seq_key:
                        modified_set.append(_process_sequence_set(seq_el, preproc_conv_df, allele, s_dir, array_trafo))
                    else:
                        modified_set.append(seq_el)
                input_set[k] = modified_set
            else:
                input_set[k] = _process_sequence_set(input_set[k], preproc_conv_df, allele, s_dir, array_trafo)
    #
    # Reformat so that effect prediction function will get its required inputs
    pred_set = {"ref": input_set["fwd_ref"], "ref_rc": input_set["rc_ref"], "alt": input_set["fwd_alt"], "alt_rc": input_set["rc_alt"]}
    pred_set["mutation_positions"] = preproc_conv_df["varpos_rel"].values
    pred_set["line_id"] = preproc_conv_df["id"].values
    return pred_set


# simple dummy function to save a bed file sequentially
class Bed_writer:
    ## At the moment
    def __init__(self, output_fname):
        self.output_fname = output_fname
        self.ofh = open(self.output_fname, "w")
    #
    def append_interval(self, chrom, start, end, id):
        self.ofh.write("\t".join([str(chrom), str(int(start)-1), str(end), str(id)]) + "\n")
    #
    def close(self):
        self.ofh.close()
    #
    def __enter__(self):
        return self
    #
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def _default_vcf_id_gen(vcf_record, id_delim=":"):
    return str(vcf_record.CHROM) + id_delim + str(vcf_record.POS) + id_delim + str(vcf_record.REF) + id_delim + str(vcf_record.ALT)


def _generate_snv_centered_seqs(vcf_iter, vcf_id_generator_fn, int_write_fn, seq_length, **kwargs):
    seq_length_half = int(seq_length / 2)
    l_offset = seq_length_half
    r_offset = seq_length_half - 1 + seq_length % 2
    for record in vcf_iter:
        #
        if not record.is_indel:
            int_write_fn(chrom = record.CHROM,
                         start = record.POS - l_offset,
                         end = record.POS + r_offset,
                         id = vcf_id_generator_fn(record))


def _generate_pos_restricted_seqs(vcf_iter, vcf_id_generator_fn, pybed_def, int_write_fn, seq_length, **kwargs):
    seq_length_half = int(seq_length / 2)
    l_offset = seq_length_half
    r_offset = seq_length_half - 1 + seq_length % 2
    tabixed = pybed_def.tabix(in_place=False)
    for record in vcf_iter:
        if not record.is_indel:
            # min. 1-bp overlap
            overlap = tabixed.tabix_intervals("%s:%d-%d" % (record.CHROM, record.start - 1, record.end - 1))
            #overlap = [[0]*101, [0]*200]
            for interval in overlap:
                i_s = interval.start + 1
                i_e = interval.end
                #i_s = 21541391
                #i_e = 21541891
                if len(interval) < seq_length:
                    continue

                if len(interval) != seq_length:
                    var_center = int((record.start + record.end)/2)
                    centered_se = np.array([(record.POS - l_offset), (record.POS + r_offset)])
                    start_missing = centered_se[0] - i_s  # >=0 if ok
                    end_missing = i_e - centered_se[1] # >=0 if ok
                    if start_missing < 0:
                        centered_se -= start_missing # shift right
                    elif end_missing < 0:
                        centered_se += end_missing # shift left
                    assert centered_se[1]-centered_se[0] + 1 == seq_length
                    assert (i_s<= centered_se[0]) and  (i_e>= centered_se[1])
                    i_s, i_e = centered_se.tolist()
                
                int_write_fn(chrom = record.CHROM,
                             start = i_s,
                             end = i_e,
                             id = vcf_id_generator_fn(record))



def predict_snvs(model,
                 vcf_fpath,
                 dataloader,
                 batch_size,
                 num_workers=0,
                 dataloader_args=None,
                 model_out_annotation=None,
                 vcf_to_region_fn=_generate_snv_centered_seqs,
                 vcf_to_region_fn_kwargs = None,
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
        raise Exception("DNA sequence output shapes must agree for fields: %s" % str(seq_fields))

    seq_shape = list(seq_shapes)[0]

    dna_seq_trafo = Reshape_dna(seq_shape)
    seq_length = dna_seq_trafo.get_seq_len()

    temp_bed3_file = tempfile.mktemp()  # file path of the temp file

    vcf_fh = cyvcf2.VCF(vcf_fpath, "r")

    if vcf_to_region_fn_kwargs is None:
        vcf_to_region_fn_kwargs = {}

    with Bed_writer(temp_bed3_file) as ofh:
        vcf_to_region_fn(vcf_iter=vcf_fh,
                        vcf_id_generator_fn=_default_vcf_id_gen,
                        int_write_fn=ofh.append_interval,
                        seq_length=seq_length,
                         **vcf_to_region_fn_kwargs)

    vcf_fh.close()

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
                # raise Exception("Variant effect prediction with list(array) model output not implemented!")
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

    # Open vcf again
    vcf_fh = cyvcf2.VCF(vcf_fpath, "r")

    # pre-process regions
    keys = set()
    for i, batch in enumerate(tqdm(it)):
        # For debugging
        # if i >= 10:
        #     break
        # becomes noticable for large vcf's. Is there a way to avoid it? (i.e. to exploit the iterative nature of dataloading)
        eval_kwargs = _generate_seq_sets(seq_fields, dataloader, batch, vcf_fh, _default_vcf_id_gen, array_trafo=dna_seq_trafo)
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
        res_here = evaluation_function(model, output_reshaper=out_reshaper, **eval_kwargs)
        for k in res_here:
            keys.add(k)
            res_here[k].index = eval_kwargs["line_id"]
        res.append(res_here)

    vcf_fh.close()

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
        _annotate_vcf(vcf_fpath, out_vcf_fpath, res_concatenated, model_name=model_name)

    try:
        os.unlink(temp_bed3_file)
    except:
        pass
    return res_concatenated