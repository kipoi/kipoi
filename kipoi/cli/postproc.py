"""Postprocessing CLI
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import sys
import copy
import logging
import os

import kipoi
from kipoi.cli.parser_utils import add_model, add_dataloader, file_exists, dir_exists
from kipoi.components import default_kwargs
from kipoi.postprocessing.variant_effects.components import VarEffectFuncType
from kipoi.utils import load_module, getargs
from kipoi.utils import parse_json_file_str
from kipoi import writers
from kipoi.utils import cd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

scoring_options = {
    # deepsea_effect diff logit_diff logit_ref logit_alt
    # TODO - we should add more options to it: ref, alt, ref_rc, alt_rc
    "ref": kipoi.variant_effects.Ref,
    "alt": kipoi.variant_effects.Alt,
    "diff": kipoi.variant_effects.Diff,
    "logit_ref": kipoi.variant_effects.LogitRef,
    "logit_alt": kipoi.variant_effects.LogitAlt,
    "logit": kipoi.variant_effects.Logit,
    "deepsea_effect": kipoi.variant_effects.DeepSEA_effect
}

scoring_names = {
    VarEffectFuncType.diff: "diff",
    VarEffectFuncType.ref: "ref",
    VarEffectFuncType.alt: "alt",
    VarEffectFuncType.logit: "logit",
    VarEffectFuncType.logit_ref: "logit_ref",
    VarEffectFuncType.logit_alt: "logit_alt",
    VarEffectFuncType.deepsea_effect: "deepsea_effect",
}

# if e.g. 'logit' is allowed then also 'logit_alt' is allowed, etc. values here refer to 'scoring_options' keys.
categorical_enable = {
    "__any__": ["diff", "ref", "alt"],
    "logit": ["logit", "logit_ref", "logit_alt", "deepsea_effect"]
}

builtin_default_kwargs = {"rc_merging": "mean"}


def get_avail_scoring_methods(model):
    if model.postprocessing.variant_effects is None:
        raise Exception("Model deosn't support variant effect prediction according model yaml file.")
    avail_scoring_fns = []  # contains callables
    avail_scoring_fn_def_args = []  # default kwargs for every callable
    avail_scoring_fn_names = []  # contains the labels
    default_scoring_fns = []  # contains the labels of the defaults
    for sf in model.postprocessing.variant_effects.scoring_functions:
        if sf.type is not VarEffectFuncType.custom:
            sn = scoring_names[sf.type]
            sf_obj = scoring_options[sn]
            s_label = sn
            if (sf.name != "") and (sf.name not in scoring_options):
                if sf.name in avail_scoring_fn_names:
                    raise Exception("Scoring function names have to unique in the model yaml file.")
                s_label = sf.name
            def_kwargs = builtin_default_kwargs
        else:
            prefix = ""
            if (sf.name == "") or (sf.name in scoring_options):
                prefix = "custom_"
            s_label = prefix + sf.name
            if s_label in avail_scoring_fn_names:
                raise Exception("Scoring function names have to unique in the model yaml file.")
            if sf.defined_as == "":
                raise Exception("`defined_as` has to be defined for a custom function.")
            file_path, obj_name = tuple(sf.defined_as.split("::"))
            sf_obj = getattr(load_module(file_path), obj_name)
            # check that the scoring function arguments match yaml arguments
            if not getargs(sf_obj) == set(sf.args.keys()):
                raise ValueError("Scoring function arguments: \n{0}\n don't match ".format(set(getargs(sf_obj))) +
                                 "the specification in the dataloader.yaml file:\n{0}".
                                 format(set(sf.args.keys())))

            def_kwargs = None
            if all([(sf.args[k].default is not None) or (sf.args[k].optional) for k in sf.args]):
                def_kwargs = default_kwargs(sf.args)
            if len(sf.args) == 0:
                def_kwargs = {}  # indicates that this callable doesn't accept any arguments for initialisation.

        if s_label in avail_scoring_fn_names:
            raise Exception("Mulitple scoring functions defined with name '%s' in the model yaml file!" % s_label)

        avail_scoring_fn_def_args.append(def_kwargs)
        avail_scoring_fns.append(sf_obj)
        avail_scoring_fn_names.append(s_label)
        if sf.default:
            default_scoring_fns.append(s_label)

    # if no default scoring functions have been set then take all of the above.
    if len(default_scoring_fns) == 0:
        default_scoring_fns = copy.copy(avail_scoring_fn_names)

    # try to complete the set of functions if needed. None of the additional ones will become part of the defaults
    additional_scoring_fn_def_args = []
    additional_scoring_fns = []
    additional_scoring_fn_names = []
    for scoring_fn, def_args in zip(avail_scoring_fns, avail_scoring_fn_def_args):
        scoring_name = None
        for sn in scoring_options:
            if scoring_fn is scoring_options[sn]:
                scoring_name = sn
                break
        if scoring_name is None:
            continue
        categories = [cat for cat in categorical_enable if scoring_name in categorical_enable[cat]]
        for cat in categories:
            for scoring_name in categorical_enable[cat]:
                if (scoring_options[scoring_name] not in avail_scoring_fns) and \
                        (scoring_options[scoring_name] not in additional_scoring_fns):
                    additional_scoring_fn_def_args.append(def_args)
                    additional_scoring_fns.append(scoring_options[scoring_name])
                    s_label = scoring_name
                    if s_label in avail_scoring_fn_names:
                        s_label = "default_" + s_label
                    additional_scoring_fn_names.append(s_label)

    avail_scoring_fns += additional_scoring_fns
    avail_scoring_fn_def_args += additional_scoring_fn_def_args
    avail_scoring_fn_names += additional_scoring_fn_names

    # add the default scoring functions if not already in there
    for scoring_name in categorical_enable["__any__"]:
        if scoring_options[scoring_name] not in avail_scoring_fns:
            avail_scoring_fn_def_args.append(builtin_default_kwargs)
            avail_scoring_fns.append(scoring_options[scoring_name])
            s_label = scoring_name
            if s_label in avail_scoring_fn_names:
                s_label = "default_" + s_label
            avail_scoring_fn_names.append(s_label)
            if len(default_scoring_fns) == 0:
                default_scoring_fns.append(s_label)

    return avail_scoring_fns, avail_scoring_fn_def_args, avail_scoring_fn_names, default_scoring_fns


def _get_scoring_fns(model, sel_scoring_labels, sel_scoring_kwargs):
    # get the scoring methods according to the model definition
    avail_scoring_fns, avail_scoring_fn_def_args, avail_scoring_fn_names, \
    default_scoring_fns = get_avail_scoring_methods(model)

    errmsg_scoring_kwargs = "When defining `--scoring_kwargs` a JSON representation of arguments (or the path of a" \
                            " file containing them) must be given for every `--scoring` function."

    dts = {}
    if len(sel_scoring_labels) >= 1:
        # Check if all scoring functions should be used:
        if sel_scoring_labels == ["all"]:
            if len(sel_scoring_kwargs) >= 1:
                raise ValueError("`--scoring_kwargs` cannot be defined in combination will `--scoring all`!")
            for arg_iter, k in enumerate(avail_scoring_fn_names):
                si = avail_scoring_fn_names.index(k)
                # get the default kwargs
                kwargs = avail_scoring_fn_def_args[si]
                if kwargs is None:
                    raise ValueError("No default kwargs for scoring function: %s"
                                     " `--scoring all` cannot be used. "
                                     "Please also define `--scoring_kwargs`." % (k))
                # instantiate the scoring fn
                dts[k] = avail_scoring_fns[si](**kwargs)
        else:
            # if -k set check that length matches with -s
            if len(sel_scoring_kwargs) >= 1:
                if not len(sel_scoring_labels) == len(sel_scoring_kwargs):
                    raise ValueError(errmsg_scoring_kwargs)
            for arg_iter, k in enumerate(sel_scoring_labels):
                # if -s set check is available for model
                if k in avail_scoring_fn_names:
                    si = avail_scoring_fn_names.index(k)
                    # get the default kwargs
                    kwargs = avail_scoring_fn_def_args[si]
                    # if the user has set scoring function kwargs then load them here.
                    if len(sel_scoring_kwargs) >= 1:
                        # all the {}s in -k replace by their defaults, if the default is None
                        # raise exception with the corrsponding scoring function label etc.
                        defined_kwargs = parse_json_file_str(sel_scoring_kwargs[si])
                        if len(defined_kwargs) != 0:
                            kwargs = defined_kwargs
                    if kwargs is None:
                        raise ValueError("No kwargs were given for scoring function %s"
                                         " with no defaults but required argmuents. "
                                         "Please also define `--scoring_kwargs`." % (k))
                    # instantiate the scoring fn
                    dts[k] = avail_scoring_fns[si](**kwargs)
                else:
                    logger.warn("Cannot choose scoring function %s. "
                                "Model only supports: %s." % (k, str(avail_scoring_fn_names)))
    # if -s not set use all defaults
    elif len(default_scoring_fns) != 0:
        for arg_iter, k in enumerate(default_scoring_fns):
            si = avail_scoring_fn_names.index(k)
            kwargs = avail_scoring_fn_def_args[si]
            dts[k] = avail_scoring_fns[si](**kwargs)

    if len(dts) == 0:
        raise Exception("No scoring method was chosen!")

    return dts


def cli_score_variants(command, raw_args):
    """CLI interface to score variants
    """
    AVAILABLE_FORMATS = ["tsv", "hdf5", "h5"]
    import pybedtools
    assert command == "score_variants"
    parser = argparse.ArgumentParser('kipoi postproc {}'.format(command),
                                     description='Predict effect of SNVs using ISM.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('-v', '--vcf_path',
                        help='Input VCF.')
    # TODO - rename path to fpath
    parser.add_argument('-a', '--out_vcf_fpath',
                        help='Output annotated VCF file path.', default=None)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-i", "--install_req", action='store_true',
                        help="Install required packages from requirements.txt")
    parser.add_argument('-r', '--restriction_bed', default=None,
                        help="Regions for prediction can only be subsets of this bed file")
    parser.add_argument('-o', '--output', required=False,
                        help="Additional output file. File format is inferred from the file path ending" +
                             ". Available file formats are: {0}".format(",".join(AVAILABLE_FORMATS)))
    parser.add_argument('-s', "--scoring", default="diff", nargs="+",
                        help="Scoring method to be used. Only scoring methods selected in the model yaml file are"
                             "available except for `diff` which is always available. Select scoring function by the"
                             "`name` tag defined in the model yaml file.")
    parser.add_argument('-k', "--scoring_kwargs", default="", nargs="+",
                        help="JSON definition of the kwargs for the scoring functions selected in --scoring. The "
                             "definiton can either be in JSON in the command line or the path of a .json file. The "
                             "individual JSONs are expected to be supplied in the same order as the labels defined in "
                             "--scoring. If the defaults or no arguments should be used define '{}' for that respective "
                             "scoring method.")
    parser.add_argument('-l', "--seq_length", type=int, default=None,
                        help="Optional parameter: Model input sequence length - necessary if the model does not have a "
                             "pre-defined input sequence length.")

    args = parser.parse_args(raw_args)

    # extract args for kipoi.variant_effects.predict_snvs
    vcf_path = args.vcf_path
    out_vcf_fpath = args.out_vcf_fpath
    dataloader_arguments = parse_json_file_str(args.dataloader_args)

    # --------------------------------------------
    # install args
    if args.install_req:
        kipoi.pipeline.install_model_requirements(args.model, args.source, and_dataloaders=True)
    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    if args.dataloader is not None:
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
    else:
        Dl = model.default_dataloader
    with cd(model.source_dir):

        # Check that all the folders exist
        file_exists(args.vcf_path, logger)
        dir_exists(os.path.dirname(args.out_vcf_fpath), logger)
        if args.output is not None:
            dir_exists(os.path.dirname(args.output), logger)

            # infer the file format
            args.file_format = args.output.split(".")[-1]
            if args.file_format not in AVAILABLE_FORMATS:
                logger.error("File ending: {0} for file {1} not from {2}".
                             format(args.file_format, args.output, AVAILABLE_FORMATS))
                sys.exit(1)

            if args.file_format in ["hdf5", "h5"]:
                # only if hdf5 output is used
                import deepdish

        if not isinstance(args.scoring, list):
            args.scoring = [args.scoring]

        dts = _get_scoring_fns(model, args.scoring, args.scoring_kwargs)

        # Load effect prediction related model info
        model_info = kipoi.postprocessing.variant_effects.ModelInfoExtractor(model, Dl)
        manual_seq_len = args.seq_length

        # Select the appropriate region generator
        if args.restriction_bed is not None:
            # Select the restricted SNV-centered region generator
            pbd = pybedtools.BedTool(args.restriction_bed)
            vcf_to_region = kipoi.postprocessing.variant_effects.SnvPosRestrictedRg(model_info, pbd)
            logger.info('Restriction bed file defined. Only variants in defined regions will be tested.'
                        'Only defined regions will be tested.')
        elif model_info.requires_region_definition:
            # Select the SNV-centered region generator
            vcf_to_region = kipoi.postprocessing.variant_effects.SnvCenteredRg(model_info, seq_length=manual_seq_len)
            logger.info('Using variant-centered sequence generation.')
        else:
            # No regions can be defined for the given model, VCF overlap will be inferred, hence tabixed VCF is necessary
            vcf_to_region = None
            # Make sure that the vcf is tabixed
            vcf_path = kipoi.postprocessing.variant_effects.ensure_tabixed_vcf(vcf_path)
            logger.info('Dataloader does not accept definition of a regions bed-file. Only VCF-variants that lie within'
                        'produced regions can be predicted')

    if model_info.use_seq_only_rc:
        logger.info('Model SUPPORTS simple reverse complementation of input DNA sequences.')
    else:
        logger.info('Model DOES NOT support simple reverse complementation of input DNA sequences.')

    with cd(model.source_dir):
        # Get a vcf output writer if needed
        if out_vcf_fpath is not None:
            logger.info('Annotated VCF will be written to %s.' % str(out_vcf_fpath))
            vcf_writer = kipoi.postprocessing.variant_effects.VcfWriter(model, vcf_path, out_vcf_fpath)
        else:
            vcf_writer = None

    keep_predictions = args.output is not None

    res = kipoi.postprocessing.variant_effects.predict_snvs(
        model,
        Dl,
        vcf_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataloader_args=dataloader_arguments,
        vcf_to_region=vcf_to_region,
        evaluation_function_kwargs={"diff_types": dts},
        sync_pred_writer=vcf_writer,
        return_predictions=keep_predictions
    )

    with cd(model.source_dir):
        # tabular files
        if args.output is not None:
            if args.file_format in ["tsv"]:
                for i, k in enumerate(res):
                    # Remove an old file if it is still there...
                    if i == 0:
                        try:
                            os.unlink(args.output)
                        except Exception:
                            pass
                    with open(args.output, "w") as ofh:
                        ofh.write("KPVEP_%s\n" % k.upper())
                        res[k].to_csv(args.output, sep="\t", mode="a")

            if args.file_format in ["hdf5", "h5"]:
                deepdish.io.save(args.output, res)

    logger.info('Successfully predicted samples')


def isint(qstr):
    import re
    return bool(re.match("^[0-9]+$", qstr))


# Parse the slice
def parse_filter_slice(in_str):
    if in_str.startswith("(") or in_str.startswith("["):
        in_str_els = in_str.lstrip("([").rstrip(")]").split(",")
        slices = []
        for slice_el in in_str_els:
            slice_el = slice_el.strip(" ")
            if slice_el == "...":
                slices.append(Ellipsis)
            elif isint(slice_el):
                slices.append(int(slice_el))
                if len(in_str_els) == 1:
                    return int(slice_el)
            else:
                # taken from https://stackoverflow.com/questions/680826/python-create-slice-object-from-string
                slices.append(slice(*map(lambda x: int(x.strip()) if x.strip() else None, slice_el.split(':'))))
        return tuple(slices)
    elif isint(in_str):
        return int(in_str)
    else:
        raise Exception("Filter index slice not valid. Allowed values are e.g.: '1', [1:3,...], [:, 0:4, ...]")


def cli_grad(command, raw_args):
    """CLI interface to predict
    """
    from .main import prepare_batch
    from kipoi.model import GradientMixin
    assert command == "grad"
    from tqdm import tqdm
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Save gradients and inputs to a hdf5 file.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-i", "--install_req", action='store_true',
                        help="Install required packages from requirements.txt")
    parser.add_argument("-l", "--layer", default=None,
                        help="Which output layer to use to make the predictions. If specified," +
                             "`model.predict_activation_on_batch` will be invoked instead of `model.predict_on_batch`",
                        required=False)
    parser.add_argument("--final_layer",
                        help="Alternatively to `--layer` this flag can be used to indicate that the last layer should "
                             "be used.", action='store_true')
    parser.add_argument("--pre_nonlinearity",
                        help="Flag indicating that it should checked whether the selected output is post activation "
                             "function. If a non-linear activation function is used attempt to use its input. This "
                             "feature is not available for all models.", action='store_true')
    parser.add_argument("-f", "--filter_ind",
                        help="Filter index that should be inspected with gradients. If not set all filters will " +
                             "be used.", default=None)
    parser.add_argument("-a", "--avg_func",
                        help="Averaging function to be applied across selected filters (`--filter_ind`) in " +
                             "layer `--layer`.", choices=GradientMixin.allowed_functions, default="sum")
    parser.add_argument('--selected_fwd_node', help="If the selected layer has multiple inbound connections in "
                                                    "the graph then those can be selected here with an integer "
                                                    "index. Not necessarily supported by all models.",
                        default=None, type=int)
    parser.add_argument('-o', '--output', required=True, nargs="+",
                        help="Output files. File format is inferred from the file path ending. Available file formats are: " +
                             ", ".join(["." + k for k in writers.FILE_SUFFIX_MAP]))
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)

    # setup the files
    if not isinstance(args.output, list):
        args.output = [args.output]
    for o in args.output:
        ending = o.split('.')[-1]
        if ending not in writers.FILE_SUFFIX_MAP:
            logger.error("File ending: {0} for file {1} not from {2}".
                         format(ending, o, writers.FILE_SUFFIX_MAP))
            sys.exit(1)
        dir_exists(os.path.dirname(o), logger)
    # --------------------------------------------
    # install args
    if args.install_req:
        kipoi.pipeline.install_model_requirements(args.model,
                                                  args.source,
                                                  and_dataloaders=True)

    layer = args.layer
    if layer is None and not args.final_layer:
        raise Exception("A layer has to be selected explicitely using `--layer` or implicitely by using the"
                        "`--final_layer` flag.")

    # Not a good idea
    # if layer is not None and isint(layer):
    #    logger.warn("Interpreting `--layer` value as integer layer index!")
    #    layer = int(args.layer)


    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    if not isinstance(model, GradientMixin):
        raise Exception("Model does not support gradient calculation.")

    if args.dataloader is not None:
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
    else:
        Dl = model.default_dataloader

    dataloader_kwargs = kipoi.pipeline.validate_kwargs(Dl, dataloader_kwargs)
    dl = Dl(**dataloader_kwargs)

    filter_ind_parsed = None
    if args.filter_ind is not None:
        filter_ind_parsed = parse_filter_slice(args.filter_ind)

    # setup batching
    it = dl.batch_iter(batch_size=args.batch_size,
                       num_workers=args.num_workers)

    # Setup the writers
    use_writers = []
    for output in args.output:
        ending = output.split('.')[-1]
        W = writers.FILE_SUFFIX_MAP[ending]
        logger.info("Using {0} for file {1}".format(W.__name__, output))
        if ending == "tsv":
            assert W == writers.TsvBatchWriter
            use_writers.append(writers.TsvBatchWriter(file_path=output, nested_sep="/"))
        elif ending == "bed":
            raise Exception("Please use tsv or hdf5 output format.")
        elif ending in ["hdf5", "h5"]:
            assert W == writers.HDF5BatchWriter
            use_writers.append(writers.HDF5BatchWriter(file_path=output))
        else:
            logger.error("Unknown file format: {0}".format(ending))
            sys.exit(1)

    # Loop through the data, make predictions, save the output
    for i, batch in enumerate(tqdm(it)):
        # validate the data schema in the first iteration
        if i == 0 and not Dl.output_schema.compatible_with_batch(batch):
            logger.warn("First batch of data is not compatible with the dataloader schema.")

        # make the prediction
        pred_batch = model.input_grad(batch['inputs'], filter_ind=filter_ind_parsed,
                                      avg_func=args.avg_func, wrt_layer=layer, wrt_final_layer=args.final_layer,
                                      selected_fwd_node=args.selected_fwd_node,
                                      pre_nonlinearity=args.pre_nonlinearity)

        # write out the predictions, metadata (, inputs, targets)
        # always keep the inputs so that input*grad can be generated!
        output_batch = prepare_batch(batch, pred_batch, keep_inputs=True)
        for writer in use_writers:
            writer.batch_write(output_batch)

    for writer in use_writers:
        writer.close()
    logger.info('Done! Gradients stored in {0}'.format(",".join(args.output)))


def cli_grad_to_file(command, raw_args):
    """ CLI to save seq inputs of grad*input to a bigwig file
    """
    assert command == "gr_inp_to_file"
    parser = argparse.ArgumentParser('kipoi postproc {}'.format(command),
                                     description='Save grad*input in a file.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    # TODO - rename path to fpath
    parser.add_argument('-f', '--input_file', required=False,
                        help="Input HDF5 file produced from `grad`")
    parser.add_argument('-o', '--output', required=False,
                        help="Output bigwig for bedgraph file")
    parser.add_argument('--input_line', required=False, type=int, default=None,
                        help="Input line for which the BigWig file should be generated. If not defined all"
                             "samples will be written.")
    parser.add_argument('--model_input', required=False, default=None,
                        help="Model input name to be used for plotting. As defined in model.yaml. Can be omitted if"
                             "model only has one input.")
    args = parser.parse_args(raw_args)

    # Check that all the folders exist
    dir_exists(os.path.dirname(args.output), logger)
    # --------------------------------------------
    # install args
    import matplotlib.pyplot
    matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pylab as plt
    from kipoi.postprocessing.variant_effects.mutation_map import MutationMapPlotter
    from kipoi.postprocessing.gradient_vis.vis import GradPlotter
    from kipoi.writers import BedGraphWriter

    logger.info('Loading gradient results file and model info...')

    gp = GradPlotter.from_hdf5(args.input_file, model=args.model, source=args.source)

    if args.input_line is not None:
        samples = [args.input_line]
    else:
        samples = list(range(gp.get_num_samples(args.model_input)))

    if args.output.endswith(".bed") or args.output.endswith(".bedgraph"):
        of_obj = BedGraphWriter(args.output)
    else:
        raise Exception("Output file format not supported!")

    logger.info('Writing...')

    for sample in samples:
        gp.write(sample, model_input=args.model_input, writer_obj=of_obj)

    logger.info('Saving...')

    of_obj.close()

    logger.info('Successfully wrote grad*input to file.')


def cli_create_mutation_map(command, raw_args):
    """CLI interface to calculate mutation map data 
    """
    assert command == "create_mutation_map"
    parser = argparse.ArgumentParser('kipoi postproc {}'.format(command),
                                     description='Predict effect of SNVs using ISM.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('-r', '--regions_file',
                        help='Region definition as VCF or bed file. Not a required input.')
    # TODO - rename path to fpath
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-i", "--install_req", action='store_true',
                        help="Install required packages from requirements.txt")
    parser.add_argument('-o', '--output', required=True,
                        help="Output HDF5 file. To be used as input for plotting.")
    parser.add_argument('-s', "--scoring", default="diff", nargs="+",
                        help="Scoring method to be used. Only scoring methods selected in the model yaml file are"
                             "available except for `diff` which is always available. Select scoring function by the"
                             "`name` tag defined in the model yaml file.")
    parser.add_argument('-k', "--scoring_kwargs", default="", nargs="+",
                        help="JSON definition of the kwargs for the scoring functions selected in --scoring. The "
                             "definiton can either be in JSON in the command line or the path of a .json file. The "
                             "individual JSONs are expected to be supplied in the same order as the labels defined in "
                             "--scoring. If the defaults or no arguments should be used define '{}' for that respective "
                             "scoring method.")
    parser.add_argument('-l', "--seq_length", type=int, default=None,
                        help="Optional parameter: Model input sequence length - necessary if the model does not have a "
                             "pre-defined input sequence length.")

    args = parser.parse_args(raw_args)

    # extract args for kipoi.variant_effects.predict_snvs

    dataloader_arguments = parse_json_file_str(args.dataloader_args)

    if args.output is None:
        raise Exception("Output file `--output` has to be set!")

    # --------------------------------------------
    # install args
    if args.install_req:
        kipoi.pipeline.install_model_requirements(args.model, args.source, and_dataloaders=True)
    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    with cd(model.source_dir):
        if not os.path.exists(args.regions_file):
            raise Exception("Regions inputs file does not exist: %s" % args.regions_file)

        # Check that all the folders exist
        file_exists(args.regions_file, logger)
        dir_exists(os.path.dirname(args.output), logger)

        if args.dataloader is not None:
            Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
        else:
            Dl = model.default_dataloader

    if not isinstance(args.scoring, list):
        args.scoring = [args.scoring]

    dts = _get_scoring_fns(model, args.scoring, args.scoring_kwargs)

    # Load effect prediction related model info
    model_info = kipoi.postprocessing.variant_effects.ModelInfoExtractor(model, Dl)
    manual_seq_len = args.seq_length

    # Select the appropriate region generator and vcf or bed file input
    args.file_format = args.regions_file.split(".")[-1]
    bed_region_file = None
    vcf_region_file = None
    bed_to_region = None
    vcf_to_region = None
    if args.file_format == "vcf" or args.regions_file.endswith("vcf.gz"):
        vcf_region_file = args.regions_file
        if model_info.requires_region_definition:
            # Select the SNV-centered region generator
            vcf_to_region = kipoi.postprocessing.variant_effects.SnvCenteredRg(model_info, seq_length=manual_seq_len)
            logger.info('Using variant-centered sequence generation.')
    elif args.file_format == "bed":
        if model_info.requires_region_definition:
            # Select the SNV-centered region generator
            bed_to_region = kipoi.postprocessing.variant_effects.BedOverlappingRg(model_info, seq_length=manual_seq_len)
            logger.info('Using bed-file based sequence generation.')
        bed_region_file = args.regions_file
    else:
        raise Exception("")

    if model_info.use_seq_only_rc:
        logger.info('Model SUPPORTS simple reverse complementation of input DNA sequences.')
    else:
        logger.info('Model DOES NOT support simple reverse complementation of input DNA sequences.')

    from kipoi.postprocessing.variant_effects.mutation_map import _generate_mutation_map
    mdmm = _generate_mutation_map(model,
                                  Dl,
                                  vcf_fpath=vcf_region_file,
                                  bed_fpath=bed_region_file,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  dataloader_args=dataloader_arguments,
                                  vcf_to_region=vcf_to_region,
                                  bed_to_region=bed_to_region,
                                  evaluation_function_kwargs={'diff_types': dts},
                                  )
    mdmm.save_to_file(args.output)

    logger.info('Successfully generated mutation map data')


def cli_plot_mutation_map(command, raw_args):
    """CLI interface to plot mutation map
    """
    assert command == "plot_mutation_map"
    parser = argparse.ArgumentParser('kipoi postproc {}'.format(command),
                                     description='Plot mutation map in a file.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    # TODO - rename path to fpath
    parser.add_argument('-f', '--input_file', required=False,
                        help="Input HDF5 file produced from `create_mutation_map`")
    parser.add_argument('-o', '--output', required=False,
                        help="Output image file")
    parser.add_argument('--input_line', required=True, type=int,
                        help="Input line for which the plot should be generated")
    parser.add_argument('--model_seq_input', required=True,
                        help="Model input name to be used for plotting. As defined in model.yaml.")
    parser.add_argument('--scoring_key', required=True,
                        help="Variant score label to be used for plotting. As defined when running "
                             "`create_mutation_map`.")
    parser.add_argument('--model_output', required=True,
                        help="Model output to be used for plotting. As defined in model.yaml.")
    args = parser.parse_args(raw_args)

    # Check that all the folders exist
    dir_exists(os.path.dirname(args.output), logger)
    # --------------------------------------------
    # install args
    import matplotlib.pyplot
    matplotlib.pyplot.switch_backend('agg')
    import matplotlib.pylab as plt
    from kipoi.postprocessing.variant_effects.mutation_map import MutationMapPlotter

    logger.info('Loading mutation map file...')

    mutmap = MutationMapPlotter(fname=args.input_file)

    fig = plt.figure(figsize=(50, 5))
    ax = plt.subplot(1, 1, 1)

    logger.info('Plotting...')

    mutmap.plot_mutmap(args.input_line, args.model_seq_input, args.scoring_key, args.model_output, ax=ax)
    fig.savefig(args.output)

    logger.info('Successfully plotted mutation map')


# --------------------------------------------
# CLI commands


command_functions = {
    'score_variants': cli_score_variants,
    'grad': cli_grad,
    'create_mutation_map': cli_create_mutation_map,
    'plot_mutation_map': cli_plot_mutation_map,
    'gr_inp_to_file': cli_grad_to_file
}
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='Kipoi model-zoo command line tool',
    usage='''kipoi postproc <command> [-h] ...

    # Available sub-commands:
    score_variants        Score variants with a kipoi model
    grad                  Save gradients and inputs to a hdf5 file
    gr_inp_to_file        Save grad*input in a file.
    create_mutation_map   Calculate variant effect scores for mutation map plotting
    plot_mutation_map     Plot mutation map from data generated in `create_mutation_map`
    ''')
parser.add_argument('command', help='Subcommand to run; possible commands: {}'.format(commands_str))


def cli_main(command, raw_args):
    args = parser.parse_args(raw_args[0:1])
    if args.command not in command_functions:
        parser.print_help()
        parser.exit(
            status=1,
            message='\nCommand `{}` not found. Possible commands: {}\n'.format(
                args.command, commands_str))
    command_fn = command_functions[args.command]
    command_fn(args.command, raw_args[1:])
