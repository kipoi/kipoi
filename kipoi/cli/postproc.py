"""Postprocessing CLI
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import argparse
import kipoi
from kipoi.cli.parser_utils import add_model, add_dataloader, file_exists, dir_exists
from kipoi.utils import parse_json_file_str, cd
import logging
import copy
from kipoi.components import default_kwargs
from kipoi.postprocessing.components import VarEffectFuncType
from kipoi.utils import load_module, getargs
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

scoring_options = {
    # deepsea_scr diff logit_diff logit_ref logit_alt
    # TODO - we should add more options to it: ref, alt, ref_rc, alt_rc
    "logit": kipoi.variant_effects.Logit,
    "diff": kipoi.variant_effects.Diff,
    "logit_ref": kipoi.variant_effects.LogitRef,
    "logit_alt": kipoi.variant_effects.LogitAlt,
    # TODO - function name and string options should be the same
    # i.e. I'd use DeepSEA_effect...
    "deepsea_scr": kipoi.variant_effects.DeepSEA_effect
}

scoring_names = {
    VarEffectFuncType.diff: "diff",
    VarEffectFuncType.logit: "logit",
    VarEffectFuncType.logit_ref: "logit_ref",
    VarEffectFuncType.logit_alt: "logit_alt",
    VarEffectFuncType.deepsea_scr: "deepsea_scr",
}

builtin_default_kwargs = {"rc_merging": "mean"}


def _get_avail_scoring_methods(model):
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

    if len(default_scoring_fns) == 0:
        default_scoring_fns = copy.copy(avail_scoring_fn_names)

    if scoring_options["diff"] not in avail_scoring_fns:
        avail_scoring_fn_def_args.append(builtin_default_kwargs)
        avail_scoring_fns.append(scoring_options["diff"])
        s_label = "diff"
        if s_label in avail_scoring_fn_names:
            s_label = "default_" + s_label
        avail_scoring_fn_names.append(s_label)
        if len(default_scoring_fns) == 0:
            default_scoring_fns.append(s_label)

    return avail_scoring_fns, avail_scoring_fn_def_args, avail_scoring_fn_names, default_scoring_fns

def _get_scoring_fns(model, sel_scoring_labels, sel_scoring_kwargs):
    # get the scoring methods according to the model definition
    avail_scoring_fns, avail_scoring_fn_def_args, avail_scoring_fn_names, \
        default_scoring_fns = _get_avail_scoring_methods(model)

    errmsg_scoring_kwargs = "When defining `--scoring_kwargs` a JSON representation of arguments (or the path of a" \
                            " file containing them) must be given for every `--scoring` function."

    dts = {}
    if len(sel_scoring_labels) >= 1:
        # if -k set check that length matches with -s
        if len(sel_scoring_kwargs) >= 1:
            if not len(sel_scoring_labels) == len(sel_scoring_kwargs):
                raise ValueError(errmsg_scoring_kwargs)
        dts = {}
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
                raise ValueError("Cannot choose scoring function %s. "
                                 "Model only supports: %s." % (k, str(avail_scoring_fn_names)))
    # if -s not set use all defaults
    elif len(default_scoring_fns) != 0:
        for arg_iter, k in enumerate(default_scoring_fns):
            si = avail_scoring_fn_names.index(k)
            kwargs = avail_scoring_fn_def_args[si]
            dts[k] = avail_scoring_fns[si](**kwargs)
    else:
        raise Exception("No scoring method was chosen!")

    return dts


# TODO - --output is not always required
def cli_score_variants(command, raw_args):
    """CLI interface to predict
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

    args = parser.parse_args(raw_args)

    # extract args for kipoi.variant_effects.predict_snvs
    vcf_path = args.vcf_path
    out_vcf_fpath = args.out_vcf_fpath
    dataloader_arguments = parse_json_file_str(args.dataloader_args)

    # infer the file format
    args.file_format = args.output.split(".")[-1]
    if args.file_format not in AVAILABLE_FORMATS:
        logger.error("File ending: {0} for file {1} not from {2}".
                     format(args.file_format, args.output, AVAILABLE_FORMATS))
        sys.exit(1)

    if args.file_format in ["hdf5", "h5"]:
        # only if hdf5 output is used
        import deepdish

    # Check that all the folders exist
    file_exists(args.vcf_path, logger)
    dir_exists(os.path.dirname(args.out_vcf_fpath), logger)
    if args.output is not None:
        dir_exists(os.path.dirname(args.output), logger)
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

    if not os.path.exists(vcf_path):
        raise Exception("VCF file does not exist: %s" % vcf_path)

    if not isinstance(args.scoring, list):
        args.scoring = [args.scoring]

    dts = _get_scoring_fns(model, args.scoring, args.scoring_kwargs)

    # Load effect prediction related model info
    model_info = kipoi.postprocessing.ModelInfoExtractor(model, Dl)

    # Select the appropriate region generator
    if args.restriction_bed is not None:
        # Select the restricted SNV-centered region generator
        pbd = pybedtools.BedTool(args.restriction_bed)
        vcf_to_region = kipoi.postprocessing.SnvPosRestrictedRg(model_info, pbd)
        logger.info('Restriction bed file defined. Only variants in defined regions will be tested.'
                    'Only defined regions will be tested.')
    elif model_info.requires_region_definition:
        # Select the SNV-centered region generator
        vcf_to_region = kipoi.postprocessing.SnvCenteredRg(model_info)
        logger.info('Using variant-centered sequence generation.')
    else:
        # No regions can be defined for the given model, VCF overlap will be inferred, hence tabixed VCF is necessary
        vcf_to_region = None
        # Make sure that the vcf is tabixed
        vcf_path = kipoi.postprocessing.ensure_tabixed_vcf(vcf_path)
        logger.info('Dataloader does not accept definition of a regions bed-file. Only VCF-variants that lie within'
                    'produced regions can be predicted')

    if model_info.use_seq_only_rc:
        logger.info('Model SUPPORTS simple reverse complementation of input DNA sequences.')
    else:
        logger.info('Model DOES NOT support simple reverse complementation of input DNA sequences.')

    # Get a vcf output writer if needed
    if out_vcf_fpath is not None:
        logger.info('Annotated VCF will be written to %s.' % str(out_vcf_fpath))
        vcf_writer = kipoi.postprocessing.VcfWriter(model, vcf_path, out_vcf_fpath)
    else:
        vcf_writer = None

    keep_predictions = args.output is not None

    res = kipoi.postprocessing.predict_snvs(
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


# --------------------------------------------
# CLI commands

command_functions = {
    'score_variants': cli_score_variants,
}
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='Kipoi model-zoo command line tool',
    usage='''kipoi postproc <command> [-h] ...

    # Available sub-commands:
    score_variants   Score variants with a kipoi model
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
