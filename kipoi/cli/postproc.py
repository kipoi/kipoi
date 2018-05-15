"""Postprocessing CLI
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import logging
import os
import sys

import kipoi
from kipoi.cli.parser_utils import add_model, add_dataloader, file_exists, dir_exists
from kipoi.postprocessing.variant_effects.components import VarEffectFuncType
from kipoi.postprocessing.variant_effects.scores import get_scoring_fns
from kipoi.utils import cd
from kipoi.utils import parse_json_file_str

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _prepare_multi_model_args(args):
    assert isinstance(args.model, list)
    assert isinstance(args.source, list)
    assert isinstance(args.seq_length, list)
    assert isinstance(args.dataloader, list)
    assert isinstance(args.dataloader_source, list)
    assert isinstance(args.dataloader_args, list)

    def ensure_matching_args(ref_arg, query_arg, ref_label, query_label, allow_zero=True):
        assert isinstance(ref_arg, list)
        assert isinstance(query_arg, list)
        n = len(ref_arg)
        if allow_zero and (len(query_arg) == 0):
            ret = [None] * n
        elif len(query_arg) == 1:
            ret = [query_arg[0]] * n
        elif not len(query_arg) == n:
            raise Exception("Either give one {q} for all {r} or one {q} for every {r} in the same order.".format(
                q=query_label, r=ref_label))
        else:
            ret = query_arg
        return ret

    args.source = ensure_matching_args(args.model, args.source, "--model", "--source", allow_zero=False)
    args.seq_length = ensure_matching_args(args.model, args.seq_length, "--model", "--seq_length")
    args.dataloader = ensure_matching_args(args.model, args.dataloader, "--model", "--dataloader")
    args.dataloader_source = ensure_matching_args(args.dataloader, args.dataloader_source, "--dataloader",
                                                  "--dataloader_source")
    args.dataloader_args = ensure_matching_args(args.model, args.dataloader_args, "--model",
                                                "--dataloader_args", allow_zero=False)


def cli_score_variants(command, raw_args):
    """CLI interface to score variants
    """
    # Updated argument names:
    # - scoring -> scores
    # - --vcf_path -> --input_vcf, -i
    # - --out_vcf_fpath -> --output_vcf, -o
    # - --output -> -e, --extra_output
    # - remove - -install_req
    # - scoring_kwargs -> score_kwargs
    AVAILABLE_FORMATS = ["tsv", "hdf5", "h5"]
    assert command == "score_variants"
    parser = argparse.ArgumentParser('kipoi postproc {}'.format(command),
                                     description='Predict effect of SNVs using ISM.')
    parser.add_argument('model', help='Model name.', nargs="+")
    parser.add_argument('--source', default=["kipoi"], nargs="+",
                        choices=list(kipoi.config.model_sources().keys()),
                        help='Model source to use. Specified in ~/.kipoi/config.yaml' +
                             " under model_sources. " +
                             "'dir' is an additional source referring to the local folder.")
    parser.add_argument('--dataloader', nargs="+", default=[],
                        help="Dataloader name. If not specified, the model's default" +
                             "DataLoader will be used")
    parser.add_argument('--dataloader_source', nargs="+", default=["kipoi"],
                        help="Dataloader source")
    parser.add_argument('--dataloader_args', nargs="+", default=[],
                        help="Dataloader arguments either as a json string:" +
                             "'{\"arg1\": 1} or as a file path to a json file")
    parser.add_argument('-i', '--input_vcf',
                        help='Input VCF.')
    parser.add_argument('-o', '--output_vcf',
                        help='Output annotated VCF file path.', default=None)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument('-r', '--restriction_bed', default=None,
                        help="Regions for prediction can only be subsets of this bed file")
    parser.add_argument('-e', '--extra_output', required=False,
                        help="Additional output file. File format is inferred from the file path ending" +
                             ". Available file formats are: {0}".format(",".join(AVAILABLE_FORMATS)))
    parser.add_argument('-s', "--scores", default="diff", nargs="+",
                        help="Scoring method to be used. Only scoring methods selected in the model yaml file are"
                             "available except for `diff` which is always available. Select scoring function by the"
                             "`name` tag defined in the model yaml file.")
    parser.add_argument('-k', "--score_kwargs", default="", nargs="+",
                        help="JSON definition of the kwargs for the scoring functions selected in --scoring. The "
                             "definiton can either be in JSON in the command line or the path of a .json file. The "
                             "individual JSONs are expected to be supplied in the same order as the labels defined in "
                             "--scoring. If the defaults or no arguments should be used define '{}' for that respective "
                             "scoring method.")
    parser.add_argument('-l', "--seq_length", type=int, nargs="+", default=[],
                        help="Optional parameter: Model input sequence length - necessary if the model does not have a "
                             "pre-defined input sequence length.")
    parser.add_argument('--std_var_id', action="store_true", help="If set then variant IDs in the annotated"
                                                                  " VCF will be replaced with a standardised, unique ID.")

    args = parser.parse_args(raw_args)
    # Make sure all the multi-model arguments like source, dataloader etc. fit together
    _prepare_multi_model_args(args)

    # Check that all the folders exist
    file_exists(args.input_vcf, logger)
    dir_exists(os.path.dirname(args.output_vcf), logger)
    if args.extra_output is not None:
        dir_exists(os.path.dirname(args.extra_output), logger)

        # infer the file format
        args.file_format = args.extra_output.split(".")[-1]
        if args.file_format not in AVAILABLE_FORMATS:
            logger.error("File ending: {0} for file {1} not from {2}".
                         format(args.file_format, args.extra_output, AVAILABLE_FORMATS))
            sys.exit(1)

        if args.file_format in ["hdf5", "h5"]:
            # only if hdf5 output is used
            import deepdish

    if not isinstance(args.scores, list):
        args.scores = [args.scores]

    score_kwargs = []
    if len(args.score_kwargs) > 0:
        score_kwargs = args.score_kwargs
        if len(args.scores) >= 1:
            # Check if all scoring functions should be used:
            if args.scores == ["all"]:
                if len(score_kwargs) >= 1:
                    raise ValueError("`--score_kwargs` cannot be defined in combination will `--scoring all`!")
            else:
                score_kwargs = [parse_json_file_str(el) for el in score_kwargs]
                if not len(args.score_kwargs) == len(score_kwargs):
                    raise ValueError("When defining `--score_kwargs` a JSON representation of arguments (or the "
                                     "path of a file containing them) must be given for every "
                                     "`--scores` function.")

    keep_predictions = args.extra_output is not None

    res = None
    if keep_predictions:
        res = {}

    n_models = len(args.model)

    for model_name, model_source, dataloader, dataloader_source, dataloader_args, seq_length in zip(args.model,
                                                                                                    args.source,
                                                                                                    args.dataloader,
                                                                                                    args.dataloader_source,
                                                                                                    args.dataloader_args,
                                                                                                    args.seq_length):
        model_name_safe = model_name.replace("/", "_")
        output_vcf_model = None
        if args.output_vcf is not None:
            output_vcf_model = args.output_vcf
            # If multiple models are to be analysed then vcfs need renaming.
            if n_models > 1:
                if output_vcf_model.endswith(".vcf"):
                    output_vcf_model = output_vcf_model[:-4]
                output_vcf_model += model_name_safe + ".vcf"

        dataloader_arguments = parse_json_file_str(dataloader_args)

        # --------------------------------------------
        # load model & dataloader
        model = kipoi.get_model(model_name, model_source)

        if dataloader is not None:
            Dl = kipoi.get_dataloader_factory(dataloader, dataloader_source)
        else:
            Dl = model.default_dataloader

        # Load effect prediction related model info
        model_info = kipoi.postprocessing.variant_effects.ModelInfoExtractor(model, Dl)

        if model_info.use_seq_only_rc:
            logger.info('Model SUPPORTS simple reverse complementation of input DNA sequences.')
        else:
            logger.info('Model DOES NOT support simple reverse complementation of input DNA sequences.')

        if output_vcf_model is not None:
            logger.info('Annotated VCF will be written to %s.' % str(output_vcf_model))

        res[model_name_safe] = kipoi.postprocessing.variant_effects.score_variants(model,
                                                                                   dataloader_arguments,
                                                                                   args.input_vcf,
                                                                                   output_vcf_model,
                                                                                   scores=args.scores,
                                                                                   score_kwargs=score_kwargs,
                                                                                   num_workers=args.num_workers,
                                                                                   batch_size=args.batch_size,
                                                                                   seq_length=seq_length,
                                                                                   std_var_id=args.std_var_id,
                                                                                   restriction_bed=args.restriction_bed,
                                                                                   return_predictions=keep_predictions)

    # tabular files
    if keep_predictions:
        if args.file_format in ["tsv"]:
            for model_name in res:
                for i, k in enumerate(res[model_name]):
                    # Remove an old file if it is still there...
                    if i == 0:
                        try:
                            os.unlink(args.extra_output)
                        except Exception:
                            pass
                    with open(args.extra_output, "w") as ofh:
                        ofh.write("KPVEP_%s:%s\n" % (k.upper(), model_name))
                        res[model_name][k].to_csv(args.extra_output, sep="\t", mode="a")

        if args.file_format in ["hdf5", "h5"]:
            deepdish.io.save(args.extra_output, res)

    logger.info('Successfully predicted samples')


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

    regions_file = os.path.realpath(args.regions_file)
    output = os.path.realpath(args.output)
    with cd(model.source_dir):
        if not os.path.exists(regions_file):
            raise Exception("Regions inputs file does not exist: %s" % args.regions_file)

        # Check that all the folders exist
        file_exists(regions_file, logger)
        dir_exists(os.path.dirname(output), logger)

        if args.dataloader is not None:
            Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
        else:
            Dl = model.default_dataloader

    if not isinstance(args.scoring, list):
        args.scoring = [args.scoring]

    dts = get_scoring_fns(model, args.scoring, args.scoring_kwargs)

    # Load effect prediction related model info
    model_info = kipoi.postprocessing.variant_effects.ModelInfoExtractor(model, Dl)
    manual_seq_len = args.seq_length

    # Select the appropriate region generator and vcf or bed file input
    args.file_format = regions_file.split(".")[-1]
    bed_region_file = None
    vcf_region_file = None
    bed_to_region = None
    vcf_to_region = None
    if args.file_format == "vcf" or regions_file.endswith("vcf.gz"):
        vcf_region_file = regions_file
        if model_info.requires_region_definition:
            # Select the SNV-centered region generator
            vcf_to_region = kipoi.postprocessing.variant_effects.SnvCenteredRg(model_info, seq_length=manual_seq_len)
            logger.info('Using variant-centered sequence generation.')
    elif args.file_format == "bed":
        if model_info.requires_region_definition:
            # Select the SNV-centered region generator
            bed_to_region = kipoi.postprocessing.variant_effects.BedOverlappingRg(model_info, seq_length=manual_seq_len)
            logger.info('Using bed-file based sequence generation.')
        bed_region_file = regions_file
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
    mdmm.save_to_file(output)

    logger.info('Successfully generated mutation map data')


def cli_plot_mutation_map(command, raw_args):
    """CLI interface to plot mutation map
    """
    assert command == "plot_mutation_map"
    parser = argparse.ArgumentParser('kipoi postproc {}'.format(command),
                                     description='Predict effect of SNVs using ISM.')
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
    'create_mutation_map': cli_create_mutation_map,
    'plot_mutation_map': cli_plot_mutation_map
}
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='Kipoi model-zoo command line tool',
    usage='''kipoi postproc <command> [-h] ...

    # Available sub-commands:
    score_variants   Score variants with a kipoi model
    create_mutation_map   Calculate variant effect scores for mutation map plotting
    plot_mutation_map   Plot mutation map from data generated in `create_mutation_map`
    
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
