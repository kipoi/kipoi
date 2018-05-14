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



def cli_score_variants_DEP(command, raw_args):
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

        dts = get_scoring_fns(model, args.scoring, args.scoring_kwargs)

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


def cli_score_variants(command, raw_args):
    """CLI interface to score variants
    """
    AVAILABLE_FORMATS = ["tsv", "hdf5", "h5"]
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

    scoring_kwargs = []
    if len(args.scoring_kwargs) >0:
        scoring_kwargs = args.scoring_kwargs
        if len(args.scoring) >= 1:
            # Check if all scoring functions should be used:
            if args.scoring == ["all"]:
                if len(scoring_kwargs) >= 1:
                    raise ValueError("`--scoring_kwargs` cannot be defined in combination will `--scoring all`!")
            else:
                scoring_kwargs = [parse_json_file_str(el) for el in scoring_kwargs]
                if not len(args.scoring_kwargs) == len(scoring_kwargs):
                    raise ValueError("When defining `--scoring_kwargs` a JSON representation of arguments (or the "
                                     "path of a file containing them) must be given for every "
                                     "`--scoring` function.")

    # Load effect prediction related model info
    model_info = kipoi.postprocessing.variant_effects.ModelInfoExtractor(model, Dl)

    if model_info.use_seq_only_rc:
        logger.info('Model SUPPORTS simple reverse complementation of input DNA sequences.')
    else:
        logger.info('Model DOES NOT support simple reverse complementation of input DNA sequences.')

    if out_vcf_fpath is not None:
        logger.info('Annotated VCF will be written to %s.' % str(out_vcf_fpath))

    keep_predictions = args.output is not None

    res = kipoi.postprocessing.variant_effects.score_variants(model,
                                                              dataloader_arguments,
                                                              args.vcf_path,
                                                              out_vcf_fpath,
                                                              scores=scoring_kwargs,
                                                              score_kwargs=None,
                                                              num_workers=0,
                                                              batch_size=32,
                                                              source='kipoi',
                                                              seq_length=args.seq_length,
                                                              restriction_bed=args.restriction_bed,
                                                              return_predictions=keep_predictions)


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
