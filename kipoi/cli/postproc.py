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
import deepdish
import logging
import pybedtools as pb
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# TODO - --output is not always required
def cli_score_variants(command, raw_args):
    """CLI interface to predict
    """
    scoring_options = {
        "logit": kipoi.variant_effects.Logit,
        "diff": kipoi.variant_effects.Diff,
        "deepsea_scr": kipoi.variant_effects.DeepSEA_effect
    }
    assert command == "score_variants"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
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
    parser.add_argument('-f', '--file_format', default="tsv",
                        choices=["tsv", "hdf5"],
                        help='File format.')
    parser.add_argument('-r', '--restriction_bed', default = None,
                        help="Regions for prediction can only be subsets of this bed file")
    parser.add_argument('-o', '--output', required=False,
                        help="Output hdf5 file")
    parser.add_argument('-s', "--scoring", choices=list(scoring_options.keys()), default="diff", nargs="+")
    args = parser.parse_args(raw_args)

    # extract args for kipoi.variant_effects.predict_snvs
    vcf_path = args.vcf_path
    out_vcf_fpath = args.out_vcf_fpath
    dataloader_arguments = parse_json_file_str(args.dataloader_args)

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

    if len(args.scoring) >= 1:
        dts = {}
        for k in args.scoring:
            dts[k] = scoring_options[k]("absmax")
    else:
        raise Exception("No scoring method was chosen!")

    # Load effect prediction related model info
    model_info = kipoi.postprocessing.ModelInfoExtractor(model, Dl)

    # Select the appropriate region generator
    if args.restriction_bed is not None:
        # Select the restricted SNV-centered region generator
        pbd = pb.BedTool(args.restriction_bed)
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

    if model_info.supports_simple_rc:
        logger.info('Model SUPPORTS simple reverse complementation of input DNA sequences.')
    else:
        logger.info('Model DOES NOT support simple reverse complementation of input DNA sequences.')

    # Get a vcf output writer if needed
    if out_vcf_fpath is not None:
        logger.info('Annotated VCF will be written to %s.'%str(out_vcf_fpath))
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
        vcf_to_region = vcf_to_region,
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
                    except:
                        pass
                with open(args.output, "w") as ofh:
                    ofh.write("KPVEP_%s\n" % k.upper())
                    res[k].to_csv(args.output, sep="\t", mode="a")
    
        if args.file_format == "hdf5":
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
