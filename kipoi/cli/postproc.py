"""Postprocessing CLI
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import kipoi
from kipoi.cli.parser_utils import add_model, add_dataloader
from kipoi.utils import parse_json_file_str, cd
import deepdish
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# TODO - --output is not always required
def cli_score_variants(command, raw_args):
    """CLI interface to predict
    """
    assert command == "score_variants"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Predict effect of SNVs using ISM.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('-v', '--vcf_path',
                        help='Input VCF.')
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
    parser.add_argument('-o', '--output', required=False,
                        help="Output hdf5 file")
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

    res = kipoi.variant_effects.predict_snvs(
        model,
        vcf_path,
        dataloader=Dl,
        batch_size=args.batch_size,
        dataloader_args=dataloader_arguments,
        evaluation_function_kwargs={"diff_type": "diff"},
        out_vcf_fpath=out_vcf_fpath
    )

    # tabular files
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
