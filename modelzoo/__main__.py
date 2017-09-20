#!/usr/bin/env python
"""Main CLI to modelzoo

adopted from https://github.com/kundajelab/tf-dragonn/blob/master/tfdragonn/__main__.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modelzoo import pipeline

import argparse
import sys
import logging
_logger = logging.getLogger('model-zoo')


def not_implemented(command, arg_list):
    print("{0} not implemented yet!".format(command))


command_functions = {
    'preproc': pipeline.cli_extract_to_hdf5,
    'predict': pipeline.cli_predict,
    'score_variants': not_implemented,
    'test': pipeline.cli_test,
    'pull': pipeline.cli_pull,
}
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='Kipoi model-zoo command line tool',
    usage='''modelzoo <command> [-h] ...

    Kipoi model-zoo command line tool. Available sub-commands:

    # Using the models
    predict          Run the model prediction.
    score_variants   Run prediction on a list of regions
    pull             Downloads the directory associated with the model
    preproc          Returns an hdf5 array.
    test             Runs a set of unit-tests for the model
    ''')
parser.add_argument('command', help='Subcommand to run; possible commands: {}'.format(commands_str))


def main():
    args = parser.parse_args(sys.argv[1:2])
    if args.command not in command_functions:
        parser.print_help()
        parser.exit(
            status=1,
            message='\nCommand `{}` not found. Possible commands: {}\n'.format(
                args.command, commands_str))
    command_fn = command_functions[args.command]
    command_fn(args.command, sys.argv[2:])


if __name__ == '__main__':
    main()
