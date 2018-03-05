#!/usr/bin/env python
"""Main CLI to kipoi

adopted from https://github.com/kundajelab/tf-dragonn/blob/master/tfdragonn/__main__.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from kipoi import cli

import argparse
import sys
import pkg_resources
import logging
import logging.config
logging.config.fileConfig(pkg_resources.resource_filename(__name__, "logging.conf"))
logger = logging.getLogger(__name__)


def not_implemented(command, arg_list):
    print("{0} not implemented yet!".format(command))


command_functions = {
    # using
    'preproc': cli.main.cli_preproc,
    'predict': cli.main.cli_predict,
    'pull': cli.main.cli_pull,
    'ls': cli.ls.cli_ls,
    'info': cli.main.cli_info,
    # further sub-commands
    'postproc': cli.postproc.cli_main,
    'env': cli.env.cli_main,
    # Contribuing
    'test': cli.main.cli_test,
    'test-source': cli.source_test.cli_test_source,
    'init': cli.main.cli_init,

}
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='Kipoi model-zoo command line tool',
    usage='''kipoi <command> [-h] ...

    # Kipoi model-zoo command line tool. Available sub-commands:
    # - using models:
    ls               List all the available models
    info             Print dataloader keyword argument info
    predict          Run the model prediction
    pull             Download the directory associated with the model
    preproc          Run the dataloader and save the results to an hdf5 array
    postproc         Tools for model postprocessing like variant effect prediction
    env              Tools for managing Kipoi conda environments

    # - contribuing models:
    init             Initialize a new Kipoi model
    test             Runs a set of unit-tests for the model
    test-source      Runs a set of unit-tests for many/all models in a source
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
