#!/usr/bin/env python
"""Main CLI to kipoi

adopted from https://github.com/kundajelab/tf-dragonn/blob/master/tfdragonn/__main__.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import kipoi
import kipoi_utils
from kipoi import cli

import argparse
import sys
import pkg_resources
import logging
import logging.config
logging.config.fileConfig(pkg_resources.resource_filename(__name__, "logging.conf"))
logger = logging.getLogger(__name__)


# def not_implemented(command, arg_list):
#     print("{0} not implemented yet!".format(command))


command_functions = {
    # using
    'preproc': cli.main.cli_preproc,
    'predict': cli.main.cli_predict,
    'pull': cli.main.cli_pull,
    'ls': cli.main.cli_ls,
    'list_plugins': cli.main.cli_list_plugins,
    'info': cli.main.cli_info,
    # further sub-commands
    'env': cli.env.cli_main,
    # Contributing
    'test': cli.main.cli_test,
    "get-example": cli.main.cli_get_example,
    'test-source': cli.source_test.cli_test_source,
    'init': cli.main.cli_init,
}
command_functions = kipoi_utils.utils.merge_dicts(command_functions,
                                            kipoi.plugin.get_plugin_cli_fns())
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='Kipoi model-zoo command line tool',
    usage='''kipoi <command> [-h] ...

    # Kipoi model-zoo command line tool. Available sub-commands:
    # - using models:
    ls               List all the available models
    list_plugins     List all the available plugins
    info             Print dataloader keyword argument info
    get-example      Download example files
    predict          Run the model prediction
    pull             Download the directory associated with the model
    preproc          Run the dataloader and save the results to an hdf5 array
    env              Tools for managing Kipoi conda environments

    # - contributing models:
    init             Initialize a new Kipoi model
    test             Runs a set of unit-tests for the model
    test-source      Runs a set of unit-tests for many/all models in a source
    ''' + kipoi.plugin.get_plugin_help())
parser.add_argument('command', help='Subcommand to run; possible commands: {}'.format(commands_str))


postproc_cmd_map = {"score_variants": "veff",
                    "create_mutation_map": "veff",
                    "plot_mutation_map": "veff",
                    "grad": "interepret",
                    "gr_inp_to_file": "interepret"}


def main():
    args = parser.parse_args(sys.argv[1:2])
    if args.command != "postproc" and args.command not in command_functions:
        parser.print_help()
        parser.exit(
            status=1,
            message='\nCommand `{}` not found. Possible commands: {}\n'.format(
                args.command, commands_str))
    # backwards compatibilty
    if args.command == "postproc":
        logger.warning("`kipoi postproc` has been deprecated. Please use kipoi <plugin> ...: {}".
                    format(kipoi.plugin.get_plugin_help()))
        if len(sys.argv) == 2:
            logger.error("Use - kipoi <plugin> <command>.")
            sys.exit(1)
        elif sys.argv[2] in postproc_cmd_map:
            args.command = postproc_cmd_map[sys.argv[2]]
        else:
            logger.error("Unable to map kipoi postproc <command> to kipoi <plugin> <command>")
            sys.exit(1)

    # check if the user used the plugin commands
    if kipoi.plugin.is_plugin("kipoi_" + args.command):
        if not kipoi.plugin.is_installed("kipoi_" + args.command):
            logger.error("Plugin {} not installed. Install with `pip install kipoi_{}`".
                         format(args.command, args.command))
            sys.exit(1)
    command_fn = command_functions[args.command]
    command_fn(args.command, sys.argv[2:])


if __name__ == '__main__':
    main()
