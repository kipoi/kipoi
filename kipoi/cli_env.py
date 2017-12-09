"""Command line interface for kipoi env
"""
import sys
import argparse
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from kipoi.pipeline import add_source, add_model, add_dataloader


def env_parser():
    pass


def export(cmd, raw_args):
    parser = argparse.ArgumentParser('kipoi env {}'.format(cmd),
                                     description='Export the environment.yaml file for a specific model.')
    add_model(parser)
    add_dataloader(parser, with_args=False)
    args = parser.parse_args(raw_args)
    # TODO - finish here

    pass


def create(cmd, raw_args):
    """Create a conda environment for a model
    """
    pass


def ls(cmd, raw_args):
    """List all kipoi-induced conda environments
    """
    pass


def activate(cmd, raw_args):
    """Activate a conda environment for a particular model
    """
    pass


command_functions = {
    'export': export,
    'create': create,
    'ls': ls,
    'activate': activate,
}
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='Kipoi model-zoo command line tool',
    usage='''kipoi env <command> [-h] ...

    Available sub-commands:

    export       Export the environment.yaml file for a specific model
    create       Create a conda environment for a model
    ls           List all kipoi-induced conda environments
    activate     Activate a conda environment for a particular model
    ''')
parser.add_argument('command', help='Subcommand to run; possible commands: {}'.format(commands_str))


def main(command, raw_args):
    args = parser.parse_args(raw_args[0:1])
    if args.command not in command_functions:
        parser.print_help()
        parser.exit(
            status=1,
            message='\nCommand `{}` not found. Possible commands: {}\n'.format(
                args.command, commands_str))
    command_fn = command_functions[args.command]
    command_fn(args.command, raw_args[1:])
