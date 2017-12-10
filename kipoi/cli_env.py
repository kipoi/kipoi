"""Command line interface for kipoi env
"""
import kipoi
import sys
import os
import argparse
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from kipoi.cli_parser import add_source, add_model, add_dataloader


def replace_slash(s, replace_with="|"):
    return s.replace("/", replace_with)


def env_parser():
    pass


def export(cmd, raw_args):
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Export the environment.yaml file for a specific model.'
    )
    add_model(parser)
    add_dataloader(parser, with_args=False)
    parser.add_argument('-o', '--output', default='environment.yaml', required=True,
                        help="Output file name")
    parser.add_argument('-e', '--env', default=None,
                        help="Environment name")
    args = parser.parse_args(raw_args)
    logger.info("Loading model: {0} description".format(args.model))
    model_descr = kipoi.get_model_descr(args.model, args.source)

    # handle the dataloader=None case
    if args.dataloader is None:
        args.dataloader = os.path.normpath(os.path.join(args.model,
                                                        model_descr.default_dataloader))
        args.dataloader_source = args.source
        logger.info("Inferred dataloader name: {0} from".format(args.dataloader) +
                    " the model.")

    # specify the default environment
    if args.env is None:
        args.env = "kipoi-{0}-{1}".format(args.model, args.dataloader)
    # normalize the path
    args.env = replace_slash(args.env)

    logger.info("Environment name: {0}".format(args.env))
    logger.info("Output fule: {0}".format(args.output))

    dataloader_descr = kipoi.get_dataloader_descr(args.dataloader, args.dataloader_source)

    deps = model_descr.dependencies.merge(dataloader_descr.dependencies)
    deps.to_env_file(args.env, args.output)
    logger.info("Done!")


def create(cmd, raw_args):
    """Create a conda environment for a model
    """
    # TODO - duplicated with export()
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Create a conda environment for a specific model.'
    )
    add_model(parser)
    add_dataloader(parser, with_args=False)
    parser.add_argument('-e', '--env', default=None,
                        help="Special environment name. default: kipoi-<model>-<dataloader>")
    args = parser.parse_args(raw_args)
    logger.info("Loading model: {0} description".format(args.model))
    model_descr = kipoi.get_model_descr(args.model, args.source)

    # handle the dataloader=None case
    if args.dataloader is None:
        args.dataloader = os.path.normpath(os.path.join(args.model,
                                                        model_descr.default_dataloader))
        args.dataloader_source = args.source
        logger.info("Inferred dataloader name: {0} from".format(args.dataloader) +
                    " the model.")

    # specify the default environment
    if args.env is None:
        args.env = "kipoi-{0}-{1}".format(args.model, args.dataloader)
    # normalize the path
    args.env = replace_slash(args.env)
    logger.info("Environment name: {0}".format(args.env))

    dataloader_descr = kipoi.get_dataloader_descr(args.dataloader, args.dataloader_source)

    deps = model_descr.dependencies.merge(dataloader_descr.dependencies)
    tmp_file = "/tmp/kipoi/envfiles/{0}.yml".format(args.env)
    os.makedirs(os.path.dirname(tmp_file), exist_ok=True)
    logger.info("Writing environment file: {0}".format(tmp_file))
    deps.to_env_file(args.env, tmp_file)
    logger.info("Creating conda env from file")
    kipoi.conda.create_env_from_file(tmp_file)
    logger.info("Done!")


def ls(cmd, raw_args):
    """List all kipoi-induced conda environments
    """
    dtm = kipoi.list_models()
    for m in list(dtm.source.str.cat(dtm.model, sep=":")):
        print(m)


def activate(cmd, raw_args):
    """Activate a conda environment for a particular model
    """
    # TODO - fix this function
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Create a conda environment for a specific model.'
    )
    add_model(parser)
    add_dataloader(parser, with_args=False)
    parser.add_argument('-e', '--env', default=None,
                        help="Special environment name. default: kipoi-<model>-<dataloader>")
    args = parser.parse_args(raw_args)
    logger.info("Loading model: {0} description".format(args.model))
    model_descr = kipoi.get_model_descr(args.model, args.source)

    # handle the dataloader=None case
    if args.dataloader is None:
        args.dataloader = os.path.normpath(os.path.join(args.model,
                                                        model_descr.default_dataloader))
        args.dataloader_source = args.source
        logger.info("Inferred dataloader name: {0} from".format(args.dataloader) +
                    " the model.")

    # specify the default environment
    if args.env is None:
        args.env = "kipoi-{0}-{1}".format(args.model, args.dataloader)
    # normalize the path
    args.env = replace_slash(args.env)
    logger.info("Environment name: {0}".format(args.env))
    # TODO - source activate
    import subprocess
    # TODO - get the environmetn variable
    p = subprocess.Popen("which python", shell=True, executable='/bin/zsh')
    stdout, stderr = p.communicate()
    print(stdout)
    # TODO - get current shell
    p = subprocess.Popen(["/bin/bash", "-i", "-c", "source", "activate", args.env],
                         shell=True,
                         stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()

    # kipoi.conda._call_command("source", ["activate", args.env])


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
