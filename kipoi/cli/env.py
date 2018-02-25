"""Command line interface for kipoi env
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import subprocess
import kipoi
from kipoi.cli.parser_utils import add_model, add_dataloader, add_vep
from kipoi.components import Dependencies
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def replace_slash(s, replace_with="__"):
    """Conda doesn't suppport slashes. Hence we have
    to replace it with another character.
    """
    return s.replace("/", replace_with)


def conda_env_name(model_name, dataloader_name, source):
    """Create a conda env name
    """
    model_name = os.path.normpath(model_name)
    dataloader_name = os.path.normpath(dataloader_name)

    if dataloader_name == model_name:
        return "{0}-{1}".format(source, replace_slash(model_name))
    else:
        return "{0}-{1}-{2}".format(source,
                                    replace_slash(model_name),
                                    replace_slash(dataloader_name))


KIPOI_DEPS = Dependencies(pip=["kipoi"])
VEP_DEPS = Dependencies(conda=["bioconda::pyvcf",
                               "bioconda::cyvcf2",
                               "bioconda::pybedtools",
                               "bioconda::pysam"],
                        pip=["intervaltree"]
                        )


def export_env(model, source, dataloader=None, dataloader_source="kipoi",
               env_file=None,
               env_dir=".",
               env=None,
               vep=False):
    """Write a conda environment file. Helper function for the cli_export and cli_create.

    Args:
      model: model name
      source: model source name
      dataloader: dataloader name
      dataloader_source: source for the dataloader
      env_file: Output environment file path (directory or yaml file)
      env_dir: Becomes relevant when env_file is None. Then the env_file is inferred
        from env and env_dir
      env: env name for the environment. If None, it will be automatically inferred.
      vep: Add variant effect prediction dependencies

    Returns:
      env name.
    """
    logger.info("Loading model: {0} description".format(model))
    model_descr = kipoi.get_model_descr(model, source)

    # handle the dataloader=None case
    if dataloader is None:
        dataloader = os.path.normpath(os.path.join(model,
                                                   model_descr.default_dataloader))
        dataloader_source = source
        logger.info("Inferred dataloader name: {0} from".format(dataloader) +
                    " the model.")

    # specify the default environment
    if env is None:
        env = conda_env_name(model, dataloader, source)

    if env_file is None:
        env_file = os.path.join(env_dir, "{env}.yaml".format(env=env))

    logger.info("Environment name: {0}".format(env))
    logger.info("Output env file: {0}".format(env_file))

    dataloader_descr = kipoi.get_dataloader_descr(dataloader, dataloader_source)

    deps = model_descr.dependencies.merge(dataloader_descr.dependencies)

    # add Kipoi to the dependencies
    deps = KIPOI_DEPS.merge(deps)

    if vep:
        # add vep dependencies
        deps = VEP_DEPS.merge(deps)

    if not os.path.exists(os.path.dirname(env_file)):
        os.makedirs(os.path.dirname(env_file))
    deps.to_env_file(env, env_file)
    logger.info("Done writing the environment file!")
    return env, env_file


def cli_export(cmd, raw_args):
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Export the environment.yaml file for a specific model.'
    )
    add_model(parser)
    add_dataloader(parser, with_args=False)
    add_vep(parser)
    parser.add_argument('-o', '--output', default='environment.yaml', required=True,
                        help="Output file name")
    parser.add_argument('-e', '--env', default=None,
                        help="Environment name")
    args = parser.parse_args(raw_args)
    env, env_file = export_env(args.model,
                               args.source,
                               args.dataloader,
                               args.dataloader_source,
                               env_file=args.output,
                               env=args.env,
                               vep=args.vep)

    print("Create the environment with:")
    print("conda env create --file {0}".format(env_file))


def cli_create(cmd, raw_args):
    """Create a conda environment for a model
    """
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Create a conda environment for a specific model.'
    )
    add_model(parser)
    add_dataloader(parser, with_args=False)
    add_vep(parser)
    parser.add_argument('-e', '--env', default=None,
                        help="Special environment name. default: kipoi-<model>[-<dataloader>]")
    args = parser.parse_args(raw_args)

    # create the tmp dir
    tmpdir = "/tmp/kipoi/envfiles"
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    # write the env file
    logger.info("Writing environment file: {0}".format(tmpdir))
    env, env_file = export_env(args.model,
                               args.source,
                               args.dataloader,
                               args.dataloader_source,
                               env_file=None,
                               env_dir=tmpdir,
                               env=args.env,
                               vep=args.vep)

    # setup the conda env from file
    logger.info("Creating conda env from file: {0}".format(env_file))
    kipoi.conda.create_env_from_file(env_file)
    logger.info("Done!")
    print("\nActivate the environment via:")
    print("source activate {0}".format(env))


def cli_list(cmd, raw_args):
    """List all kipoi-induced conda environments
    """
    print("# Kipoi environments:")
    subprocess.call("conda env list | grep ^kipoi | cut -f 1 -d ' '", shell=True)


def cli_install(cmd, raw_args):
    """Install the required packages for the model
    """
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Install all the dependencies for a model into the current conda environment.'
    )
    add_model(parser)
    add_dataloader(parser, with_args=False)
    add_vep(parser)
    args = parser.parse_args(raw_args)

    if args.dataloader is None:
        kipoi.pipeline.install_model_requirements(args.model,
                                                  args.source,
                                                  and_dataloaders=True)
    else:
        kipoi.pipeline.install_model_requirements(args.model,
                                                  args.source,
                                                  and_dataloaders=False)
        kipoi.pipeline.install_dataloader_requirements(args.dataloader, args.source)
    if args.vep:
        # install also the vep dependencies
        VEP_DEPS.install()
    logger.info("Done!")


command_functions = {
    'export': cli_export,
    'create': cli_create,
    'list': cli_list,
    'install': cli_install,
}
commands_str = ', '.join(command_functions.keys())

parser = argparse.ArgumentParser(
    description='Kipoi model-zoo command line tool',
    usage='''kipoi env <command> [-h] ...

    # Available sub-commands:
    export       Export the environment.yaml file for a specific model
    create       Create a conda environment for a model
    list         List all kipoi-induced conda environments
    install      Install all the dependencies for a model into the current conda environment
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
