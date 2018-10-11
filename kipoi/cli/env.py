"""Command line interface for kipoi env
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
from sys import platform
import os
import argparse
import subprocess
import kipoi
from kipoi.cli.parser_utils import add_env_args, parse_source_name
from kipoi.specs import Dependencies, DataLoaderImport
from kipoi.sources import list_subcomponents
from kipoi.utils import cd
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _replace_slash(s, replace_with="__"):
    """Conda doesn't suppport slashes. Hence we have
    to replace it with another character.
    """
    return s.replace("/", replace_with)


def get_env_name(model_name, dataloader_name=None, source="kipoi", gpu=False):
    """Create a conda env name

    Args:
      model_name: String or a list of strings
      dataloader_name: String or a list of strings
      source: source name
      gpu: if True, add the gpu string to the environment
    """
    if not isinstance(model_name, list):
        model_name = [model_name]
    assert model_name

    model_name = [_replace_slash(os.path.normpath(m))
                  for m in model_name]

    if gpu:
        gpu_str = "-gpu"
    else:
        gpu_str = ""
    env_name = "{0}{1}-{2}".format(source, gpu_str, ",".join(model_name))

    # Optional: add dataloader
    if dataloader_name is not None:
        # add also the dataloaders to the string
        if not isinstance(dataloader_name, list):
            dataloader_name = [dataloader_name]
        dataloader_name = [_replace_slash(os.path.normpath(d))
                           for d in dataloader_name]
        if len(dataloader_name) != 0 and dataloader_name != model_name:
            env_name += "-DL-{0}".format(",".join(dataloader_name))

    # limit the env name to 110 characters
    if len(env_name) > 110:
        logger.info("Environment name exceeds 110 characters. Limiting it to 110 characters")
        env_name = env_name[:110]
    return env_name


# Website compatibility
conda_env_name = get_env_name


KIPOI_DEPS = Dependencies(pip=["kipoi"])
# TODO - update once kipoi_veff will be on bioconda
VEP_DEPS = Dependencies(conda=["bioconda::pyvcf",
                               "bioconda::cyvcf2",
                               "bioconda::pybedtools",
                               "bioconda::pysam"],
                        pip=["kipoi_veff"])


def merge_deps(models,
               dataloaders=None,
               source="kipoi",
               vep=False,
               gpu=False):
    """Setup the dependencies
    """
    deps = Dependencies()
    for model in models:
        logger.info("Loading model: {0} description".format(model))

        parsed_source, parsed_model = parse_source_name(source, model)

        sub_models = list_subcomponents(parsed_model, parsed_source, "model")
        if len(sub_models) == 0:
            raise ValueError("Model {0} not found in source {1}".format(parsed_model, parsed_source))
        if len(sub_models) > 1:
            logger.info("Found {0} models under the model name: {1}. Merging dependencies for all".
                        format(len(sub_models), parsed_model))

        for sub_model in sub_models:
            model_descr = kipoi.get_model_descr(sub_model, parsed_source)
            deps = deps.merge(model_descr.dependencies)

            # handle the dataloader=None case
            if dataloaders is None or not dataloaders:
                if isinstance(model_descr.default_dataloader, DataLoaderImport):
                    # add also the directory
                    with cd(os.path.dirname(model_descr.path)):
                        dataloader_descr = model_descr.default_dataloader.get()
                else:
                    dataloader = os.path.normpath(os.path.join(sub_model,
                                                               str(model_descr.default_dataloader)))
                    logger.info("Inferred dataloader name: {0} from".format(dataloader) +
                                " the model.")
                    dataloader_descr = kipoi.get_dataloader_descr(dataloader, parsed_source)
                deps = deps.merge(dataloader_descr.dependencies)
    if dataloaders is not None or dataloaders:
        for dataloader in dataloaders:
            parsed_source, parsed_dataloader = parse_source_name(source, dataloader)
            sub_dataloaders = list_subcomponents(parsed_dataloader, parsed_source, "dataloader")
            if len(sub_dataloaders) == 0:
                raise ValueError("Dataloader: {0} not found in source {1}".format(parsed_dataloader,
                                                                                  parsed_source))

            if len(sub_dataloaders) > 1:
                logger.info("Found {0} dataloaders under the dataloader name: {1}. Merging dependencies for all".
                            format(len(sub_dataloaders), parsed_dataloader))
            for sub_dataloader in sub_dataloaders:
                dataloader_descr = kipoi.get_dataloader_descr(sub_dataloader, parsed_source)
                deps = deps.merge(dataloader_descr.dependencies)

    # add Kipoi to the dependencies
    deps = KIPOI_DEPS.merge(deps)

    if vep:
        # add vep dependencies
        logger.info("Adding the vep dependencies")
        deps = VEP_DEPS.merge(deps)

    if gpu:
        logger.info("Using gpu-compatible dependencies")
        deps = deps.gpu()

    if platform == "darwin":
        logger.info("Using osx-type dependencies")
        deps = deps.osx()

    return deps


def export_deps_to_env(deps, env_file=None, env_dir=".", env=None):
    if env_file is None:
        env_file = os.path.join(env_dir, "{env}.yaml".format(env=env))

    logger.info("Environment name: {0}".format(env))
    logger.info("Output env file: {0}".format(env_file))
    if not os.path.exists(os.path.abspath(os.path.dirname(env_file))):
        os.makedirs(os.path.dirname(env_file))
    deps.to_env_file(env, env_file)
    logger.info("Done writing the environment file!")
    return env, env_file


def export_env(models,
               dataloaders=None,
               source='kipoi',
               env_file=None,
               env_dir=".",
               env=None,
               vep=False,
               gpu=False):
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
      gpu: Use gpu-compatible dependencies. Example: instead of using 'tensorflow',
        'tensorflow-gpu' will be used

    Returns:
      env name.
    """

    # specify the default environment
    if env is None:
        env = get_env_name(models, dataloaders, source, gpu=gpu)

    deps = merge_deps(models=models,
                      dataloaders=dataloaders,
                      source=source,
                      vep=vep,
                      gpu=gpu)
    return export_deps_to_env(deps, env_file=env_file, env_dir=env_dir, env=env)


def cli_export(cmd, raw_args):
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Export the environment.yaml file for a specific model.'
    )
    add_env_args(parser)
    parser.add_argument('-o', '--output', default='environment.yaml',
                        required=True,
                        help="Output file name")
    parser.add_argument('-e', '--env', default=None,
                        help="Environment name")
    args = parser.parse_args(raw_args)
    env, env_file = export_env(args.model,
                               args.dataloader,
                               args.source,
                               env_file=args.output,
                               env=args.env,
                               vep=args.vep,
                               gpu=args.gpu)

    print("Create the environment with:")
    print("conda env create --file {0}".format(env_file))


def cli_create(cmd, raw_args):
    """Create a conda environment for a model
    """
    import uuid
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Create a conda environment for a specific model.'
    )
    add_env_args(parser)
    parser.add_argument('-e', '--env', default=None,
                        help="Special environment name. default: kipoi-<model>[-<dataloader>]")
    parser.add_argument('-t', '--tmpdir', default=None,
                        help="Temporary directory path where to create the conda environment file" +
                        "Defaults to /tmp/kipoi/envfiles/<uuid>/")
    args = parser.parse_args(raw_args)

    # create the tmp dir
    if args.tmpdir is None:
        tmpdir = "/tmp/kipoi/envfiles/" + str(uuid.uuid4())[:8]
    else:
        tmpdir = args.tmpdir
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    # write the env file
    logger.info("Writing environment file: {0}".format(tmpdir))
    env, env_file = export_env(args.model,
                               args.dataloader,
                               args.source,
                               env_file=None,
                               env_dir=tmpdir,
                               env=args.env,
                               vep=args.vep,
                               gpu=args.gpu)

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
    add_env_args(parser)
    args = parser.parse_args(raw_args)

    deps = merge_deps(models=args.model,
                      dataloaders=args.dataloader,
                      source=args.source,
                      vep=args.vep,
                      gpu=args.gpu)
    deps.install()
    logger.info("Done!")


def cli_name(cmd, raw_args):
    """Show the name of the environment
    """
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Show the name of the environment.'
    )
    add_env_args(parser)
    args = parser.parse_args(raw_args)
    env = get_env_name(args.model, args.dataloader, args.source, gpu=args.gpu)
    print("\nEnvironment name: {0}".format(env))


command_functions = {
    'export': cli_export,
    'create': cli_create,
    'list': cli_list,
    'install': cli_install,
    'name': cli_name,
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
