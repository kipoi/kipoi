"""Command line interface for kipoi env
"""
from __future__ import absolute_import
from __future__ import print_function
from io import open

import argparse
import logging
import os
import subprocess
import time
from builtins import input
from sys import platform

import yaml

import kipoi
import kipoi_conda
from kipoi.cli.parser_utils import add_env_args, parse_source_name
from kipoi import env_db
from kipoi.env_db import get_model_env_db
from kipoi.sources import list_subcomponents, list_models_by_group
from kipoi.specs import Dependencies, DataLoaderImport
from kipoi_utils.utils import cd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
SPECIAL_ENV_PREFIX = "shared/envs/"


def _replace_slash(s, replace_with="__"):
    """Conda doesn't suppport slashes. Hence we have
    to replace it with another character.
    """
    return s.replace("/", replace_with)


def get_env_name(model_name, dataloader_name=None, source="kipoi", gpu=False):
    """Create a conda env name

    # Arguments
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

# constant dependencies
KIPOI_DEPS = Dependencies(pip=["kipoi"])
# TODO - update once kipoi_veff will be on bioconda
VEP_DEPS = Dependencies(conda=["bioconda::pyvcf",
                               "bioconda::cyvcf2",
                               "bioconda::pybedtools",
                               "bioconda::pysam"],
                        pip=["kipoi_veff"])
INTERPRET_DEPS = Dependencies(pip=["kipoi_interpret"])

# Hard-code kipoi-seq dataloaders
KIPOISEQ_DEPS = Dependencies(conda=['bioconda::pybedtools', 'bioconda::pyfaidx', 'numpy', 'pandas'], pip=['kipoiseq'])


def split_models_special_envs(models):
    special_envs = []  # handcrafted environments
    only_models = []  # actual models excluding handcrafted environments
    for model in models:
        if SPECIAL_ENV_PREFIX in model:
            special_envs.append(model)
        else:
            only_models.append(model)
    return special_envs, only_models


def merge_deps(models,
               dataloaders=None,
               source="kipoi",
               vep=False,
               interpret=False,
               gpu=False):
    """Setup the dependencies
    """

    special_envs, only_models = split_models_special_envs(models)
    deps = Dependencies()

    # Treat the handcrafted environments differently
    for special_env in special_envs:
        from related import from_yaml
        logger.info("Loading environment definition: {0}".format(special_env))

        # Load and merge the handcrafted deps.
        yaml_path = os.path.join(kipoi.get_source(source).local_path, special_env + ".yaml")

        if not os.path.exists(yaml_path):
            raise ValueError("Environment definition file {0} not found in source {1}".format(yaml_path, source))

        with open(yaml_path, "r", encoding="utf-8") as fh:
            special_env_deps = Dependencies.from_env_dict(from_yaml(fh))
        deps = deps.merge(special_env_deps)

    for model in only_models:
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
            model_dir = kipoi.get_source(parsed_source).get_model_dir(sub_model)
            deps = deps.merge(model_descr.dependencies)

            # handle the dataloader=None case
            if dataloaders is None or not dataloaders:
                if isinstance(model_descr.default_dataloader, DataLoaderImport):
                    # dataloader specified by the import
                    deps = deps.merge(model_descr.default_dataloader.dependencies)
                    if model_descr.default_dataloader.parse_dependencies:
                        # add dependencies specified in the yaml file
                        # load from the dataloader description if you can
                        try:
                            with cd(model_dir):
                                dataloader_descr = model_descr.default_dataloader.get()
                            deps = deps.merge(dataloader_descr.dependencies)
                        except ImportError as e:
                            # package providing the dataloader is not installed yet
                            if model_descr.default_dataloader.defined_as.startswith("kipoiseq."):
                                logger.info(
                                    "kipoiseq not installed. Using default kipoiseq dependencies for the dataloader: {}"
                                    .format(model_descr.default_dataloader.defined_as))
                                deps = deps.merge(KIPOISEQ_DEPS)
                            else:
                                logger.warning("Unable to extract dataloader description. "
                                            "Make sure the package containing the dataloader `{}` is installed".
                                            format(model_descr.default_dataloader.defined_as))
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

    if interpret:
        # add vep dependencies
        logger.info("Adding the interpret dependencies")
        deps = INTERPRET_DEPS.merge(deps)

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
               interpret=False,
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
                      interpret=interpret,
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
                               interpret=args.interpret,
                               gpu=args.gpu)

    print("Create the environment with:")
    print("conda env create --file {0}".format(env_file))


def delete_envs(to_delete):
    db = get_model_env_db()
    for e in to_delete:
        try:
            kipoi_conda.remove_env(e.create_args.env)
            db.remove(e)
            db.save()
        except Exception as err:
            logger.warning("Conda delete of environment {0} failed with error: {1}. This environment entry was not "
                        "removed from the database.".format(e.create_args.env, str(err)))


def _env_db_model_name(source, model):
    ret = model
    if source != "kipoi":
        source_path = kipoi.get_source(source).local_path
        ret = os.path.join(source_path, model)
    return ret

def get_envs_by_model(models, source, only_most_recent=True, only_valid=False):
    if isinstance(models, str):
        models = [models]

    source_path = kipoi.get_source(source).local_path
    entries = []
    db = env_db.get_model_env_db()
    for m in models:
        res = db.get_entry_by_model(_env_db_model_name(source, m), only_most_recent=only_most_recent,
                                    only_valid=only_valid)
        if only_most_recent:
            entries.append(res)
        else:
            entries.extend(res)
    entries = [e for e in entries if e is not None]
    return entries


def generate_env_db_entry(args, args_env_overload=None):
    from collections import OrderedDict
    from kipoi.env_db import EnvDbEntry
    from kipoi_conda import get_conda_version

    special_envs, only_models = split_models_special_envs(args.model)

    sub_models = []
    for model in only_models:
        parsed_source, parsed_model = parse_source_name(args.source, model)
        models = list_subcomponents(parsed_model, parsed_source, "model")
        sub_models.extend([_env_db_model_name(parsed_source, m) for m in models])

    if len(special_envs) != 0:
        # for the special envs load the corresponding models:
        for special_env in special_envs:
            special_env_folder = "/".join(special_env.rstrip("/").split("/")[:-1])
            source_path = kipoi.get_source(args.source).local_path
            with open(os.path.join(source_path, special_env_folder, "models.yaml"), "r", encoding="utf-8") as fh:
                special_env_models = yaml.load(fh)
            # extend the sub_models by all the submodels covered by the handcrafted environments (special_envs)
            # Those models **always** refer to the kipoi source
            for model_group_name in special_env_models[os.path.basename(special_env)]:
                models = list_subcomponents(model_group_name, "kipoi", "model")
                sub_models.extend([_env_db_model_name("kipoi", m) for m in models])

    entry = EnvDbEntry(
        conda_version=get_conda_version(),
        kipoi_version=kipoi.__version__,
        timestamp=time.time(),
        compatible_models=sub_models,
        create_args=OrderedDict(args._get_kwargs())
    )
    if args_env_overload is not None:
        entry.create_args.env = args_env_overload
    return entry


def cli_create(cmd, raw_args):
    """Create a conda environment for a model
    """
    from kipoi_conda import get_kipoi_bin
    import uuid
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Create a conda environment for a specific model.'
    )
    add_env_args(parser)
    parser.add_argument('-e', '--env', default=None,
                        help="Special environment name. default: kipoi-<model>[-<dataloader>]")
    parser.add_argument('--dry-run', action='store_true',
                        help="Don't actually create the environment")
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

    if args.model == ['all']:
        from kipoi.cli.source_test import get_common_env
        src = kipoi.get_source(args.source)
        model_envs = yaml.load(open(os.path.join(src.local_path, SPECIAL_ENV_PREFIX, "models.yaml")))

        # TODO - test this by mocking up the CLI command execution

        # setup the args for all the models
        df = kipoi.list_models()
        dfg = list_models_by_group(df, "")
        for model_group in dfg.group.unique().tolist():
            existing_envs = get_envs_by_model(model_group, args.source, only_valid=True)
            if existing_envs or existing_envs is None:
                # No need to create the environment
                existing_envs_str = "\n".join([e.create_args.env for e in existing_envs])
                logger.info("Environment for {} already exists ({}). Skipping installation".format(model_group, existing_envs_str))
                continue

            logger.info("Environment doesn't exists for model group {}. Installing it".format(model_group))

            # Figure out which <model> to use for installation
            common_env = get_common_env(model_group, model_envs)
            if common_env is not None:
                # common environment exists for the model. Use it
                logger.info("Using common environment: {}".format(common_env))
                model_group = os.path.join(SPECIAL_ENV_PREFIX, common_env)

            # Run cli_create
            def optional_replace(x, ref, alt):
                if x == ref:
                    return alt
                else:
                    return x
            new_raw_args = [optional_replace(x, 'all', model_group)
                            for x in raw_args if x is not None]
            cli_create(cmd, new_raw_args)
        logger.info("Done installing all environments!")
        return None

    env, env_file = export_env(args.model,
                               args.dataloader,
                               args.source,
                               env_file=None,
                               env_dir=tmpdir,
                               env=args.env,
                               vep=args.vep,
                               interpret=args.interpret,
                               gpu=args.gpu)

    if not args.dry_run:
        env_db_entry = generate_env_db_entry(args, args_env_overload=env)
        envdb = get_model_env_db()
        envdb.append(env_db_entry)
        envdb.save()

        # setup the conda env from file
        logger.info("Creating conda env from file: {0}".format(env_file))
        kipoi_conda.create_env_from_file(env_file)
        env_db_entry.successful = True

        # env is environment name
        env_db_entry.cli_path = get_kipoi_bin(env)
        get_model_env_db().save()
        logger.info("Done!")
        print("\nActivate the environment via:")
        print("source activate {0}".format(env))
    else:
        print("Dry run. Conda file path: {}".format(env_file))


def confirm(message):
    answer = ""
    while answer not in ["y", "n"]:
        answer = input(message + " [Y/N]? ").lower()
    return answer == "y"


def ask_and_delete_envs(to_delete, args):
    db = get_model_env_db()
    del_only_db = False
    if "db" in dir(args) and args.db:
        del_only_db = True

    if not del_only_db:
        warn_msg = "Are you sure you want to delete the following environments:\n"
    else:
        warn_msg = "Are you sure you want to delete the following environments from the database ONLY:\n"
    warn_msg += "\n".join(["{0} ({1})".format(e.create_args.env, e.cli_path) for e in to_delete])
    warn_msg += "\n"

    if args.yes or confirm(warn_msg):
        if not del_only_db:
            delete_envs(to_delete)
        else:
            for e in to_delete:
                db.remove(e)
                db.save()


def print_env_names(entries):
    if len(entries) != 0:
        print("\n".join([e.create_args.env for e in entries]))


def print_env_cli_paths(entries):
    if len(entries) != 0:
        print("\n".join([e.cli_path for e in entries]))


def cli_get(cmd, raw_args):
    """Print a conda environment name for a model
    """
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Print conda environment name for specific model.'
    )
    add_env_args(parser)
    parser.add_argument('-a', '--all', action='store_true',
                        help="If set all environments compatible with this model will be printed!")
    args = parser.parse_args(raw_args)

    entries = get_envs_by_model(args.model, args.source, only_most_recent=not args.all, only_valid=True)

    print_env_names(entries)


def cli_get_cli(cmd, raw_args):
    """Print a kipoi cli path for a model
    """
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Print kipoi cli path for specific model.'
    )
    add_env_args(parser)
    parser.add_argument('-a', '--all', action='store_true',
                        help="If set all environments compatible with this model will be printed!")
    args = parser.parse_args(raw_args)

    entries = get_envs_by_model(args.model, args.source, only_most_recent=not args.all, only_valid=True)

    print_env_cli_paths(entries)


def cli_cleanup(cmd, raw_args):
    """Remove all environments that have failed during setup. Or remove all environments
    """
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Clean-up conda environments.'
    )
    parser.add_argument('-a', '--all', action='store_true',
                        help="If set all environments will be removed!")
    parser.add_argument('-d', '--db', action='store_true',
                        help="Clean-up DB only. Make sure that you run this only if `kipoi env cleanup` has been "
                             "executed first. Running this will remove environments only from the kipoi database and "
                             "will NOT attempt to remove conda environments")
    parser.add_argument('-y', '--yes', action='store_true',
                        help="If set then do NOT ask before deleting environments.")
    args = parser.parse_args(raw_args)
    db = get_model_env_db()

    if not args.all:
        to_delete = db.get_all_unfinished()
    else:
        to_delete = db.get_all()

    if len(to_delete) != 0:
        ask_and_delete_envs(to_delete, args)
    else:
        logger.info("Nothing to clean up!")

    logger.info("Done!")


def cli_remove(cmd, raw_args):
    """Remove a conda environment for a model
    """
    parser = argparse.ArgumentParser(
        'kipoi env {}'.format(cmd),
        description='Remove environment compatible with this model. If `--all` is not set then only remove the most '
                    'recently generated environment compatible with this model.'
    )
    add_env_args(parser)
    parser.add_argument('-a', '--all', action='store_true',
                        help="Remove all environments every created that are compatible with this model")
    parser.add_argument('-y', '--yes', action='store_true',
                        help="If set then do NOT ask before deleting environments.")
    args = parser.parse_args(raw_args)

    to_delete = get_envs_by_model(args.model, args.source, only_most_recent=not args.all)

    if len(to_delete) != 0:
        ask_and_delete_envs(to_delete, args)
    else:
        logger.info("Nothing to remove!")

    logger.info("Done!")


print_valid_env_names = print_env_names
print_invalid_env_names = print_env_names


def cli_list(cmd, raw_args):
    """List all kipoi-induced conda environments
    """
    entries = get_model_env_db().get_all(only_valid=True)
    if len(entries) != 0:
        print("# Functional kipoi environments:")
        print_valid_env_names(entries)

    invalid_entries = get_model_env_db().get_all_unfinished()
    if len(invalid_entries) != 0:
        print("# Non-Functional kipoi environments:")
        print_invalid_env_names(invalid_entries)

    print("# Conda environments starting with kipoi:")
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
                      interpret=args.interpret,
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
    'cleanup': cli_cleanup,
    'remove': cli_remove,
    'list': cli_list,
    'get': cli_get,
    'get_cli': cli_get_cli,
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
    cleanup      Remove environments that failed during installation
    remove       Remove environment compatible with a model
    list         List all kipoi-induced conda environments
    get          Get environment name for given model
    get_cli      Get path to Kipoi CLI for a given model
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
