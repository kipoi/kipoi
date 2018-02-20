"""test-source command
"""
import argparse
from colorlog import escape_codes, default_log_colors
import re
import sys
import os
import subprocess as sp
import kipoi
from kipoi.conda import get_kipoi_bin, env_exists, remove_env, _call_command
from kipoi.cli.env import conda_env_name
from kipoi.remote import list_softlink_dependencies
from kipoi.utils import list_files_recursively, read_txt, get_file_path
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def run(cmds, env=None, mask=None, **kwargs):
    """
    Wrapper around subprocess.run()
    Explicitly decodes stdout to avoid UnicodeDecodeErrors that can occur when
    using the `universal_newlines=True` argument in the standard
    subprocess.run.
    Also uses check=True and merges stderr with stdout. If a CalledProcessError
    is raised, the output is decoded.
    Returns the subprocess.CompletedProcess object.
    """
    try:
        p = sp.run(cmds, stdout=sp.PIPE, stderr=sp.STDOUT, check=True, env=env,
                   **kwargs)
        p.stdout = p.stdout.decode(errors='replace')
    except sp.CalledProcessError as e:
        e.stdout = e.stdout.decode(errors='replace')
        print(e)
        # mask command arguments

        def do_mask(arg):
            if mask is None:
                # caller has not considered masking, hide the entire command
                # for security reasons
                return '<hidden>'
            elif mask is False:
                # masking has been deactivated
                return arg
            for m in mask:
                arg = arg.replace(m, '<hidden>')
            return arg
        e.cmd = [do_mask(c) for c in e.cmd]
        logger.error('COMMAND FAILED: %s', ' '.join(e.cmd))
        logger.error('STDOUT+STDERR:\n%s', do_mask(e.stdout))
        raise e
    return p


def modified_files(git_range, source_folder, relative=True):
    """
    Returns files under the models dir that have been modified within the git
    range. Filenames are returned with the `source_folder` included.

    Args:
      git_range : list or tuple of length 1 or 2
          For example, ['00232ffe', '10fab113'], or commonly ['master', 'HEAD']
          or ['master']. If length 2, then the commits are provided to `git diff`
          using the triple-dot syntax, `commit1...commit2`. If length 1, the
          comparison is any changes in the working tree relative to the commit.
      source_folder : str
          Root of the model source/git repo
      relative=True: return the relative path
    """
    assert isinstance(git_range, list)
    cmds = ['git', 'diff', '--name-only'] + git_range

    p = run(cmds, cwd=source_folder)

    modified = [os.path.join(source_folder, m)
                for m in p.stdout.strip().split('\n')]

    # exclude files that were deleted in the git-range
    existing = list(filter(os.path.exists, modified))

    # if the only diff is that files were deleted, we can have ['model/'], so
    # filter on existing *files*
    existing = list(filter(os.path.isfile, existing))
    if relative:
        return [os.path.relpath(f, source_folder)
                for f in existing]
    else:
        return existing


def all_models_to_test(src):
    """Returns a list of models to test

    By default, this method returns all the model. In case a model group has a
    `test_subset.txt` file present in the group directory, then testing is only
    performed for models listed in `test_subset.txt`.

    Args:
      src: Model source
    """
    txt_files = list_files_recursively(src.local_path, "test_subset", "txt")

    exclude = []
    include = []
    for x in txt_files:
        d = os.path.dirname(x)
        exclude += [d]
        include += [os.path.join(d, l)
                    for l in read_txt(os.path.join(src.local_path, x))]

    # try to load every model extra included -- will get tested downstream
    # for m in include:
    #     src.get_model_descr(m)

    models = src.list_models().model
    for excl in exclude:
        models = models[~models.str.startswith(excl)]
    return list(models) + include


def test_model(model_name, source_name, env_name, batch_size):
    """kipoi test ...

    Args:
      model_name (str)
      source_name: source name
    """
    if env_exists(env_name):
        logger.info("Environment {0} exists. Removing it.".format(env_name))
        remove_env(env_name)

    # create the model test environment
    cmd = "kipoi"
    args = ["env", "create",
            "--source", source_name,
            "--env", env_name,
            model_name]
    returncode = _call_command(cmd, args, use_stdout=True)
    assert returncode == 0

    # run the tests in the environment
    cmd = get_kipoi_bin(env_name)
    args = ["test",
            "--batch_size", str(batch_size),
            "--source", source_name,
            model_name]
    returncode, logs = _call_command(cmd, args, use_stdout=True,
                                     return_logs_with_stdout=True)
    assert returncode == 0

    # detect WARNING in the output log
    warn = 0
    for line in logs:
        warn_start = escape_codes[default_log_colors['WARNING']] + \
            'WARNING' + escape_codes['reset']
        if line.startswith(warn_start):
            logger.error("Warning present: {0}".format(line))
            warn += 1
    if warn > 0:
        raise ValueError("{0} warnings were observed for model {1}".
                         format(warn, model_name))


def restrict_models_to_test(all_models, source, git_range):
    """Subset all_models to the ones with changed files

    Args:
      all_models: list of all models to test
      source: used source
      git_range: a tuple of (from, to) master branches

    Inspired by bioconda-utils ... --git-range
    1. check
    """
    modified = modified_files(git_range, source.local_path, relative=True)

    def dependency_modified(model_name, modified_files):
        """Test if the dependency was modified
        """
        def contains_any(x, file_list):
            """Test if the directory x contains any files from file_list
            """
            for y in file_list:
                if y.startswith(x + os.sep) or x == y:
                    return True
            return False
        # get all the dependency directories
        dep_dirs = list_softlink_dependencies(os.path.join(source.local_path,
                                                           model_name),
                                              source.local_path)

        for model_dir in [model_name] + list(dep_dirs):
            if contains_any(model_dir, modified_files):
                return True
        return False

    return [x for x in all_models if dependency_modified(x, modified)]


def rm_env(env_name):
    """Alias for remove_env
    """
    if env_exists(env_name):
        logger.info("Removing environment: {0}".
                    format(env_name))
        remove_env(env_name)


def get_batch_size(cfg, model_name, default=4):
    """Get the default batch size to test
    """
    if cfg is not None and model_name in cfg.test.constraints:
        bs = cfg.test.constraints[model_name].batch_size
        if bs is not None:
            if bs != default:
                logger.info("Using a different batch_size ({0})".
                            format(bs) +
                            "for model {0} than the default ({1})".
                            format(model_name, default))
            return bs
    return default


def cli_test_source(command, raw_args):
    """Runs test on the model
    """
    assert command == "test-source"
    # setup the arg-parsing
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Test models in model source')
    parser.add_argument('source', default="kipoi",
                        help='Which source to test')
    parser.add_argument('--git-range', nargs='+',
                        help='''Git range (e.g. commits or something like
                        "master HEAD" to check commits in HEAD vs master, or just "HEAD" to
                        include uncommitted changes). All models modified within this range will
                        be tested.''')
    parser.add_argument('-n', '--dry_run', action='store_true',
                        help='Dont run model testing')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        help='Batch size')
    parser.add_argument('-x', '--exitfirst', action='store_true',
                        help='exit instantly on first error or failed test.')
    parser.add_argument('-k', default=None,
                        help='only run tests which match the given substring expression')
    parser.add_argument('-c', '--clean_env', action='store_true',
                        help='clean the environment after running.')
    parser.add_argument('--all', action='store_true',
                        help="Test all models in the source")

    args = parser.parse_args(raw_args)
    # --------------------------------------------
    source = kipoi.get_source(args.source)
    all_models = all_models_to_test(source)
    if args.k is not None:
        all_models = [x for x in all_models if re.match(args.k, x)]

    if len(all_models) == 0:
        logger.info("No models found in the source")
        sys.exit(1)
    if args.all:
        test_models = all_models
        logger.info('Testing all models:\n- {0}'.
                    format('\n- '.join(test_models)))
    else:
        test_models = restrict_models_to_test(all_models,
                                              source,
                                              args.git_range)
        if len(test_models) == 0:
            logger.info("No model modified according to git, exiting.")
            sys.exit(0)
        logger.info('{0}/{1} models modified according to git:\n- {2}'.
                    format(len(test_models), len(all_models),
                           '\n- '.join(test_models)))

    # Parse the repo config
    cfg_path = get_file_path(source.local_path, "config",
                             extensions=[".yml", ".yaml"],
                             raise_err=False)
    if cfg_path is not None:
        cfg = kipoi.components.SourceConfig.load(cfg_path, append_path=False)
        logger.info("Found config {0}:\n{1}".format(cfg_path, cfg))
    else:
        cfg = None

    if args.dry_run:
        logger.info("-n/--dry_run enabled. Skipping model testing and exiting.")
        sys.exit(0)

    logger.info("Running {0} tests..".format(len(test_models)))
    failed_models = []
    for i in range(len(test_models)):
        m = test_models[i]
        print('-' * 20)
        print("{0}/{1} - model: {2}".format(i + 1,
                                            len(test_models),
                                            m))
        print('-' * 20)
        try:
            env_name = conda_env_name(m, m, args.source)
            env_name = "test-" + env_name  # prepend "test-"
            test_model(m, args.source, env_name,
                       get_batch_size(cfg, m, args.batch_size))
        except Exception as e:
            logger.error("Model {0} failed: {1}".format(m, e))
            failed_models += [m]
            if args.exitfirst:
                if args.clean_env:
                    rm_env(env_name)
                sys.exit(1)
        finally:
            if args.clean_env:
                rm_env(env_name)
    print('-' * 40)
    if failed_models:
        logger.error("{0}/{1} tests failed for models:\n- {2}".
                     format(len(failed_models),
                            len(test_models),
                            "\n- ".join(failed_models)))
        sys.exit(1)

    logger.info('All tests ({0}) passed'.format(len(test_models)))
