"""Useful functions for the Singularity containers

TODO:
- [x] figure out how to mount in other file-systems
  -B dir1,dir2

Put to release notes:
`conda install -c bioconda singularity`

OR

`conda install -c conda-forge singularity`

"""
from __future__ import absolute_import
from __future__ import print_function

import six
import os
from kipoi_utils.utils import unique_list, makedir_exist_ok, is_subdir
from kipoi_conda import _call_command
import subprocess
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Python wrapper for the Singularity CLI
# def assert_installed():
#     """Make sure singularity is installed
#     """
#     pass


def singularity_pull(remote_path, local_path):
    """Run `singularity pull`

    Args:
      remote_path: singularity remote path. Example: shub://kipoi/models:latest
      local_path: local file path to the ".sif" file
    """
    makedir_exist_ok(os.path.dirname(local_path))
    if os.path.exists(local_path):
        logger.info("Container file {} already exists. Skipping `singularity pull`".
                    format(local_path))
    else:
        if os.environ.get('SINGULARITY_CACHEDIR'):
            downloaded_path = os.path.join(os.environ.get('SINGULARITY_CACHEDIR'),
                                           os.path.basename(local_path))
            pull_dir = os.path.dirname(downloaded_path)
            logger.info("SINGULARITY_CACHEDIR is set to {}".
                        format(os.environ.get('SINGULARITY_CACHEDIR')))
            if os.path.exists(downloaded_path):
                logger.info("Container file {} already exists. Skipping `singularity pull` and softlinking it".
                            format(downloaded_path))
                if os.path.islink(local_path):
                    logger.info("Softlink {} already exists. Removing it".format(local_path))
                    os.remove(local_path)

                logger.info("Soflinking the downloaded file: ln -s {} {}".
                            format(downloaded_path,
                                   local_path))
                os.symlink(downloaded_path, local_path)
                return None
        else:
            pull_dir = os.path.dirname(local_path)

        logger.info("Container file {} doesn't exist. Pulling the container from {}. Saving it to: {}".
                    format(local_path, remote_path, pull_dir))
        cmd = ['singularity', 'pull', '--name', os.path.basename(local_path), remote_path]
        logger.info(" ".join(cmd))
        returncode = subprocess.call(cmd,
                                     cwd=pull_dir)
        if returncode != 0:
            raise ValueError("Command: {} failed".format(" ".join(cmd)))

        # softlink it
        if os.environ.get('SINGULARITY_CACHEDIR'):
            if os.path.islink(local_path):
                logger.info("Softlink {} already exists. Removing it".format(local_path))
                os.remove(local_path)
            logger.info("Soflinking the downloaded file: ln -s {} {}".
                        format(downloaded_path,
                               local_path))
            os.symlink(downloaded_path, local_path)

        if not os.path.exists(local_path):
            raise ValueError("Container doesn't exist at the download path: {}".format(local_path))


def singularity_exec(container, command, bind_directories=[], dry_run=False):
    """Run `singularity exec`

    Args:
      container: path to the singularity image (*.sif)
      command: command to run (as a list)
      bind_directories: Additional directories to bind
    """
    if bind_directories:
        options = ['-B', ",".join(bind_directories)]
    else:
        options = []

    cmd = ['singularity', 'exec'] + options + [container] + command
    logger.info(" ".join(cmd))
    if dry_run:
        return print(" ".join(cmd))
    else:
        returncode = subprocess.call(cmd,
                                     stdin=subprocess.PIPE)
    if returncode != 0:
        raise ValueError("Command: {} failed".format(" ".join(cmd)))


# --------------------------------------------
# Figure out relative paths:
# - container path (e.g. shub://kipoi/models:latest)
# - local path (e.g. ~/.kipoi/envs/singularity/kipoi/models_latest.sif)

def container_remote_url(source='kipoi'):
    if source == 'kipoi':
        return 'shub://kipoi/models:latest'
    else:
        raise NotImplementedError("Containers for sources other than Kipoi are not yet implemented")


def container_local_path(remote_path):
    from kipoi.config import _kipoi_dir
    tmp = os.path.join(remote_path.split("://")[1])
    if ":" in tmp:
        relative_path, tag = tmp.split(":")
    else:
        relative_path = tmp
        tag = 'latest'
    return os.path.join(_kipoi_dir, "envs/singularity/", relative_path + "_" + tag + ".sif")

# ---------------------------------


def involved_directories(dataloader_kwargs, output_files=[], exclude_dirs=[]):
    """Infer the involved directories given dataloader kwargs
    """
    dirs = []
    # dataloader kwargs
    for k, v in six.iteritems(dataloader_kwargs):
        if os.path.exists(v):
            dirs.append(os.path.dirname(os.path.abspath(v)))

    # output files
    for v in output_files:
        dirs.append(os.path.dirname(os.path.abspath(v)))

    # optionally exclude directories
    def in_any_dir(fname, dirs):
        return any([is_subdir(fname, os.path.expanduser(d))
                    for d in dirs])
    dirs = [x for x in dirs
            if not in_any_dir(x, exclude_dirs)]

    return unique_list(dirs)


def create_conda_run():
    """Create conda_run bash script to ~/.kipoi/bin/conda_run

    NOTE: this should be changed to `conda run` once conda=4.6.0 is released
    https://github.com/conda/conda/issues/2379
    """
    from kipoi.config import _kipoi_dir
    crun = """#!/bin/bash
# Run a bash command in a new conda environment
set -e # stop on error

if [[ $# -lt 2 ]] ; then
    echo "Usage: "
    echo "       conda_run <conda envrionment> <command> "
    exit 0
fi

env=$1
cmd=${@:2}
echo "Running command in env: $env"
echo "Command: $cmd"

source activate $env
$cmd
source deactivate $env
"""
    bin_dir = os.path.join(_kipoi_dir, 'bin')
    makedir_exist_ok(bin_dir)
    crun_path = os.path.join(bin_dir, 'conda_run')
    with open(crun_path, 'w') as f:
        f.write(crun)

    # make it executable
    subprocess.call(["chmod", "u+x", crun_path])
    return crun_path


def singularity_command(kipoi_cmd, model, dataloader_kwargs, output_files=[], source='kipoi', dry_run=False):

    remote_path = container_remote_url(source)
    local_path = container_local_path(remote_path)
    singularity_pull(remote_path, local_path)

    assert kipoi_cmd[0] == 'kipoi'

    # remove all spaces within each command
    kipoi_cmd = [x.replace(" ", "").replace("\n", "").replace("\t", "") for x in kipoi_cmd]

    # figure out the right environment name
    stdout, stderr = _call_command('singularity', ['exec', local_path, 'kipoi', 'env', 'get', model], stdin=subprocess.PIPE)
    env_name = stdout.decode().strip()

    # create/get the `conda_run` command
    conda_run = create_conda_run()

    singularity_exec(local_path,
                     [conda_run, env_name] + kipoi_cmd,
                     # kipoi_cmd_conda,
                     bind_directories=involved_directories(dataloader_kwargs, output_files, exclude_dirs=['/tmp', '~']), dry_run=dry_run)
