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
import json
from kipoi_utils.utils import unique_list, makedir_exist_ok, is_subdir
from kipoi import get_source
import subprocess
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Python wrapper for the Singularity CLI
# def assert_installed():
#     """Make sure singularity is installed
#     """
#     pass

CONTAINER_PREFIX = "shared/containers"


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

def container_remote_url(model, source='kipoi'):
    src = get_source(source)
    singularity_container_json = os.path.join(src.local_path, CONTAINER_PREFIX, "model-to-singularity.json")
    with open(singularity_container_json, 'r') as singularity_container_json_filehandle:
        model_to_singularity_container_dict = json.load(singularity_container_json_filehandle)
    if model in model_to_singularity_container_dict: # Exact match such as MMSplice/mtsplice and APARENT/veff, Basset
        return model_to_singularity_container_dict[model]
    elif model.split('/')[0] in model_to_singularity_container_dict:
        return model_to_singularity_container_dict[model.split('/')[0]]
    else:
        return {}



def container_local_path(remote_path, container_name):
    from kipoi.config import _kipoi_dir
    if os.environ.get('SINGULARITY_CACHEDIR'):
        local_path = os.environ.get('SINGULARITY_CACHEDIR')
    else:
        local_path = os.path.join(_kipoi_dir, "envs/singularity/")
    if "versionId" in remote_path:
        version_id = remote_path.split("versionId=")[1]
        local_path = os.path.join(local_path, f"{container_name}/{version_id}")
    return local_path
    
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


def singularity_command(kipoi_cmd, model, dataloader_kwargs, output_files=[], source='kipoi', dry_run=False):
    if source != 'kipoi':
        raise NotImplementedError("Containers for sources other than Kipoi are not yet implemented")
    singularity_container_dict = container_remote_url(model, source)
    if singularity_container_dict:
        remote_path = singularity_container_dict['url']
        container_name = singularity_container_dict['name']
        local_path = container_local_path(remote_path, container_name)
        from kipoi_utils.external.torchvision.dataset_utils import download_url
        download_url(url=remote_path, root=local_path, filename=f"{container_name}.sif", md5=singularity_container_dict['md5'])

        assert kipoi_cmd[0] == 'kipoi'

        # remove all spaces within each command
        kipoi_cmd = [x.replace(" ", "").replace("\n", "").replace("\t", "") for x in kipoi_cmd]


        singularity_exec(f"{local_path}/{container_name}.sif",
                        kipoi_cmd,
                        # kipoi_cmd_conda,
                        bind_directories=involved_directories(dataloader_kwargs, output_files, 
                        exclude_dirs=['/tmp', '~']), 
                        dry_run=dry_run
                        )
    else:
        logger.warning(f"Singularity container for {model} either is not available yet or {model} is not in Kipoi.")