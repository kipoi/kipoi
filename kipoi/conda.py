"""Conda helper files

Reusing code from: https://github.com/conda/conda-api/blob/master/conda_api.py
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import json
import sys
import subprocess
from subprocess import Popen, PIPE, STDOUT
from collections import OrderedDict
from kipoi.utils import yaml_ordered_dump
import logging


class CondaError(Exception):
    "General Conda error"
    pass


def create_env(env_name, conda_deps):
    """Create new environment given parsed dependencies

    Args:
      conda_dependencies: OrderedDict of the `dependencies` field in Conda's environment.yaml.
        `model.dependencies.conda` field in Model or Dataloader.
      env_name: Environment name
    """
    # check if the environment already exists
    if env_exists(env_name):
        logging.info("Conda environment: {0} already exists. Skipping the installation.".
                     format(env_name))
        return None

    # write the env to file
    env_dict = OrderedDict([
        ("name", env_name),
        ("dependencies", conda_deps)
    ])
    tmp_dir = "/tmp/kipoi"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tmp_file_path = "{tmp_dir}/{env_name}.yml".format(tmp_dir=tmp_dir,
                                                      env_name=env_name)
    with open(tmp_file_path, 'w') as f:
        f.write(yaml_ordered_dump(env_dict, indent=4, default_flow_style=False))

    # create the environment
    return create_env_from_file(tmp_file_path)


def create_env_from_file(env_file):
    cmd_list = ["env", "create", "--file", env_file]
    return _call_conda(cmd_list, use_stdout=True)


def install_conda(conda_deps):
    conda_deps_wo_python = [x for x in conda_deps if "python" != x[:6]]
    if conda_deps_wo_python:
        cmd_list = ["install", "-y"] + conda_deps_wo_python
        return _call_conda(cmd_list, use_stdout=True)


def install_pip(pip_deps):
    if pip_deps:
        cmd_list = ["install"] + list(pip_deps)
        return _call_pip(cmd_list, use_stdout=True)


def remove_env(env_name):
    cmd_list = ["env", "remove", "-y", "-n", env_name]
    return _call_conda(cmd_list, use_stdout=True)


def get_envs():
    """
    Return all of the (named) environment (this does not include the root
    environment), as a list of absolute path to their prefixes.
    """
    info = _call_and_parse(['info', '--json'])
    return info['envs']


def env_exists(env):
    return env in [os.path.basename(x) for x in get_envs()]


def _call_command(cmd, extra_args, use_stdout=False):
    # call conda with the list of extra arguments, and return the tuple
    # stdout, stderr
    cmd_list = [cmd]  # just use whatever conda is on the path

    cmd_list.extend(extra_args)

    try:
        if use_stdout:
            p = Popen(cmd_list, stdout=PIPE, universal_newlines=True)
            # Poll process for new output until finished
            for stdout_line in iter(p.stdout.readline, ""):
                print(stdout_line, end='')
            p.stdout.close()
            return_code = p.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, cmd_list)
            return return_code
        else:
            p = Popen(cmd_list, stdout=PIPE, stderr=PIPE)
    except OSError:
        raise Exception("could not invoke {0}\n".format(cmd_list))
    return p.communicate()


def _call_conda(extra_args, use_stdout=False):
    return _call_command("conda", extra_args, use_stdout)


def _call_pip(extra_args, use_stdout=False):
    return _call_command("pip", extra_args, use_stdout)


def _call_and_parse(extra_args):
    stdout, stderr = _call_conda(extra_args)
    if stderr.decode().strip():
        raise Exception('conda %r:\nSTDERR:\n%s\nEND' % (extra_args,
                                                         stderr.decode()))
    return json.loads(stdout.decode())
