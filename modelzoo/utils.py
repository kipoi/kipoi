import os
import sys
import subprocess
import logging
import numpy as np
import yaml
from .config import kipoi_models_repo

_logger = logging.getLogger('model-zoo')


def load_module(path, module_name=None):
    """Load python module from file
    """
    assert path.endswith(".py")
    if module_name is None:
        module_name = os.path.basename(path)[:-3]  # omit .py

    if sys.version_info[0] == 2:
        import imp
        module = imp.load_source(module_name, path)
    elif sys.version_info[0] == 3:
        # TODO: implement dynamic loading of preprocessor module for python3
        """
        import importlib.machinery
        loader = importlib.machinery.SourceFileLoader
        handle = loader.load_module
        """
        if sys.version_info[1] == 4:
            # way 1 (=python3.4)
            from importlib.machinery import SourceFileLoader
            module = SourceFileLoader(module_name, path).load_module()
        if sys.version_info[1] >= 5:
            # way 2 (>=python3.5)
            import importlib.util

            # alternative
            import types
            loader = importlib.machinery.SourceFileLoader(module_name, path)
            module = types.ModuleType(loader.name)
            loader.exec_module(module)

            # module_spec_ = importlib.util.spec_from_file_location(module_name, path)
            # module = importlib.util.module_from_spec(module_spec_)
            # module_spec_.loader.exec_module(module)
        else:
            raise RuntimeError(
                'dynamic loading of preprocessor module is not implemented for python3!')
    return module


def pip_install_requirements(requirements_fname):
    if os.path.exists(requirements_fname):  # install dependencies
        _logger.info('Running pip install -r {}...'.format(requirements_fname))
        subprocess.call(['pip', 'install', '-r', requirements_fname])
    else:
        _logger.info('requirements.txt not found under {}'.format(requirements_fname))


def compare_numpy_dict(a, b, exact=True):
    """
    Compare two recursive numpy dictionaries or lists
    """
    if type(a) != type(b) and type(a) != np.ndarray and type(b) != np.ndarray:
        return False

    # Compare two dictionaries
    if type(a) == dict and type(b) == dict:
        if not a.keys() == b.keys():
            return False
        for key in a.keys():
            res = compare_numpy_dict(a[key], b[key], exact)
            if not res:
                print("false for key = ", key)
                return False
        return True

    # compare two lists
    if type(a) == list and type(b) == list:
        assert len(a) == len(b)
        return all([compare_numpy_dict(a[i], b[i], exact=exact)
                    for i in range(len(a))])

    # if type(a) == np.ndarray and type(b) == np.ndarray:
    if type(a) == np.ndarray or type(b) == np.ndarray:
        if exact:
            return (a == b).all()
        else:
            return np.testing.assert_almost_equal(a, b)

    if a is None and b is None:
        return True

    raise NotImplementedError


def parse_json_file_str(extractor_args):
    """Parse a string either as a json string or
    as a file path to a .json file
    """
    if extractor_args.startswith("{") or extractor_args.endswith("}"):
        _logger.debug("Parsing the extractor_args as a json string")
        return yaml.load(extractor_args)
    else:
        if not os.path.exists(extractor_args):
            raise ValueError("File path: {0} doesn't exist".format(extractor_args))
        _logger.debug("Parsing the extractor_args as a json file path")
        with open(extractor_args, "r") as f:
            return yaml.load(f.read())

# --------------- GIT functionality ----------------


def cmd_exists(cmd):
    """Check if a certain command exists
    """
    return subprocess.call("type " + cmd, shell=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0


def lfs_installed(raise_exception=False):
    """Check if git lfs is installed localls
    """
    ce = cmd_exists("git-lfs")
    if raise_exception:
        if not ce:
            raise OSError("git-lfs not installed")
    return ce


def clone_kipoi_models():
    """Clone the kipoi/models repo without the model weights
    """
    lfs_installed(raise_exception=True)

    # git lfs clone git@github.com:kipoi/models.git '-I /'

    _logger.info("Cloning git@github.com:kipoi/models.git into {0}".
                 format(kipoi_models_repo()))
    subprocess.call(["git-lfs",
                     "clone",
                     "-I /",
                     "git@github.com:kipoi/models.git",
                     kipoi_models_repo()])


def pull_kipoi_models():
    """Update the kipoi models repository
    """
    lfs_installed(raise_exception=True)

    assert os.path.exists(kipoi_models_repo())

    _logger.info("Update {0}".
                 format(kipoi_models_repo()))
    subprocess.call(["git",
                     "pull"],
                    cwd=kipoi_models_repo())
    subprocess.call(["git-lfs",
                     "pull",
                     "-I /"],
                    cwd=kipoi_models_repo())


def pull_kipoi_model(model):
    """Pull the weights of a particular model
    """
    lfs_installed(raise_exception=True)

    if not os.path.exists(kipoi_models_repo()):
        clone_kipoi_models()
    else:
        pull_kipoi_models()

    if not os.path.exists(os.path.join(kipoi_models_repo(), model)):
        raise ValueError("Model: {0} doesn't exist in https://github.com/kipoi/models".
                         format(model))

    cmd = ["git-lfs",
           "pull",
           "-I {model}/**".format(model=model)]
    _logger.info(" ".join(cmd))
    subprocess.call(cmd,
                    cwd=kipoi_models_repo()
                    )
    _logger.info("model {0} loaded".format(model))
