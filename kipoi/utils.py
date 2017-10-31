from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import subprocess
import logging
import numpy as np
import yaml
import six
from collections import OrderedDict
from contextlib import contextmanager
import inspect
_logger = logging.getLogger('kipoi')


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


# https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts


def yaml_ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
    OrderedLoader.add_constructor(_mapping_tag, dict_constructor)
    return yaml.load(stream, OrderedLoader)


def yaml_ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def dict_representer(dumper, data):
        return dumper.represent_dict(six.iteritems(data))
    OrderedDumper.add_representer(OrderedDict, dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


@contextmanager
def cd(newdir):
    """Temporarily change the directory
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def getargs(x):
    """Get function arguments
    """
    if sys.version_info[0] == 2:
        if inspect.isfunction(x):
            return set(inspect.getargspec(x).args[1:])
        else:
            return set(inspect.getargspec(x.__init__).args[1:])
    else:
        return set(inspect.signature(x).parameters.keys())


def read_yaml(path):
    with open(path) as f:
        return yaml.load(f)


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


def get_file_path(file_dir, basename, extensions=[".yml", ".yaml"]):
    """Get the file path allowing for multiple file extensions
    """
    for ext in extensions:
        path = os.path.join(file_dir, basename + ext)
        if os.path.exists(path):
            return path
    raise ValueError("File path doesn't exists: {0}/{1}{2}".
                     format(file_dir, basename, set(extensions)))


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    try:
        return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')
    except:
        return "NA"
