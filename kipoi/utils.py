from __future__ import absolute_import
from __future__ import print_function

import six
import pickle
import glob
import os
import sys
import subprocess
import numpy as np
import yaml
from collections import OrderedDict
from contextlib import contextmanager
import inspect
import logging
import collections
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_module(path, module_name=None):
    """Load python module from file

    Args:
       path: python file path
       module_name: import as `module_name` name. If none, use `path[:3]`
    """
    assert path.endswith(".py")
    if module_name is None:
        module_name = os.path.basename(path)[:-3]  # omit .py

    logger.debug("loading module: {0} as {1}".format(path, module_name))
    if sys.version_info[0] == 2:
        import imp
        # add the directory to system's path - allows loading submodules
        sys.path.append(os.path.join(os.path.dirname(module_name)))
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
        logger.info('Running pip install -r {}...'.format(requirements_fname))
        subprocess.call(['pip', 'install', '-r', requirements_fname])
    else:
        logger.info('requirements.txt not found under {}'.format(requirements_fname))


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
    extractor_args = extractor_args.strip("'").strip('"')
    if extractor_args.startswith("{") or extractor_args.endswith("}"):
        logger.debug("Parsing the extractor_args as a json string")
        return yaml.load(extractor_args)
    else:
        if not os.path.exists(extractor_args):
            raise ValueError("File path: {0} doesn't exist".format(extractor_args))
        logger.debug("Parsing the extractor_args as a json file path")
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
            return set(inspect.getargspec(x).args)
        else:
            # skip the self parameter
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
    if not ce:
        if raise_exception:
            raise OSError("git-lfs not installed")
        else:
            logger.warn("git-lfs not installed")
    return ce


def get_file_path(file_dir, basename, extensions=[".yml", ".yaml"],
                  raise_err=True):
    """Get the file path allowing for multiple file extensions
    """
    for ext in extensions:
        path = os.path.join(file_dir, basename + ext)
        if os.path.exists(path):
            return path
    if raise_err:
        raise ValueError("File path doesn't exists: {0}/{1}{2}".
                         format(file_dir, basename, set(extensions)))
    else:
        return None


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    try:
        return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')
    except Exception:
        return "NA"


class Slice_conv:

    def __getitem__(self, key):
        return key


def unique_list(seq):
    """Make a list unique and preserve the elements order

    Modified version of Dave Kirby solution
    """
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]


def read_txt(file_path, comment_str="#"):
    """Txt file reader that ignores comments and
    empty lines
    """
    out = []
    with open(file_path) as f:
        for line in f:
            line = line.partition(comment_str)[0]
            line = line.strip()
            if len(line) > 0:
                out.append(line)
    return out


def read_pickle(f):
    with open(f, "rb") as f:
        return pickle.load(f)


def merge_dicts(x, y):
    """https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
    """
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def list_files_recursively(root_dir, basename, suffix='y?ml'):
    """search for filenames matching the pattern: {root_dir}/**/{basename}.{suffix}
    """
    if sys.version_info >= (3, 5):
        return [filename[len(root_dir):] for filename in
                glob.iglob(root_dir + '**/{0}.{1}'.format(basename, suffix), recursive=True)]
    else:
        import fnmatch
        return [os.path.join(root, filename)[len(root_dir):]
                for root, dirnames, filenames in os.walk(root_dir)
                for filename in fnmatch.filter(filenames, '{0}.{1}'.format(basename, suffix))]


def map_nested(dd, fn):
    """Map a function to a nested data structure (containing lists or dictionaries

    Args:
      dd: nested data structure
      fn: function to apply to each leaf
    """
    if isinstance(dd, collections.Mapping):
        return {key: map_nested(dd[key], fn) for key in dd}
    elif isinstance(dd, collections.Sequence):
        return [map_nested(x, fn) for x in dd]
    else:
        return fn(dd)


def take_first_nested(dd):
    """Get a single element from the nested list/dictionary

    Args:
      dd: nested data structure

    Example: take_first_nested({"a": [1,2,3], "b": 4}) == 1
    """
    if isinstance(dd, collections.Mapping):
        return take_first_nested(six.next(six.itervalues(dd)))
    elif isinstance(dd, collections.Sequence):
        return take_first_nested(dd[0])
    else:
        return dd

def print_dl_kwargs(dataloader_class, format_examples_json=False):
    from .external.related.fields import UNSPECIFIED
    if hasattr(dataloader_class, "args"):
        args = dataloader_class.args
        for k in args:
            print("Keyword argument: `{0}`".format(k))
            for elm in ["doc", "type", "optional", "example"]:
                if hasattr(args[k], elm) and \
                        (not isinstance(getattr(args[k], elm), UNSPECIFIED)):
                    print("    {0}: {1}".format(elm, getattr(args[k], elm)))
        example_kwargs = dataloader_class.example_kwargs
        print("-" * 80)
        if hasattr(dataloader_class, "example_kwargs"):
            if format_examples_json:
                import json
                example_kwargs = json.dumps(example_kwargs)
            print("Example keyword arguments are: {0}".format(str(example_kwargs)))
    else:
        print("No keyword arguments defined for the given dataloader.")
