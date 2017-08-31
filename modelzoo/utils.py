import os
import sys
import subprocess
import logging

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
