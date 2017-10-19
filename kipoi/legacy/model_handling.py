from __future__ import absolute_import
from __future__ import print_function

import argparse
import importlib
import logging
import os
import subprocess
import sys
import yaml
import copy
import inspect
from .utils import load_module, pip_install_requirements
from .model import load_model
from .data import load_extractor, numpy_collate
from torch.utils.data import DataLoader

# HACK prevent this issue: https://github.com/kundajelab/genomelake/issues/4
import genomelake

_logger = logging.getLogger('kipoi')


PREPROC_FIELDS = ['function_name', 'type', 'arguments']
PREPROC_TYPES = ['generator', 'return']
PREPROC_IFILE_TYPES = ['DNA_regions']
PREPROC_IFILE_FORMATS = ['bed3']
MODEL_FIELDS = ['inputs', 'targets']
DATA_TYPES = ['dna', 'bigwig', 'v-plot']
RESERVED_PREPROC_KWS = ['intervals_file']

# Special files
MODULE_KERAS_OBJ = "custom_keras_objects.py"


# TODO - append the preprocessor yaml to the preproc __doc___
# Make Preprocessor a class-factory method?
class Preprocessor(object):

    def __init__(self, preprocessor_dir):
        """Main interface to provided preprocessors
        """
        with open(os.path.join(preprocessor_dir, 'preprocessor.yaml')) as ifh:
            description_yaml = yaml.load(ifh)
        self.preproc_spec = description_yaml['preprocessor']
        self.validate_preproc_spec()

        # import function_name from preprocessor.py
        preproc_file = os.path.join(preprocessor_dir, 'preprocessor.py')
        preproc = load_module(preproc_file)

        self.preproc_func = getattr(preproc, self.preproc_spec['function_name'])
        fargs = inspect.getargspec(self.preproc_func)
        self.req_preproc_func_args = fargs.args[:-len(fargs.defaults)]
        _logger.info('successfully imported {} from preprocessor.py'.
                     format(self.preproc_spec['function_name']))

        self.preproc_func_type = self.preproc_spec['type']

    def get_avail_arguments(self):
        return self.preproc_spec['arguments'].keys()

    def get_output_label_by_type(self, typestr):
        return [el for el in self.preproc_spec['output'] if el['type'] == typestr]

    def validate_preproc_spec(self):
        # check preprocessor fields
        assert (all(field in self.preproc_spec for field in PREPROC_FIELDS))

        # check preproc type
        assert self.preproc_spec['type'] in PREPROC_TYPES

        # I would say this is unneccessary - some outputs don't need a type (default np.array)
        # assert (all('type' in el.keys() for el in self.preproc_spec['output'].values()))

    def run_preproc(self, files_path=None, extra_files=None):
        # TODO - why do we need extra files?
        if extra_files is not None:
            assert(isinstance(extra_files, dict))
        else:
            extra_files = {}
        kwargs = {}
        if files_path is not None:
            kwargs = yaml.load(open(files_path))

        for k in extra_files:
            if k in self.preproc_spec['arguments']:
                kwargs[k] = extra_files[k]

        # check if there is a value for every required preprocessor function parameter is given
        assert (all(arg in kwargs for arg in self.req_preproc_func_args))

        # TODO: Check if this works with a generator preprocessor function
        # TODO: Return and yield cannot be combined in python 2, what do we want to support?!
        if self.preproc_func_type == "generator":
            for el in self.preproc_func(**kwargs):
                yield el
        else:
            yield self.preproc_func(**kwargs)


class ModelExtractor(object):

    def __init__(self, model_dir, install_req=False):
        """Combines model + preprocessor
        """
        self.model_dir = model_dir

        # TODO: This should not be done here, but a new environment
        # should have been created before calling this.
        if install_req:
            pip_install_requirements(os.path.join(model_dir, 'requirements.txt'))

        self.model = load_model(model_dir)
        self.extractor = load_extractor(model_dir)

    def validate_compatibility(self):
        # Test whether all the model input requirements are fulfilled
        # by the preprocessor output and whether types match
        raise Exception("Not yet implemented")

    def predict(self, extractor_kwargs, batch_size=32):
        """
        # Arguments
            preproc_kwargs: Keyword arguments passed to the pre-processor

        :return: model prediction result, list in case preprocessor is a generator
        """
        # TODO - don't return just a list... or?

        ret = []
        _logger.info('Initialized data generator. Running batches...')
        it = DataLoader(self.extractor(**extractor_kwargs),
                        batch_size=batch_size, collate_fn=numpy_collate)

        for i, batch in enumerate(it):
            ret.append(self.model.predict_on_batch(batch['inputs']))

        # TODO - test that predicted == true_target
        _logger.info('Successfully ran predict_on_batch')
        return ret


def cli_test(command, args):
    """CLI interface
    """
    assert command == "test"

    # setup the arg-parsing
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='script to test model zoo submissions')
    parser.add_argument('model_dir',
                        help='Model zoo submission directory.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-i", "--install-req", action='store_true',
                        help="Install required packages from requirements.txt")
    parsed_args = parser.parse_args(args)

    # run the model
    model_dir = os.path.abspath(parsed_args.model_dir)
    mh = ModelExtractor(model_dir, install_req=parsed_args.install_req)  # force the requirements to be installed

    test_dir = os.path.join(model_dir, 'test_files')

    if os.path.exists(test_dir):
        _logger.info(
            'Found test files in {}. Initiating test...'.format(test_dir))
        # cd to test directory
        os.chdir(test_dir)

    with open(os.path.join(test_dir, 'test.json')) as f_kwargs:
        extractor_kwargs = yaml.parse(f_kwargs)
    mh.predict(extractor_kwargs, batch_size=parsed_args.batch_size)
