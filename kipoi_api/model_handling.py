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
from .utils import load_module

# HACK prevent this issue: https://github.com/kundajelab/genomelake/issues/4
import genomelake

PREPROC_FIELDS = ['function_name', 'type', 'arguments']
PREPROC_TYPES = ['generator', 'return']
PREPROC_IFILE_TYPES = ['DNA_regions']
PREPROC_IFILE_FORMATS = ['bed3']
MODEL_FIELDS = ['inputs', 'targets']
DATA_TYPES = ['dna', 'bigwig', 'v-plot']
RESERVED_PREPROC_KWS = ['intervals_file']

# Special files
MODULE_KERAS_OBJ = "custom_keras_objects.py"


log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('kipoi model-zoo')
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_handler)


class Preprocessor:

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


class Model:

    def __init__(self, model_dir):
        """Main interface to provided models
        """
        with open(os.path.join(model_dir, 'model.yaml')) as ifh:
            description_yaml = yaml.load(ifh)
        self.model_spec = description_yaml['model']
        self.validate_model_spec()

        # test model loading
        _logger.info('Testing model files...')
        from keras.models import model_from_json

        # load custom Keras objects
        custom_objects_path = os.path.join(model_dir, MODULE_KERAS_OBJ)
        if custom_objects_path:
            self.custom_objects = load_module(custom_objects_path).OBJECTS
        else:
            self.custom_objects = {}

        arch_fname = os.path.join(model_dir, 'model.json')
        weights_fname = os.path.join(model_dir, 'weights.h5')

        # load arch
        with open(arch_fname, "r") as arch:
            self.model = model_from_json(arch.read(),
                                         custom_objects=self.custom_objects)
        _logger.info('successfully loaded model architecture from {}'.
                     format(arch_fname))

        # load weights
        self.model.load_weights(weights_fname)
        _logger.info('successfully loaded model weights from {}'.
                     format(weights_fname))

    def validate_model_spec(self):
        # check model fields
        assert (all(field in self.model_spec for field in MODEL_FIELDS))

        # check input and target data types
        for data_name, data_spec in self.model_spec['inputs'].items():
            if type in data_spec:
                assert data_spec['type'] in DATA_TYPES
        for data_name, data_spec in self.model_spec['targets'].items():
            if type in data_spec:
                assert data_spec['type'] in DATA_TYPES

    def predict_on_batch(self, input):
        return self.model.predict_on_batch(input)

    def get_model_obj(self):
        return self.model


class Model_handler:

    def __init__(self, model_dir, install_requirements=False):
        """Combines model + preprocessor
        """
        self.model_dir = model_dir

        # TODO: This should not be done here, but a new environment
        # should have been created before calling this.
        if install_requirements:
            requirements_fname = os.path.join(model_dir, 'requirements.txt')
            self.install_requirements(requirements_fname)

        self.model = Model(model_dir)
        self.preproc = Preprocessor(model_dir)

    def validate_compatibility(self):
        # Test whether all the model input requirements are fulfilled
        # by the preprocessor output and whether types match
        raise Exception("Not yet implemented")

    def run_preproc(self, files_path=None, extra_files=None):
        self.preproc.run_preproc(files_path, extra_files)

    def get_model_obj(self):
        self.model.get_model_obj()

    def predict(self, files_path=None, extra_files=None):
        """
        :param files_path: yaml file containing values for all arguments
        necessary for running preproc_func

        :param extra_files: If a key in extra_files matches a preproc_func
        the value in extra_files will be used for that key

        :return: model prediction result, list in case preprocessor is a generator
        """

        ret = []
        _logger.info('Initialized data generator. Running batches...')
        for i, batch in enumerate(self.preproc.run_preproc(files_path, extra_files)):
            ret.append(self.model.predict_on_batch(batch['inputs']))
        _logger.info('Successfully ran predict_on_batch')
        return ret

    def install_requirements(self, requirements_fname):
        if os.path.exists(requirements_fname):  # install dependencies
            _logger.info('found requirements.txt in {}'.format(self.model_dir))
            _logger.info('Running pip install -r {}...'.format(requirements_fname))
            subprocess.call(['pip', 'install', '-r', requirements_fname])
        else:
            _logger.info('requirements.txt not found in {}'.format(self.model_dir))

