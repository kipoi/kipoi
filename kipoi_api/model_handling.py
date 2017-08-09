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

PREPROC_FIELDS = ['function_name', 'type', 'arguments']
PREPROC_TYPES = ['generator', 'return']
PREPROC_IFILE_TYPES = ['DNA_regions']
PREPROC_IFILE_FORMATS = ['bed3']
MODEL_FIELDS = ['inputs', 'targets']
DATA_TYPES = ['dna', 'bigwig', 'v-plot']
RESERVED_PREPROC_KWS = ['intervals_file']


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
        with open(os.path.join(preprocessor_dir, 'preprocessor.yaml')) as ifh:
            description_yaml = yaml.load(ifh)
        self.preproc_spec = description_yaml['preprocessor']
        self.validate_preproc_spec()

        # import function_name from preprocessor.py
        preproc_file = os.path.join(preprocessor_dir, 'preprocessor.py')
        if sys.version_info[0] == 2:
            import imp
            preproc = imp.load_source('preprocessor', preproc_file)
        else:  # TODO: implement dynamic loading of preprocessor module for python3
            """
            import importlib.machinery
            loader = importlib.machinery.SourceFileLoader
            handle = loader.load_module
            # way 1 (=python3.4)
            from importlib.machinery import SourceFileLoader
            preproc = SourceFileLoader("preprocessor", preproc_file).load_module()
            # way 2 (>=python3.5)
            import importlib.util
            preproc_spec = importlib.util.spec_from_file_location("preprocessor", preproc_file)
            preproc = importlib.util.module_from_spec(preproc_spec)
            preproc_spec.loader.exec_module(preprocessor)
            """
            raise RuntimeError(
                'dynamic loading of preprocessor module is not implemented for python3!')

        self.preproc_func = getattr(preproc, self.preproc_spec['function_name'])
        _logger.info('successfully imported {} from preprocessor.py'.format(self.preproc_spec['function_name']))

        self.preproc_func_type = self.preproc_spec['type']

    def get_avail_arguments(self):
        return self.preproc_spec['arguments'].keys()

    def get_output_label_by_type(self, typestr):
        return [el for el in self.preproc_spec['output'] if el['type']==typestr]

    def validate_preproc_spec(self):
        # check preprocessor fields
        assert (all(field in self.preproc_spec for field in PREPROC_FIELDS))

        # check preproc type
        assert self.preproc_spec['type'] in PREPROC_TYPES

        assert (all('type' in el.keys() for el in self.preproc_spec['output']))


    def run_preproc(self, files_path=None, extra_files = None):
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

        # check if there is a value for every argument and there are not more inputs
        # given than preprocessor arguments are available
        assert (all(arg in kwargs for arg in self.preproc_spec['arguments'].keys()))
        assert (len(self.preproc_spec['arguments']) == len(kwargs))

        #TODO: Check if this works with a generator preprocessor function
        #TODO: Return and yield cannot be combined in python 2, what do we want to support?!
        if self.preproc_func_type == "generator":
            for el in self.preproc_func(**kwargs):
                yield el
        else:
            yield self.preproc_func(**kwargs)

class Model:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, 'model.yaml')) as ifh:
            description_yaml = yaml.load(ifh)
        self.model_spec = description_yaml['model']
        self.validate_model_spec()

        # test model loading
        _logger.info('Testing model files...')
        from keras.models import model_from_json
        arch_fname = os.path.join(model_dir, 'model.json')
        weights_fname = os.path.join(model_dir, 'weights.h5')
        self.model = model_from_json(open(arch_fname).read())
        _logger.info(
            'successfully loaded model architecture from {}'.format(arch_fname))
        self.model.load_weights(weights_fname)
        _logger.info(
            'successfully loaded model weights from {}'.format(weights_fname))

    def validate_model_spec(self):
        # check model fields
        assert (all(field in self.model_spec for field in MODEL_FIELDS))

        # check input and target data types
        for data_name, data_spec in self.model_spec['inputs'].items():
            assert data_spec['type'] in DATA_TYPES
        for data_name, data_spec in self.model_spec['targets'].items():
            assert data_spec['type'] in DATA_TYPES

    def predict_on_batch(self, input):
        return self.model.predict_on_batch(input)

class Model_handler:
    def __init__(self, model_dir, install_requirements=False):
        self.model_dir = model_dir

        # TODO: This should not be done here, but a new environment should have been created before calling this.
        if install_requirements:
            requirements_fname = os.path.join(model_dir, 'requirements.txt')
            self.install_requirements(requirements_fname)

        self.model = Model(model_dir)
        self.preproc = Preprocessor(model_dir)

    def validate_compatibility(self):
        # Test whether all the model input requirements are fulfilled by the preprocessor output and whether types match
        raise Exception("Not yet implemented")

    def predict(self, files_path=None, extra_files = None):
        """
        :param files_path: yaml file containing values for all arguments necessary for running preproc_func  
        :param extra_files: If a key in extra_files matches a preproc_func the value in extra_files will be used for that key
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



