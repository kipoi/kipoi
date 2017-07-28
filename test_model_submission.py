from __future__ import absolute_import
from __future__ import print_function

import argparse
import importlib
import logging
import os
import subprocess
import sys
import yaml

EXTRACTOR_FIELDS = ['function_name', 'type', 'arguments']
EXTRACTOR_TYPES = ['generator']
MODEL_FIELDS = ['inputs', 'targets']
DATA_TYPES = ['dna', 'bigwig', 'v-plot']

log_formatter = logging.Formatter(
    '%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('kipoi model-zoo')
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_handler)

parser = argparse.ArgumentParser(
    description='script to test model zoo submissions')
parser.add_argument('model_dir', help='Model zoo submission directory.')


def main():
    args = parser.parse_args()
    model_dir = os.path.abspath(args.model_dir)

    # check for requirements file
    requirements_fname = os.path.join(model_dir, 'requirements.txt')
    if os.path.exists(requirements_fname):  # install dependencies
        _logger.info('found requirements.txt in {}'.format(model_dir))
        _logger.info('Running pip install -r {}...'.format(requirements_fname))
        subprocess.call(['pip', 'install', '-r', requirements_fname])
    else:
        _logger.info('requirements.txt not found in {}'.format(model_dir))

    # parse description yaml
    _logger.info('parsing description.yaml')
    description_yaml = yaml.load(
        open(os.path.join(model_dir, 'description.yaml')))
    extractor_spec = description_yaml['data_extractor']
    model_spec = description_yaml['model']

    # check extractor fields
    assert(all(field in extractor_spec for field in EXTRACTOR_FIELDS))

    # check extractor type
    assert extractor_spec['type'] in EXTRACTOR_TYPES

    # check model fields
    assert(all(field in model_spec for field in MODEL_FIELDS))

    # check input and target data types
    for data_name, data_spec in model_spec['inputs'].items():
        assert data_spec['type'] in DATA_TYPES
    for data_name, data_spec in model_spec['targets'].items():
        assert data_spec['type'] in DATA_TYPES

    # import function_name from extractor.py
    if sys.version_info[0] == 2:
        import imp
        extractor = imp.load_source(
            'extractor', os.path.join(model_dir, 'extractor.py'))
    else:  # TODO: implement dynamic loading of extractor module for python3
        """
        import importlib.machinery
        loader = importlib.machinery.SourceFileLoader
        handle = loader.load_module
        """
        raise RuntimeError(
            'dynamic loading of extractor module is not implemented for python3!')

    extractor_func = getattr(extractor, extractor_spec['function_name'])
    _logger.info('successfully imported {} from extractor.py'.format(
        extractor_spec['function_name']))

    # test model loading
    _logger.info('Testing model files...')
    from keras.models import model_from_json
    arch_fname = os.path.join(model_dir, 'model.json')
    weights_fname = os.path.join(model_dir, 'weights.h5')
    model = model_from_json(open(arch_fname).read())
    _logger.info(
        'successfully loaded model architecture from {}'.format(arch_fname))
    model.load_weights(weights_fname)
    _logger.info(
        'successfully loaded model weights from {}'.format(weights_fname))

    # check for test files directory
    test_dir = os.path.join(model_dir, 'test_files')
    if os.path.exists(test_dir):
        _logger.info(
            'Found test files in {}. Initiating test...'.format(test_dir))

        # cd to test directory
        os.chdir(test_dir)

        # parse test.yaml
        test_spec_fname = os.path.join(test_dir, 'test.yaml')
        test_spec = yaml.load(open(test_spec_fname))
        _logger.info(
            'Successfully parsed test specification file {}'.format(test_spec_fname))

        _logger.info('Running data extractor with the following arguments:')
        for arg, value in test_spec.items():
            _logger.info('{}: {}'.format(arg, value))

        if extractor_spec['type'] == 'generator':
            # get generator
            generator = extractor_func(**test_spec)
            _logger.info('Initialized data generator. Running 10 batches...')
            for i in range(10):
                batch = next(generator)
                model.predict_on_batch(batch['inputs'])
            _logger.info('Successfully ran predict_on_batch')
        else:
            # TODO
            raise RuntimeError(
                'Testing extraction for non-generator extractor is not implemented!')
    else:
        _logger.info('test_files not found.')

    _logger.info('model submission test finished successfully!')

if __name__ == '__main__':
    main()
