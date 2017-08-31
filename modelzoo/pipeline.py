"""Whole model pipeline: extractor + model 
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import logging
import os
import yaml
from .utils import pip_install_requirements
from .model import load_model
from .data import load_extractor, numpy_collate
from torch.utils.data import DataLoader

# HACK prevent this issue: https://github.com/kundajelab/genomelake/issues/4
import genomelake

_logger = logging.getLogger('model-zoo')


PREPROC_FIELDS = ['function_name', 'type', 'arguments']
PREPROC_TYPES = ['generator', 'return']
PREPROC_IFILE_TYPES = ['DNA_regions']
PREPROC_IFILE_FORMATS = ['bed3']
MODEL_FIELDS = ['inputs', 'targets']
DATA_TYPES = ['dna', 'bigwig', 'v-plot']
RESERVED_PREPROC_KWS = ['intervals_file']


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
    parser = argparse.ArgumentParser('modelzoo {}'.format(command),
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
        extractor_kwargs = yaml.load(f_kwargs)
    mh.predict(extractor_kwargs, batch_size=parsed_args.batch_size)
