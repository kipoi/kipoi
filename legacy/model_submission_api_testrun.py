from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import logging
from kipoi_api import model_handling

_logger = logging.getLogger('model-zoo')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='script to test model zoo submissions')
    parser.add_argument('model_dir', help='Model zoo submission directory.')

    args = parser.parse_args()
    model_dir = os.path.abspath(args.model_dir)

    mh = model_handling.Model_handler(model_dir)

    test_dir = os.path.join(model_dir, 'test_files')

    if os.path.exists(test_dir):
        _logger.info(
            'Found test files in {}. Initiating test...'.format(test_dir))
        # cd to test directory
        os.chdir(test_dir)

    mh.predict(os.path.join(test_dir, 'test.json'))
