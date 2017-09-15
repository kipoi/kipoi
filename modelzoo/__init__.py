from __future__ import absolute_import

__author__ = 'Kipoi team'
__version__ = '0.0.1'


# available modules
from . import config
from . import pipeline
from . import utils
from . import variant_effects
from . import remote

from . import model
from .model import load_model
from .data import load_extractor
from .pipeline import install_model_requirements

sources = config.model_sources

# Setup logging
import logging

log_formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('model-zoo')

_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)
