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
from . import data_utils
from . import data

# shortcuts
from .model import Model
from .data import DataLoader_factory
from .remote import model_info, requirements_file
from .pipeline import install_model_requirements
from .config import list_models, get_source
from .config import model_sources as sources


# Setup logging
import logging

log_formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('kipoi')

_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)
