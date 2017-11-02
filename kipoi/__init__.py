from __future__ import absolute_import

__author__ = 'Kipoi team'
__version__ = '0.0.1'


# available modules
from . import config
from . import pipeline
from . import utils
from kipoi.postprocessing import variant_effects
from . import remote
from . import model
from . import data_utils
from . import data

# shortcuts
from .model import get_model
from .data import get_dataloader_factory
from .remote import get_model_info, get_dataloader_info
from .pipeline import install_model_requirements, install_dataloader_requirements
from .config import list_models, get_source, list_sources
# from .config import model_sources as sources


# Setup logging
import logging

log_formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
_logger = logging.getLogger('kipoi')

_handler = logging.StreamHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(log_formatter)
_logger.setLevel(logging.INFO)
_logger.addHandler(_handler)
