from __future__ import absolute_import

__author__ = 'Kipoi team'
__email__ = 'avsec@in.tum.de'
__version__ = '0.5.2'


# available modules
from . import config
from . import pipeline
from . import utils
from . import sources
from . import remote  # backward compat
from . import model
from . import data_utils
from . import data
from . import conda
from . import specs
from . import components  # backward compat
from . import readers
from . import writers
from . import plugin

# shortcuts
from .model import get_model
from .data import get_dataloader_factory
from .sources import get_model_descr, get_dataloader_descr
from .pipeline import install_model_requirements, install_dataloader_requirements
from .config import get_source, list_sources, list_models, list_dataloaders
from . import cli
from .plugin import list_plugins
# from .config import model_sources as sources
