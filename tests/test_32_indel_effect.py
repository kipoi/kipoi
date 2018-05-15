import kipoi
import kipoi.postprocessing.variant_effects as ve
import numpy as np
import pytest
import sys
from kipoi.pipeline import install_model_requirements
import warnings
import filecmp
import config
import os
import copy
from kipoi.utils import cd
import pandas as pd
import tempfile
from kipoi.metadata import GenomicRanges

warnings.filterwarnings('ignore')


# general structure:
# for an indel we need a) the reference output, b) the alternative output with the variant.
# conda config --add channels r
# conda config --add channels bioconda
# pip install bx-python
# conda install -c kbchoi g2gtools


def test__generate_subset_inputfiles():
    pass
