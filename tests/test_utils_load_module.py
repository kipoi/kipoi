"""Test load module
"""
import kipoi
from kipoi.utils import cd

import pytest



def test_sequential_model_loading():
    m2 = kipoi.get_model("example/models/extended_coda", source='dir')
    m1 = kipoi.get_model("example/models/kipoi_dataloader_decorator", source='dir')

    with cd(m2.source_dir):
        next(m2.default_dataloader.init_example().batch_iter())
    with cd(m1.source_dir):
        next(m1.default_dataloader.init_example().batch_iter())
