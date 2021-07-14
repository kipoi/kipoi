"""Test load module
"""
import kipoi
from kipoi.utils import cd

import pytest



def test_sequential_model_loading():
    m1 = kipoi.get_model("example/models/kipoi_dataloader_decorator", source='dir')
    with cd(m1.source_dir):
        next(m1.default_dataloader.init_example().batch_iter())
