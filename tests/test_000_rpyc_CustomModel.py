import warnings

warnings.filterwarnings('ignore')
import pytest
import sys
import os
import yaml
from contextlib import contextmanager
import kipoi
from kipoi.pipeline import install_model_requirements
from kipoi_utils.utils import _call_command
import config
import numpy as np
from kipoi_utils.utils import cd
from kipoi_conda.utils import get_kipoi_bin
from kipoi.rpyc_model import *
from kipoi.cli.env import *


from utils import *


import subprocess


from test_16_KerasModel import (get_sample_functional_model,cd, get_sample_functional_model_input,get_test_kwargs,read_json_yaml, get_sample_sequential_model)


INSTALL_REQ = config.install_req




#@pytest.mark.flaky(max_runs=5)
@pytest.mark.parametrize("example",    ['dummy_custom'])
@pytest.mark.parametrize("port",  [4000,4010])
def test_pipeline(example , port):

    example_dir = "example/models/{0}".format(example)
    test_kwargs = {}

    env_name = create_env_if_not_exist(bypass=False, model=example_dir, source='dir')

    # get remote model
    s = kipoi.rpyc_model.ServerArgs(env_name=env_name, use_current_python=False,  address='localhost', port=port, logging_level=0)
    with  kipoi.get_model(example_dir, source="dir", server_settings=s) as remote_model:
        
        newdir = example_dir + "/example_files"
        with remote_model.cd_local_and_remote(newdir):

            pipeline = remote_model.pipeline
            the_pred = pipeline.predict(dataloader_kwargs=test_kwargs)
            assert the_pred is not None
