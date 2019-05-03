import warnings
warnings.filterwarnings('ignore')
import pytest
import sys
import os
from kipoi.rpyc_model import *


@pytest.mark.parametrize("port", [18838,18839,18840,18838,18839,18840])
def test_rpyc_server(port):

    server = RpycServer(env_name=None,address='localhost', port=port)
    assert not server.is_running
    server.start()
    assert server.is_running
    assert server.echo(x='hello world') == 'hello world'




@pytest.mark.parametrize("port", [18838,18839,18840,18838,18839,18840])
def test_rpyc_server_contextmanager(port):

    with RpycServer(env_name=None,address='localhost', port=port) as server:
        assert server.is_running
        assert server.echo(x='hello world') == 'hello world'