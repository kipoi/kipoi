"""Test kipoi cli
"""
import kipoi
from kipoi.cli.main import ls_helper
import pytest


@pytest.fixture
def df():
    return kipoi.list_models()


def test_ls_helper(df):
    ls_helper(df, '', False)
    ls_helper(df, '', True)

    ls_helper(df, 'Basset', False)

    ls_helper(df, 'Basset/asd', False)

    ls_helper(df, 'Basset/asd', False)
    ls_helper(df, 'all', False)
    ls_helper(df, 'all', True)
