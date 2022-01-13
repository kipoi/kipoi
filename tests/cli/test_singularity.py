"""Test kipoi.cli.singularity
"""
import os

import pytest

from kipoi.cli.singularity import singularity_command

def test_singularity_command_dry_run():
    singularity_command(['kipoi', 'test', 'Basset', '--source=kipoi'], 'Basset', {}, dry_run=True)

def test_singularity_command_non_existing_model():
    singularity_command(['kipoi', 'test', 'abc', '--source=kipoi'], 'abc', {})
    

def test_singularity_command_non_kipoi_model():
    with pytest.raises(NotImplementedError):
        singularity_command(['kipoi', 'test', 'abc', '--source=local'], 'abc', {}, source='local')
        