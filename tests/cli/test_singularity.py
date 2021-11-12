"""Test kipoi.cli.singularity
"""
import os
import pytest
from kipoi.cli.singularity import singularity_command

def test_singularity_command_dry_run():
    singularity_command(['kipoi', 'test', 'Basset', '--source=kipoi'], 'Basset', {}, dry_run=True)
