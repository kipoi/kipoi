"""Test kipoi.cli.singularity
"""
import os
import pytest
from kipoi.cli.singularity import (singularity_pull, singularity_exec,
                                   container_remote_url, container_local_path,
                                   involved_directories,
                                   singularity_command)

#@pytest.mark.skip(reason="no singularity tests atm")
def test_singularity_pull_run(tmpdir):
    tmpdir = '/tmp/singularity'
    output_file = os.path.join(str(tmpdir), "hello-world_latest.sif")

    singularity_pull("shub://vsoch/hello-world:latest", output_file)
    # For now, don't run the command
    # singularity_exec(output_file, ['echo', 'hello-world'])


#@pytest.mark.skip(reason="circle-ci fails for unknown reasons on this test")
def test_singularity_command_dry_run():
    singularity_command(['kipoi', 'test', 'Basset', '--source=kipoi'], 'Basset', {}, dry_run=True)
