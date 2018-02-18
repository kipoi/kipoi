"""test: kipoi test-source
"""

from kipoi.cli.source_test import modified_files
from kipoi.remote import list_softlink_dependencies
import kipoi
import os


def test_list_softlink_dependencies():
    """Test if finding model dependencies works
    """
    component_dir = kipoi.get_source("kipoi").local_path
    assert list_softlink_dependencies(os.path.join(component_dir, 'rbp_eclip/UPF1'),
                                      component_dir) == {'rbp_eclip/template'}
    assert list_softlink_dependencies(os.path.join(component_dir, 'HAL'),
                                      component_dir) == {'MaxEntScan/template',
                                                         'MaxEntScan/template/example_files',
                                                         'labranchor/example_files'}
    assert list_softlink_dependencies(os.path.join(component_dir, 'deepTarget'),
                                      component_dir) == set()

def dont_test_diff():
    git_range = ["master", "HEAD"]
    local_path = "/home/avsec/.kipoi/models"
    modified_files(["master", "HEAD"], "/home/avsec/.kipoi/models", relative=True)

    import subprocess as sp
    sp.call(['git', 'diff', '--relative=/home/avsec/.kipoi/models',
             '--name-only', 'master...HEAD',
             '--', '/home/avsec/.kipoi/models/*', '/home/avsec/.kipoi/models/*/*'])
