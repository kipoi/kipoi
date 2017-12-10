"""Command-line interface for kipoi ls
"""
from __future__ import absolute_import
from __future__ import print_function

import kipoi


def cli_ls(cmd, raw_args):
    """List all kipoi-induced conda environments
    """
    dtm = kipoi.list_models()
    for m in list(dtm.source.str.cat(dtm.model, sep=":")):
        print(m)
