"""Utility functions for CLI parsing
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import kipoi


# parsers

def add_source(parser, default="kipoi"):
    parser.add_argument('--source', default=default,
                        choices=list(kipoi.config.model_sources().keys()),
                        help='Model source to use (default={}). Specified in ~/.kipoi/config.yaml'.format(default) +
                        " under model_sources. " +
                        "When 'dir' is used, use the local directory path when specifying the model/dataloader.")


def add_model(parser, source="kipoi"):
    parser.add_argument('model', help='Model name.')
    add_source(parser, default=source)


def add_dataloader_main(parser, with_args=True):
    parser.add_argument('dataloader', help='Dataloader name.')
    parser.add_argument('--source', default="kipoi",
                        choices=list(kipoi.config.model_sources().keys()),
                        help='Dataloader source to use. Specified in ~/.kipoi/config.yaml' +
                        " under model_sources. " +
                        "'dir' is an additional source referring to the local folder.")
    if with_args:
        parser.add_argument('--dataloader_args',
                            help="Dataloader arguments either as a json string:'{\"arg1\": 1} or " +
                            "as a file path to a json file")


def add_dataloader(parser, with_args=True):
    parser.add_argument('--dataloader', default=None,
                        help="Dataloader name. If not specified, the model's default" +
                        "DataLoader will be used")
    parser.add_argument('--dataloader_source', default="kipoi",
                        help="Dataloader source")

    if with_args:
        parser.add_argument('--dataloader_args',
                            help="Dataloader arguments either as a json string:" +
                            "'{\"arg1\": 1} or as a file path to a json file")


# Multiple models/dataloaders
def add_env_args(parser, source="kipoi"):
    parser.add_argument('model', nargs="+",
                        help='Model name(s). You can use <source>::<model> to use models from different sources. \n'
                        '<model> can also refer to a model-group - e.g. if you specify MaxEntScan then the dependencies\n'
                        'for MaxEntScan/5prime and MaxEntScan/3prime will be installed')
    add_source(parser, default=source)
    parser.add_argument('--dataloader', default=[], nargs="+",
                        help="Dataloader name(s). If not specified, the model's default " +
                        "Dataloader will be used. You can use <source>::<dataloader> to use dataloaders from different sources\n"
                        "As for the --model tag, you can specify whole dataloader groups.")
    parser.add_argument("--vep", action="store_true",
                        help="Include also the dependencies for variant effect prediction")
    parser.add_argument("--gpu", action="store_true",
                        help="Use gpu-compatible dependencies. Example: instead " +
                        "of using 'tensorflow', 'tensorflow-gpu' will be used")


# other utils
def file_exists(fpath, logger):
    if not os.path.exists(fpath):
        logger.error("File {0} doesn't exist".format(fpath))
        sys.exit(1)


def dir_exists(dirname, logger):
    if not os.path.exists(os.path.abspath(dirname)):
        logger.error("Directory {0} doesn't exist".format(dirname))
        sys.exit(1)


def parse_source_name(source, name):
    """Parse strings in form: <special_source>::<name> into (<special_source>, <name>)

    If :: is not present in the string, return (<source>, <name>)
    """
    if "::" in name:
        try:
            source, name = name.split("::")
        except ValueError:
            raise ValueError("The name: {0} couldn't be properly parsed. ".format(name) +
                             "Use the following synthax: <source>::<model/dataloader> or <model/dataloader>")

    # Check that the source is correctly specified
    available_sources = list(kipoi.config.model_sources().keys())
    if source not in available_sources:
        raise ValueError("Source: {0} is not present in the available sources: {1}".
                         format(source, available_sources))

    return (source, name)
