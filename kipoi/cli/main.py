"""Main CLI commands
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import sys
import os
from ..utils import parse_json_file_str, cd
import kipoi  # for .config module
from kipoi.cli.parser_utils import add_model, add_source, add_dataloader, add_dataloader_main, file_exists, dir_exists
from ..data import numpy_collate_concat
# import h5py
# import six
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import logging
from kipoi import writers
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# TODO - write out the hdf5 file in batches:
#        - need a recursive function for creating groups ...
#           - infer the right data-type + shape


def prepare_batch(dl_batch, pred_batch,
                  keep_inputs=False):
    dl_batch["preds"] = pred_batch

    if not keep_inputs:
        dl_batch.pop("inputs", None)
        dl_batch.pop("targets", None)
    return dl_batch


def cli_test(command, raw_args):
    """Runs test on the model
    """
    assert command == "test"
    # setup the arg-parsing
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='script to test model zoo submissions')
    add_model(parser, source="dir")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-i", "--install_req", action='store_true',
                        help="Install required packages from requirements.txt")
    args = parser.parse_args(raw_args)
    # --------------------------------------------
    if args.install_req:
        kipoi.pipeline.install_model_requirements(args.model,
                                                  args.source,
                                                  and_dataloaders=True)
    mh = kipoi.get_model(args.model, args.source)
    # force the requirements to be installed

    # Load the test files from model source
    # with cd(mh.source_dir):
    mh.pipeline.predict_example(batch_size=args.batch_size)
    # if not match:
    #     # logger.error("Expected targets don't match model predictions")
    #     raise Exception("Expected targets don't match model predictions")

    logger.info('Successfully ran test_predict')


def cli_preproc(command, raw_args):
    """Preprocess:
    - Run the dataloader and store the results to a (hdf5) file
    """
    assert command == "preproc"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Run the dataloader and save the output to an hdf5 file.')
    add_dataloader_main(parser, with_args=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in data loading')
    parser.add_argument("-i", "--install_req", action='store_true',
                        help="Install required packages from requirements.txt")
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-o", "--output", required=True,
                        help="Output hdf5 file")
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)

    dir_exists(os.path.dirname(args.output), logger)
    # --------------------------------------------
    # install args
    if args.install_req:
        kipoi.pipeline.install_dataloader_requirements(args.dataloader, args.source)
    Dataloader = kipoi.get_dataloader_factory(args.dataloader, args.source)

    dataloader = Dataloader(**dataloader_kwargs)

    it = dataloader.batch_iter(batch_size=args.batch_size, num_workers=args.num_workers)

    logger.info("Writing to the hdf5 file: {0}".format(args.output))
    writer = writers.HDF5BatchWriter(file_path=args.output)

    for i, batch in enumerate(tqdm(it)):
        # check that the first batch was indeed correct
        if i == 0 and not Dataloader.output_schema.compatible_with_batch(batch):
            logger.warn("First batch of data is not compatible with the dataloader schema.")
        writer.batch_write(batch)

    writer.close()
    logger.info("Done!")


def cli_predict(command, raw_args):
    """CLI interface to predict
    """
    assert command == "predict"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Run the model prediction.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('-f', '--file_format', default="tsv", nargs="+",
                        choices=["tsv", "bed", "hdf5"],
                        help='Output file formats to use (can contain multiple).')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-i", "--install_req", action='store_true',
                        help="Install required packages from requirements.txt")
    parser.add_argument("-k", "--keep_inputs", action='store_true',
                        help="Keep the inputs in the output file. ")
    parser.add_argument('-o', '--output', required=True,
                        help="Output file(s). In case multiple file_formats are specified, the" +
                        "output file name for each file_format is then: '{output}.{file_format}'")
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)

    if args.keep_inputs and args.file_format != "hdf5":
        raise ValueError("--keep_inputs flag is only compatible with --file_format=hdf5")

    dir_exists(os.path.dirname(args.output), logger)

    if not isinstance(args.file_format, list):
        args.file_format = [args.file_format]
        args.output = [args.output]
    elif len(args.file_format) == 1:
        args.output = [args.output]
    else:
        # multiple file_formats present
        args.output = ["{0}.{1}".format(args.output, f) for f in args.file_format]
    # --------------------------------------------
    # install args
    if args.install_req:
        kipoi.pipeline.install_model_requirements(args.model,
                                                  args.source,
                                                  and_dataloaders=True)
    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    if args.dataloader is not None:
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
    else:
        Dl = model.default_dataloader

    dl = Dl(**dataloader_kwargs)

    # setup batching
    it = dl.batch_iter(batch_size=args.batch_size,
                       num_workers=args.num_workers)

    use_writers = []
    for f, output in zip(args.file_format, args.output):
        if f == "tsv":
            use_writers.append(writers.TsvBatchWriter(file_path=output,
                                                      nested_sep="/"))
        elif f == "bed":
            use_writers.append(writers.BedBatchWriter(file_path=output,
                                                      dataloader_schema=dl.output_schema.metadata,
                                                      header=True))
        elif f == "hdf5":
            use_writers.append(writers.HDF5BatchWriter(file_path=output))
        else:
            raise ValueError("Unknown file format: {0}".format(f))

    for i, batch in enumerate(tqdm(it)):
        if i == 0 and not Dl.output_schema.compatible_with_batch(batch):
            logger.warn("First batch of data is not compatible with the dataloader schema.")
        pred_batch = model.predict_on_batch(batch['inputs'])

        output_batch = prepare_batch(batch, pred_batch)

        # write out the predictions
        for writer in use_writers:
            writer.batch_write(output_batch)

    for writer in use_writers:
        writer.close()
    logger.info('Done! Predictions stored in {0}'.format(args.output))


def cli_pull(command, raw_args):
    """Pull the repository
    """
    assert command == "pull"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description="Downloads the directory" +
                                     " associated with the model.")
    parser.add_argument('model', help='Model name.')
    add_source(parser)
    parser.add_argument('-e', '--env_file', default=None,
                        help='If set, export the conda environment to a file.' +
                        'Example: kipoi pull mymodel -e mymodel.yaml')
    args = parser.parse_args(raw_args)

    kipoi.config.get_source(args.source).pull_model(args.model)

    if args.env_file is not None:
        env = kipoi.cli.env.export_env(args.env_file, args.model, args.source)
        print("Activate the environment with:")
        print("source activate {0}".format(env))


def cli_init(command, raw_args, **kwargs):
    """Initialize the repository using cookiecutter
    """
    assert command == "init"
    logger.info("Initializing a new Kipoi model")

    print("\nPlease answer the questions bellow. Defaults are shown in square brackets.\n")
    print("You might find the following links useful: ")
    print("- (model_type) https://github.com/kipoi/kipoi/blob/master/docs/writing_models.md")
    print("- (dataloader_type) https://github.com/kipoi/kipoi/blob/master/docs/writing_dataloaders.md")
    print("--------------------------------------------\n")

    from cookiecutter.main import cookiecutter
    from cookiecutter.exceptions import FailedHookException

    # Get the information about the current directory
    import inspect
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    this_dir = os.path.dirname(os.path.abspath(filename))
    template_path = os.path.join(this_dir, "../model_template/")

    # remove the pyc files in the template directory
    # bug in cookiecutter: https://github.com/audreyr/cookiecutter/pull/1037
    pyc_file = os.path.join(template_path, "hooks", "pre_gen_project.pyc")
    if os.path.exists(pyc_file):
        os.remove(pyc_file)

    # Create project from the cookiecutter-pypackage/ template
    try:
        out_dir = cookiecutter(template_path, **kwargs)
    except FailedHookException:
        # pre_gen_project.py detected an error in the configuration
        logger.error("Failed to initialize the model")
        sys.exit(1)
    print("--------------------------------------------")
    logger.info("Done!\nCreated the following folder into the current working directory: {0}".format(os.path.basename(out_dir)))
