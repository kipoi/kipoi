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
from kipoi.sources import list_subcomponents
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
                                     description='script to test model zoo submissions. Example usage:\n'
                                     '`kipoi test model/directory`, where `model/directory` is the '
                                     'path to a directory containing a model.yaml file.')
    add_model(parser, source="dir")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    args = parser.parse_args(raw_args)
    # --------------------------------------------
    mh = kipoi.get_model(args.model, args.source)

    if not mh._sufficient_deps(mh.dependencies):
        # model requirements should be installed
        logger.warn("Required package '{0}' for model type: {1} is not listed in the dependencies".
                    format(mh.MODEL_PACKAGE, mh.type))

    # Load the test files from model source
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
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-o", "--output", required=True,
                        help="Output hdf5 file")
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)

    dir_exists(os.path.dirname(args.output), logger)
    # --------------------------------------------
    Dataloader = kipoi.get_dataloader_factory(args.dataloader, args.source)

    dataloader_kwargs = kipoi.pipeline.validate_kwargs(Dataloader, dataloader_kwargs)
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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-k", "--keep_inputs", action='store_true',
                        help="Keep the inputs in the output file. ")
    parser.add_argument("-l", "--layer",
                        help="Which output layer to use to make the predictions. If specified," +
                        "`model.predict_activation_on_batch` will be invoked instead of `model.predict_on_batch`")
    parser.add_argument('-o', '--output', required=True, nargs="+",
                        help="Output files. File format is inferred from the file path ending. Available file formats are: " +
                        ", ".join(["." + k for k in writers.FILE_SUFFIX_MAP]))
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)

    # setup the files
    if not isinstance(args.output, list):
        args.output = [args.output]
    for o in args.output:
        ending = o.split('.')[-1]
        if ending not in writers.FILE_SUFFIX_MAP:
            logger.error("File ending: {0} for file {1} not from {2}".
                         format(ending, o, writers.FILE_SUFFIX_MAP))
            sys.exit(1)
        dir_exists(os.path.dirname(o), logger)
    # --------------------------------------------
    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    if args.dataloader is not None:
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
    else:
        Dl = model.default_dataloader

    dataloader_kwargs = kipoi.pipeline.validate_kwargs(Dl, dataloader_kwargs)
    dl = Dl(**dataloader_kwargs)

    # setup batching
    it = dl.batch_iter(batch_size=args.batch_size,
                       num_workers=args.num_workers)

    # Setup the writers
    use_writers = []
    for output in args.output:
        ending = output.split('.')[-1]
        W = writers.FILE_SUFFIX_MAP[ending]
        logger.info("Using {0} for file {1}".format(W.__name__, output))
        if ending == "tsv":
            assert W == writers.TsvBatchWriter
            use_writers.append(writers.TsvBatchWriter(file_path=output, nested_sep="/"))
        elif ending == "bed":
            assert W == writers.BedBatchWriter
            use_writers.append(writers.BedBatchWriter(file_path=output,
                                                      dataloader_schema=dl.output_schema.metadata,
                                                      header=True))
        elif ending in ["hdf5", "h5"]:
            assert W == writers.HDF5BatchWriter
            use_writers.append(writers.HDF5BatchWriter(file_path=output))
        else:
            logger.error("Unknown file format: {0}".format(ending))
            sys.exit(1)

    # Loop through the data, make predictions, save the output
    for i, batch in enumerate(tqdm(it)):
        # validate the data schema in the first iteration
        if i == 0 and not Dl.output_schema.compatible_with_batch(batch):
            logger.warn("First batch of data is not compatible with the dataloader schema.")

        # make the prediction
        if args.layer is None:
            pred_batch = model.predict_on_batch(batch['inputs'])
        else:
            pred_batch = model.predict_activation_on_batch(batch['inputs'], layer=args.layer)

        # write out the predictions, metadata (, inputs, targets)
        output_batch = prepare_batch(batch, pred_batch, keep_inputs=args.keep_inputs)
        for writer in use_writers:
            writer.batch_write(output_batch)

    for writer in use_writers:
        writer.close()
    logger.info('Done! Predictions stored in {0}'.format(",".join(args.output)))


def cli_pull(command, raw_args):
    """Pull the repository
    """
    assert command == "pull"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description="Downloads the directory" +
                                     " associated with the model.")
    parser.add_argument('model', help='Model name. '
                        '<model> can also refer to a model-group - e.g. if you '
                        'specify MaxEntScan then the dependencies\n'
                        'for MaxEntScan/5prime and MaxEntScan/3prime will be installed')
    add_source(parser)
    args = parser.parse_args(raw_args)

    src = kipoi.config.get_source(args.source)
    sub_models = list_subcomponents(args.model, args.source, which='model')
    if len(sub_models) == 0:
        logger.error("Model {0} not found in source {1}".format(args.model, args.source))
        sys.exit(1)
    if len(sub_models) > 1:
        logger.info("Found {0} models under the model name: {1}. Pulling all of them".
                    format(len(sub_models), args.model))
    for sub_model in sub_models:
        src.pull_model(sub_model)


def cli_init(command, raw_args, **kwargs):
    """Initialize the repository using cookiecutter
    """
    assert command == "init"
    logger.info("Initializing a new Kipoi model")

    print("\nPlease answer the questions below. Defaults are shown in square brackets.\n")
    print("You might find the following links useful: ")
    print("- getting started: http://www.kipoi.org/docs/contributing/01_Getting_started/")
    print("- model_type: http://www.kipoi.org/docs/contributing/02_Writing_model.yaml/#type")
    print("- dataloader_type: http://www.kipoi.org/docs/contributing/04_Writing_dataloader.py/#dataloader-types")
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


def cli_info(command, raw_args):
    """CLI interface to predict
    """
    assert command == "info"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description="Prints dataloader" +
                                                 " keyword arguments.")
    add_model(parser)
    add_dataloader(parser, with_args=False)
    args = parser.parse_args(raw_args)

    # --------------------------------------------
    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    if args.dataloader is not None:
        dl_info = "dataloader '{0}' from source '{1}'".format(str(args.dataloader), str(args.dataloader_source))
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
    else:
        dl_info = "default dataloader for model '{0}' from source '{1}'".format(str(model.name), str(args.source))
        Dl = model.default_dataloader

    print("-" * 80)
    print("Displaying keyword arguments for {0}".format(dl_info))
    print(Dl.print_args())
    print("-" * 80)


def cli_list_plugins(command, raw_args):
    """CLI interface to predict
    """
    assert command == "list_plugins"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description="Lists available pluging")
    parser.parse_args(raw_args)
    print(kipoi.list_plugins().to_string(index=False, justify="unset"))


def cli_ls(command, raw_args):
    """List all kipoi models
    """
    assert command == "ls"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description="Lists available models")
    parser.add_argument("group_filter", nargs='?', default='',
                        help="A relative path to the model group used to subset the model list. Use 'all' to show all models")
    parser.add_argument("--tsv", action='store_true',
                        help="Print the output in the tsv format.")
    add_source(parser)
    args = parser.parse_args(raw_args)
    grp = kipoi.get_source(args.source)
    df = grp.list_models()
    ls_helper(df, args.group_filter, args.tsv)


#  split it up for easier testing

def ls_helper(df, group_filter='', tsv=False):
    if group_filter == 'all':
        for m in list(df.model):
            print(m)
    else:
        from kipoi.sources import list_models_by_group
        dfg = list_models_by_group(df, group_filter)
        if dfg is None:
            # print the model list
            models = df.model[df.model.str.contains(group_filter)]
            for m in models:
                print(m)
        else:
            if tsv:
                dfg[['group', 'N_models', 'N_subgroups']].to_csv(sys.stdout, sep='\t', index=False)
            else:
                for i, row in dfg.iterrows():
                    if row.N_subgroups == 0 and row.N_models == 1:
                        print("{}".format(row.group))
                    else:
                        print("{}/ ({})".format(row.group, row.N_models))
