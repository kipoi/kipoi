"""Main CLI commands
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import json
import sys
import os
from kipoi_utils import parse_json_file_str_or_arglist, cd, makedir_exist_ok, compare_numpy_dict
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

import ast

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
    parser.add_argument("-o", "--output", default=None, required=False,
                        help="Output hdf5 file")
    parser.add_argument("-s", "--skip-expect", action='store_true',
                        help="Skip validating the expected predictions if test.expect field is specified under model.yaml")
    parser.add_argument("-e", "--expect", default=None,
                        help="File path to the hdf5 file of predictions produced by kipoi test -o file.h5 "
                        "or kipoi predict -o file.h5 --keep_inputs. Overrides test.expect in model.yaml")
    args = parser.parse_args(raw_args)
    # --------------------------------------------
    mh = kipoi.get_model(args.model, args.source)

    if not mh._sufficient_deps(mh.dependencies):
        # model requirements should be installed
        logger.warning("Required package '{0}' for model type: {1} is not listed in the dependencies".
                    format(mh.MODEL_PACKAGE, mh.type))

    # Load the test files from model source
    mh.pipeline.predict_example(batch_size=args.batch_size, output_file=args.output)

    if (mh.test.expect is not None or args.expect is not None) \
            and not args.skip_expect and args.output is None:
        if args.expect is not None:
            # `expect` specified from the CLI
            expect = args.expect
        else:
            # `expect` taken from model.yaml
            if isinstance(mh.test.expect, kipoi.specs.RemoteFile):
                # download the file
                output_dir = kipoi.get_source(args.source).get_model_download_dir(args.model)
                makedir_exist_ok(output_dir)
                mh.test.expect = mh.test.expect.get_file(os.path.join(output_dir, 'test.expect.h5'))
            expect = mh.test.expect
        logger.info('Testing if the predictions match the expected ones in the file: {}'.format(expect))
        logger.info('Desired precision (number of matching decimal places): {}'.format(mh.test.precision_decimal))

        # iteratively load the expected file
        expected = kipoi.readers.HDF5Reader(expect)
        expected.open()
        it = expected.batch_iter(batch_size=args.batch_size)
        for i, batch in enumerate(tqdm(it, total=len(expected) // args.batch_size)):
            if i == 0 and ('inputs' not in batch or 'preds' not in batch):
                raise ValueError("test.expect file requires 'inputs' and 'preds' "
                                 "to be specified. Available keys: {}".format(list(expected)))
            pred_batch = mh.predict_on_batch(batch['inputs'])
            # compare to the predictions
            # import ipdb
            # ipdb.set_trace()
            try:
                compare_numpy_dict(pred_batch, batch['preds'], exact=False, decimal=mh.test.precision_decimal)
            except Exception as e:
                logger.error("Model predictions don't match the expected predictions."
                             "expected: {}\nobserved: {}. Exception: {}".format(batch['preds'], pred_batch, e))
                expected.close()
                sys.exit(1)
        expected.close()
        logger.info('All predictions match')
    logger.info('Successfully ran test_predict')


def cli_get_example(command, raw_args):
    """Downloads the example files to the desired directory
    """
    assert command == "get-example"
    # setup the arg-parsing
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Get example files')
    add_model(parser, source="kipoi")
    parser.add_argument("-o", "--output", default="example", required=False,
                        help="Output directory where to store the examples. Default: 'example'")
    args = parser.parse_args(raw_args)
    # --------------------------------------------
    md = kipoi.get_model_descr(args.model, args.source)
    src = kipoi.get_source(args.source)

    # load the default dataloader
    if isinstance(md.default_dataloader, kipoi.specs.DataLoaderImport):
        with cd(src.get_model_dir(args.model)):
            dl_descr = md.default_dataloader.get()
    else:
        # load from directory
        # attach the default dataloader already to the model
        dl_descr = kipoi.get_dataloader_descr(os.path.join(args.model, md.default_dataloader),
                                              source=args.source)

    kwargs = dl_descr.download_example(output_dir=args.output, dry_run=False)

    logger.info("Example files downloaded to: {}".format(args.output))
    logger.info("use the following dataloader kwargs:")
    print(json.dumps(kwargs))


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

    dataloader_kwargs = parse_json_file_str_or_arglist(args.dataloader_args, parser)

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
        if i == 0 and not Dataloader.get_output_schema().compatible_with_batch(batch):
            logger.warning("First batch of data is not compatible with the dataloader schema.")
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
    parser.add_argument("--singularity", action='store_true',
                        help="Run `kipoi predict` in the appropriate singularity container. "
                        "Containters will get downloaded to ~/.kipoi/envs/ or to "
                        "$SINGULARITY_CACHEDIR if set")
    parser.add_argument('-o', '--output', required=True, nargs="+",
                        help="Output files. File format is inferred from the file path ending. Available file formats are: " +
                        ", ".join(["." + k for k in writers.FILE_SUFFIX_MAP]))
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str_or_arglist(args.dataloader_args, parser)

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

    # singularity_command
    if args.singularity:
        from kipoi.cli.singularity import singularity_command
        logger.info("Running kipoi predict in the singularity container")
        # Drop the singularity flag
        raw_args = [x for x in raw_args if x != '--singularity']
        singularity_command(['kipoi', command] + raw_args,
                            args.model,
                            dataloader_kwargs,
                            output_files=args.output,
                            source=args.source,
                            dry_run=False)
        return None
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
        writer = writers.get_writer(output, metadata_schema=dl.get_output_schema().metadata)
        if writer is None:
            logger.error("Unknown file format: {0}".format(ending))
            sys.exit()
        else:
            use_writers.append(writer)
    output_writers = writers.MultipleBatchWriter(use_writers)

    # Loop through the data, make predictions, save the output
    for i, batch in enumerate(tqdm(it)):
        # validate the data schema in the first iteration
        if i == 0 and not Dl.get_output_schema().compatible_with_batch(batch):
            logger.warning("First batch of data is not compatible with the dataloader schema.")

        # make the prediction
        if args.layer is None:
            pred_batch = model.predict_on_batch(batch['inputs'])
        else:
            pred_batch = model.predict_activation_on_batch(batch['inputs'], layer=args.layer)

        # write out the predictions, metadata (, inputs, targets)
        output_batch = prepare_batch(batch, pred_batch, keep_inputs=args.keep_inputs)
        output_writers.batch_write(output_batch)

    output_writers.close()
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
    md = kipoi.get_model_descr(args.model, args.source)
    src = kipoi.get_source(args.source)

    # load the default dataloader
    if isinstance(md.default_dataloader, kipoi.specs.DataLoaderImport):
        with cd(src.get_model_dir(args.model)):
            dl_descr = md.default_dataloader.get()
    else:
        # load from directory
        # attach the default dataloader already to the model
        dl_descr = kipoi.get_dataloader_descr(os.path.join(args.model, md.default_dataloader),
                                              source=args.source)

    print("-" * 80)
    print("'{0}' from source '{1}'".format(str(args.model), str(args.source)))
    print("")
    print("Model information")
    print("-----------")
    print(md.info.get_config_as_yaml())
    print("Dataloader arguments")
    print("--------------------")
    dl_descr.print_args()
    print("--------------------\n")
    print("Run `kipoi get-example {} -o example` to download example files.\n".format(args.model))


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
