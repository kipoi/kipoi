"""Whole model pipeline: extractor + model
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import logging
import os
import yaml
from .utils import pip_install_requirements, parse_json_file_str
import kipoi  # for .config module
from .data import numpy_collate, numpy_collate_concat
from torch.utils.data import DataLoader
# import h5py
# import six
import numpy as np
import pandas as pd
from tqdm import tqdm
import deepdish
from collections import OrderedDict
# HACK prevent this issue: https://github.com/kundajelab/genomelake/issues/4
import genomelake

# TODO - write out the hdf5 file in batches:
#        - need a recursive function for creating groups ...
#           - infer the right data-type + shape

# TODO - handle environment creation

_logger = logging.getLogger('kipoi')


# TODO - remove these as they are duplicates...
# PREPROC_FIELDS = ['function_name', 'type', 'arguments']
# PREPROC_TYPES = ['generator', 'return']
# PREPROC_IFILE_TYPES = ['DNA_regions']
# PREPROC_IFILE_FORMATS = ['bed3']
# MODEL_FIELDS = ['inputs', 'targets']
# DATA_TYPES = ['dna', 'bigwig', 'v-plot']
# RESERVED_PREPROC_KWS = ['intervals_file']


def install_model_requirements(model_dir):
    pip_install_requirements(os.path.join(model_dir, 'requirements.txt'))


class Pipeline(object):
    """Provides the test_predict, predict and predict_generator to the kipoi.Model
    """

    def __init__(self, model, dataloader_cls):
        self.model = model
        self.dataloader_cls = dataloader_cls

        # TODO - validate if model and datalaoder_cls are compatible

    def test_predict(self, dataloader_kwargs, batch_size=32, test_equal=False):
        _logger.info('Initialized data generator. Running batches...')

        dl = self.dataloader_cls(**dataloader_kwargs)
        _logger.info('Returned data schema correct')

        # TODO - use batch_iter
        it = DataLoader(dl, batch_size=batch_size, collate_fn=numpy_collate)

        # test that all predictions go through
        for i, batch in enumerate(tqdm(it)):
            self.model.predict_on_batch(batch['inputs'])

        # ?TODO? - check that the predicted values match the targets

        #     if test_equal:
        #         match.append(compare_numpy_dict(y_pred, batch['targets'], exact=False))
        # if not all(match):
        #     _logger.warning("For {0}/{1} batch samples: target != model(inputs)")
        # else:
        #     _logger.info("All target values match model predictions")

        _logger.info('test_predict done!')

    def predict(self, dataloader_kwargs, batch_size=32):
        """
        # Arguments
            preproc_kwargs: Keyword arguments passed to the pre-processor

        :return: Predict the whole array
        """
        return numpy_collate_concat(list(self.predict_generator(dataloader_kwargs,
                                                                batch_size)))

    def predict_generator(self, dataloader_kwargs, batch_size=32):
        """Prediction generator

        # Arguments
            preproc_kwargs: Keyword arguments passed to the pre-processor

        # Yields
            model batch prediction
        """
        _logger.info('Initialized data generator. Running batches...')

        # TODO - implement batch_iter
        it = DataLoader(self.dataloader_cls(**dataloader_kwargs),
                        batch_size=batch_size, collate_fn=numpy_collate)

        for i, batch in enumerate(it):
            yield self.model.predict_on_batch(batch['inputs'])


def cli_test(command, raw_args):
    """CLI interface
    """
    assert command == "test"
    # setup the arg-parsing
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='script to test model zoo submissions')
    parser.add_argument('model', help='Model name.')
    parser.add_argument('--source', default="dir",
                        choices=list(kipoi.config.model_sources().keys()) + ["dir"],
                        help='Model source to use. Specified in ~/.kipoi/config.yaml' +
                        " under model_sources")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-i", "--install_req", action='store_true',
                        help="Install required packages from requirements.txt")
    args = parser.parse_args(raw_args)
    # --------------------------------------------

    mh = kipoi.Model(args.model, args.source)
    if args.install_req:
        pip_install_requirements(os.path.join(mh.source_dir, 'requirements.txt'))
    # force the requirements to be installed

    test_dir = os.path.join(mh.source_dir, 'test_files')

    if os.path.exists(test_dir):
        _logger.info(
            'Found test files in {}. Initiating test...'.format(test_dir))
        # cd to test directory
        os.chdir(test_dir)

    with open(os.path.join(test_dir, 'test.json')) as f_kwargs:
        dataloader_kwargs = yaml.load(f_kwargs)
    match = mh.pipeline.test_predict(dataloader_kwargs, batch_size=args.batch_size)
    # if not match:
    #     # _logger.error("Expected targets don't match model predictions")
    #     raise Exception("Expected targets don't match model predictions")

    _logger.info('Successfully ran test_predict')


def cli_extract_to_hdf5(command, raw_args):
    """CLI interface to run the dataloader
    """
    assert command == "preproc"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Run the dataloader and save the output to an hdf5 file.')
    parser.add_argument('model', help='Model name.')
    parser.add_argument('--source', default="kipoi",
                        choices=list(kipoi.config.model_sources().keys()) + ["dir"],
                        help='Model source to use. Specified in ~/.kipoi/config.yaml' +
                        " under model_sources")
    parser.add_argument('--dataloader_args',
                        help="Dataloader arguments either as a json string:'{\"arg1\": 1} or " +
                        "as a file path to a json file")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-i", "--install-req", action='store_true',
                        help="Install required packages from requirements.txt")
    # parser.add_argument("-n", "--num_workers", type=int, default=1,
    #                     help="Number of parallel workers for loading the dataset")
    parser.add_argument("-o", "--output",
                        help="Output hdf5 file")
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)
    # --------------------------------------------
    # install args
    Dataloader = kipoi.DataLoader_factory(args.model, args.source)
    if args.install_req:
        pip_install_requirements(os.path.join(Dataloader.source_dir, 'requirements.txt'))

    dataloader = Dataloader(**dataloader_kwargs)

    _logger.info("Loading all the points into memory")
    obj = numpy_collate_concat([x for x in tqdm(dataloader)])

    _logger.info("Writing everything to the hdf5 array at {0}".format(args.output))
    deepdish.io.save(args.output, obj)
    _logger.info("Done!")

    # TODO - hack - read the whole dataset into memory at first...

    # sample = dataloader[0]
    # n = len(dataloader)

    # f = h5py.File(args.output, "w")
    # inputs = f.create_group("inputs")
    # arr_inputs = {k: inputs.create_dataset(k, (n, ) + v.shape) for k, v in six.iteritems(sample["inputs"])}
    # targets = f.create_group("targets")

    # if "targets" in sample and isinstance(sample["targets"], dict):
    #     arr_inputs = {k: inputs.create_dataset(k, (n, ) + v.shape) for k, v in six.iteritems(sample["targets"])}
    # else:
    #     arr_targets = None

    # metadata = f.create_group("metadata")
    # arr_inputs = {k: inputs.create_dataset(k, (n, ) + np.asarray(v)) for k, v in six.iteritems(sample["metadata"])}


def cli_predict(command, raw_args):
    """CLI interface to predict
    """
    assert command == "predict"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Run the model prediction.')
    parser.add_argument('model', help='Model name.')
    parser.add_argument('--source', default="kipoi",
                        choices=list(kipoi.config.model_sources().keys()) + ["dir"],
                        help='Model source to use. Specified in ~/.kipoi/config.yaml' +
                        " under model_sources")
    parser.add_argument('--dataloader', default=None,
                        help="Dataloader name. If not specified, the model's default" +
                        "DataLoader will be used")
    parser.add_argument('--dataloader_source', default="kipoi",
                        help="Dataloader source. If not specified, the model's default" +
                        "DataLoader will be used")
    parser.add_argument('--dataloader_args',
                        help="Dataloader arguments either as a json string:" +
                        "'{\"arg1\": 1} or as a file path to a json file")
    parser.add_argument('-f', '--file_format', default="tsv",
                        choices=["tsv", "bed", "hdf5"],
                        help='File format.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=1,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-i", "--install-req", action='store_true',
                        help="Install required packages from requirements.txt")
    parser.add_argument("-k", "--keep_inputs", action='store_true',
                        help="Keep the inputs in the output file. " +
                        "Only compatible with hdf5 file format")
    parser.add_argument('-o', '--output',
                        help="Output hdf5 file")
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)

    if args.keep_inputs and args.file_format != "hdf5":
        raise ValueError("--keep_inputs flag is only compatible with --file_format=hdf5")
    # --------------------------------------------
    # load model & dataloader
    model = kipoi.Model(args.model, args.source)

    # install args
    if args.install_req:
        pip_install_requirements(os.path.join(model.source_dir, 'requirements.txt'))

    if args.dataloader is not None:
        Dl = kipoi.DataLoader_factory(args.dataloader, args.dataloader_source)
    else:
        Dl = model.default_dataloader

    dl = Dl(**dataloader_kwargs)

    # setup batching
    it = DataLoader(dl, batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    collate_fn=numpy_collate)

    obj_list = []
    for i, batch in enumerate(tqdm(it)):
        pred_batch = model.predict_on_batch(batch['inputs'])

        # tabular files
        if args.file_format in ["tsv", "bed"]:
            df = io_batch2df(batch, pred_batch)
            if i == 0:
                df.to_csv(args.output, sep="\t", index=False)
            else:
                df.to_csv(args.output, sep="\t", index=False, header=None, mode="a")

        # binary nested arrays
        elif args.file_format == "hdf5":
            batch["predictions"] = pred_batch
            if not args.keep_inputs:
                del batch["inputs"]
            # TODO - implement the batching version of it
            obj_list.append(batch)
        else:
            raise ValueError("Unknown file format: {0}".format(args.file_format))

    # Write hdf5 file in bulk
    if args.file_format == "hdf5":
        deepdish.io.save(args.output, numpy_collate_concat(obj_list))

    _logger.info('Successfully predictde samples')


def io_batch2df(batch, pred_batch):
    """Convert the batch + prediction batch to a pd.DataFrame
    """
    if not isinstance(pred_batch, np.ndarray) or pred_batch.ndim > 2:
        raise ValueError("Model's output is not a 1D or 2D np.ndarray")

    # TODO - generalize to multiple arrays (list of arrays)

    if pred_batch.ndim == 1:
        pred_batch = pred_batch[:, np.newaxis]
    df = pd.DataFrame(pred_batch,
                      columns=["y_{0}".format(i)
                               for i in range(pred_batch.shape[1])])

    if "metadata" in batch and "ranges" in batch["metadata"]:
        rng = batch["metadata"]["ranges"]
        df_ranges = pd.DataFrame(OrderedDict([("chr", rng.get("chr")),
                                              ("start", rng.get("start")),
                                              ("end", rng.get("end")),
                                              ("name", rng.get("id", None)),
                                              ("score", rng.get("score", None)),
                                              ("strand", rng.get("strand", None))]))
        df = pd.concat([df_ranges, df], axis=1)
    return df


def cli_pull(command, raw_args):
    """Pull the repository
    """
    assert command == "pull"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description="Downloads the directory associated with the model.")
    parser.add_argument('model', help='Model name.')
    parser.add_argument('--source', default="kipoi",
                        choices=list(kipoi.config.model_sources().keys()) + ["dir"],
                        help='Model source to use. Specified in ~/.kipoi/config.yaml ' +
                        "under model_sources")

    args = parser.parse_args(raw_args)

    kipoi.config.get_source(args.source).pull_model(args.model)
