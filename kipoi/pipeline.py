"""Whole model pipeline: dataloader + model
"""
from __future__ import absolute_import
from __future__ import print_function

import os
from kipoi_utils.utils import cd
import kipoi  # for .config module
from .data import numpy_collate_concat
# import h5py
import six
from tqdm import tqdm
import six
import deprecation
from ._version import __version__
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())



@deprecation.deprecated(deprecated_in="0.6.8", removed_in="0.7.0",
                        current_version=__version__,
                        details=""" installing packages in a running python env is error prone.
                        Use command line interface of kipoi to install packages.
                        """)
def install_model_requirements(model, source="kipoi", and_dataloaders=True):
    """Install model dependencies

    # Arguments
        model (str): model name
        source (str): model source
        and_dataloaders (bool): if True, install also the dependencies
            for the default dataloader
    """
    md = kipoi.get_source(source).get_model_descr(model)
    md.dependencies.install()
    if and_dataloaders:
        if ":" in md.default_dataloader:
            dl_source, dl_path = md.default_dataloader.split(":")
        else:
            dl_source = source
            dl_path = md.default_dataloader

        default_dataloader_path = os.path.join("/" + model, dl_path)[1:]
        dl = kipoi.config.get_source(dl_source).get_dataloader_descr(default_dataloader_path)
        dl.dependencies.install()

@deprecation.deprecated(deprecated_in="0.6.8", removed_in="0.7.0",
                        current_version=__version__,
                        details=""" installing packages in a running python env is error prone.
                        Use command line interface of kipoi to install packages.
                        """)
def install_dataloader_requirements(dataloader, source="kipoi"):
    """Install dataloader dependencies

    # Arguments
        datalaoder (str): dataloader name
        source (str): model source
    """
    kipoi.get_source(source).get_model_descr(dataloader).dependencies.install()


def validate_kwargs(dataloader, dataloader_kwargs):
    # check that the dataloader_kwargs indeed match
    if not isinstance(dataloader_kwargs, dict):
        raise ValueError("Dataloader_kwargs need to be a dictionary")

    missing_arg = []
    req_args = {k for k in dataloader.args
                if not dataloader.args[k].optional}
    missing_arg = req_args - set(dataloader_kwargs.keys())
    if len(missing_arg) > 0:
        logger.warning("Required arguments for the dataloader: {0} were not specified".
                    format(",".join(list(missing_arg))))
    unused = set(dataloader_kwargs.keys()) - set(dataloader.args.keys())
    if len(unused) > 0:
        logger.warning("Some provided dataloader kwargs were not used: {0}".format(unused))
    return {k: v for k, v in six.iteritems(dataloader_kwargs) if k in dataloader.args}


class Pipeline(object):
    """Runs model predictions from raw files:

    ```
    raw files --(dataloader)--> data batches --(model)--> prediction
    ```

    # Arguments
        model: model returned by `kipoi.get_model`
        dataloader_cls: dataloader class returned by `kipoi.get_dataloader_factory`
            of `kipoi.get_model().default_dataloader`
    """

    def __init__(self, model, dataloader_cls):
        self.model = model
        self.dataloader_cls = dataloader_cls

        # validate if model and datalaoder_cls are compatible
        if not self.model.schema.compatible_with_schema(self.dataloader_cls.get_output_schema()):
            logger.warning("dataloader.output_schema is not compatible with model.schema")
        else:
            logger.info("dataloader.output_schema is compatible with model.schema")

    def predict_example(self, batch_size=32, output_file=None):
        """Run model prediction for the example file

        # Arguments
            batch_size: batch_size
            output_file: if not None, inputs and predictions are stored to `output_file` path
            **kwargs: Further arguments passed to batch_iter
        """
        logger.info('Initialized data generator. Running batches...')

        from kipoi.writers import get_writer
        from kipoi.cli.main import prepare_batch

        if output_file is not None:
            output_file = os.path.abspath(output_file)
            if os.path.exists(output_file):
                raise ValueError("Output file: {} already exists.".format(output_file))
        with cd(self.dataloader_cls.source_dir):
            # init the dataloader
            dl = self.dataloader_cls.init_example()
            logger.info('Returned data schema correct')

            if output_file is not None:
                writer = get_writer(output_file, dl.get_output_schema().metadata)

            it = dl.batch_iter(batch_size=batch_size)

            # test that all predictions go through
            pred_list = []
            for i, batch in enumerate(tqdm(it)):
                if i == 0 and not self.dataloader_cls.get_output_schema().compatible_with_batch(batch):
                    logger.warning("First batch of data is not compatible with the dataloader schema.")
                pred_batch = self.model.predict_on_batch(batch['inputs'])
                pred_list.append(pred_batch)

                if output_file is not None:
                    output_batch = prepare_batch(batch, pred_batch, keep_inputs=True)
                    writer.batch_write(output_batch)

            if output_file is not None:
                writer.close()

        logger.info('predict_example done!')
        return numpy_collate_concat(pred_list)

    def predict(self, dataloader_kwargs, batch_size=32, **kwargs):
        """
        # Arguments
            dataloader_kwargs: Keyword arguments passed to the pre-processor
            **kwargs: Further arguments passed to batch_iter

        # Returns
            np.array, dict, list: Predict the whole array
        """
        pred_list = [batch for batch in tqdm(self.predict_generator(dataloader_kwargs,
                                                                    batch_size, **kwargs))]
        return numpy_collate_concat(pred_list)

    def predict_generator(self, dataloader_kwargs, batch_size=32, layer=None, **kwargs):
        """Prediction generator

        # Arguments
            dataloader_kwargs: Keyword arguments passed to the dataloader
            batch_size: Size of batches produced by the dataloader
            layer: If not None activation of specified layer will be returned. Only possible for models that are a
            subclass of `LayerActivationMixin`.
            **kwargs: Further arguments passed to batch_iter

        # Yields
        - `dict`: model batch prediction
        """
        logger.info('Initialized data generator. Running batches...')

        validate_kwargs(self.dataloader_cls, dataloader_kwargs)
        it = self.dataloader_cls(**dataloader_kwargs).batch_iter(batch_size=batch_size, **kwargs)

        from .model import LayerActivationMixin
        if layer is not None and not isinstance(self.model, LayerActivationMixin):
            raise Exception("Attempting to extract layer activation (argument `layer` is not None) on a model that"
                            " is not a subclass of `LayerActivationMixin`.")

        for i, batch in enumerate(it):
            if i == 0 and not self.dataloader_cls.get_output_schema().compatible_with_batch(batch):
                logger.warning("First batch of data is not compatible with the dataloader schema.")
            if layer is None:
                yield self.model.predict_on_batch(batch['inputs'])
            else:
                yield self.model.predict_activation_on_batch(batch['inputs'], layer=layer)

    def predict_to_file(self, output_file, dataloader_kwargs, batch_size=32, keep_inputs=False, **kwargs):
        """Make predictions and write them iteratively to a file

        # Arguments
            output_file: output file path. File format is inferred from the file path ending. Available file formats are:
                 'bed', 'h5', 'hdf5', 'tsv'
            dataloader_kwargs: Keyword arguments passed to the dataloader
            batch_size: Batch size used for the dataloader
            keep_inputs: if True, inputs and targets will also be written to the output file.
            **kwargs: Further arguments passed to batch_iter
        """
        from kipoi.writers import get_writer
        from kipoi.cli.main import prepare_batch

        # setup dataloader
        validate_kwargs(self.dataloader_cls, dataloader_kwargs)
        dl = self.dataloader_cls(**dataloader_kwargs)
        it = dl.batch_iter(batch_size=batch_size, **kwargs)
        writer = get_writer(output_file, dl.get_output_schema().metadata)

        for i, batch in enumerate(tqdm(it)):
            if i == 0 and not self.dataloader_cls.get_output_schema().compatible_with_batch(batch):
                logger.warning("First batch of data is not compatible with the dataloader schema.")
            pred_batch = self.model.predict_on_batch(batch['inputs'])
            output_batch = prepare_batch(batch, pred_batch, keep_inputs=keep_inputs)
            writer.batch_write(output_batch)
        writer.close()

    def input_grad(self, dataloader_kwargs, batch_size=32, filter_idx=None, avg_func=None, layer=None,
                   final_layer=True, selected_fwd_node=None, pre_nonlinearity=False, **kwargs):
        """Get input gradients

        # Arguments
            dataloader_kwargs: Keyword arguments passed to the dataloader
            batch_size: Batch size used for the dataloader
            filter_idx: filter index of `layer` for which the gradient should be returned
            avg_func: String name of averaging function to be applied across filters in layer `layer`
            layer: layer from which backwards the gradient should be calculated
            final_layer: Use the final (classification) layer as `layer`
            selected_fwd_node: None - not supported by KerasModel at the moment
            pre_nonlinearity: Try to use the layer output prior to activation (will not always be possible in an
            automatic way)
            **kwargs: Further arguments passed to input_grad

        # Returns
            dict: A dictionary of all model inputs and the gradients. Gradients are stored in key 'grads'
        """

        batches = [batch for batch in tqdm(self.input_grad_generator(dataloader_kwargs, batch_size, filter_idx,
                                                                     avg_func, layer, final_layer,
                                                                     selected_fwd_node, pre_nonlinearity, **kwargs))]
        return numpy_collate_concat(batches)

    def input_grad_generator(self, dataloader_kwargs, batch_size=32, filter_idx=None, avg_func=None, layer=None,
                             final_layer=True, selected_fwd_node=None, pre_nonlinearity=False, **kwargs):
        """Get input gradients

        # Arguments
            dataloader_kwargs: Keyword arguments passed to the dataloader
            batch_size: Batch size used for the dataloader
            filter_idx: filter index of `layer` for which the gradient should be returned
            avg_func: String name of averaging function to be applied across filters in layer `layer`
            layer: layer from which backwards the gradient should be calculated
            final_layer: Use the final (classification) layer as `layer`
            selected_fwd_node: None - not supported by KerasModel at the moment
            pre_nonlinearity: Try to use the layer output prior to activation (will not always be possible in an
            automatic way)
            **kwargs: Further arguments passed to input_grad

        # Yields
        - `dict`: A dictionary of all model inputs and the gradients. Gradients are stored in key 'grads'
        """

        if not isinstance(self.model, kipoi.model.GradientMixin):
            raise Exception("Model does not implement GradientMixin, so `input_grad` is not available.")

        logger.info('Initialized data generator. Running batches...')

        validate_kwargs(self.dataloader_cls, dataloader_kwargs)
        it = self.dataloader_cls(**dataloader_kwargs).batch_iter(batch_size=batch_size, **kwargs)

        for i, batch in enumerate(it):
            if i == 0 and not self.dataloader_cls.get_output_schema().compatible_with_batch(batch):
                logger.warning("First batch of data is not compatible with the dataloader schema.")

            pred = self.model.input_grad(batch['inputs'], filter_idx, avg_func, layer, final_layer,
                                         selected_fwd_node, pre_nonlinearity, **kwargs)

            # store the predictions with the inputs, so that they can be analysed together afterwards.
            batch['grads'] = pred
            yield batch
