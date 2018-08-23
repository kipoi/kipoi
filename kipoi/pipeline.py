"""Whole model pipeline: dataloader + model
"""
from __future__ import absolute_import
from __future__ import print_function

import os
from .utils import cd
import kipoi  # for .config module
from .data import numpy_collate_concat
# import h5py
import six
from tqdm import tqdm
import logging
import six

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
        logger.warn("Required arguments for the dataloader: {0} were not specified".
                    format(",".join(list(missing_arg))))
    unused = set(dataloader_kwargs.keys()) - set(dataloader.args.keys())
    if len(unused) > 0:
        logger.warn("Some provided dataloader kwargs were not used: {0}".format(unused))
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
        if not self.model.schema.compatible_with_schema(self.dataloader_cls.output_schema):
            logger.warn("dataloader.output_schema is not compatible with model.schema")
        else:
            logger.info("dataloader.output_schema is compatible with model.schema")

    def predict_example(self, batch_size=32, test_equal=False):
        """Run model prediction for the example file

        # Arguments
            batch_size: batch_size
            test_equal: currently not implemented
            **kwargs: Further arguments passed to batch_iter
        """
        logger.info('Initialized data generator. Running batches...')

        with cd(self.dataloader_cls.source_dir):
            dl = self.dataloader_cls.init_example()
            logger.info('Returned data schema correct')

            it = dl.batch_iter(batch_size=batch_size)

            # test that all predictions go through
            pred_list = []
            for i, batch in enumerate(tqdm(it)):
                if i == 0 and not self.dataloader_cls.output_schema.compatible_with_batch(batch):
                    logger.warn("First batch of data is not compatible with the dataloader schema.")
                pred_list.append(self.model.predict_on_batch(batch['inputs']))

        # TODO - check that the predicted values match the model targets

        #     if test_equal:
        #         match.append(compare_numpy_dict(y_pred, batch['targets'], exact=False))
        # if not all(match):
        #     logger.warning("For {0}/{1} batch samples: target != model(inputs)")
        # else:
        #     logger.info("All target values match model predictions")

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
            if i == 0 and not self.dataloader_cls.output_schema.compatible_with_batch(batch):
                logger.warn("First batch of data is not compatible with the dataloader schema.")
            if layer is None:
                yield self.model.predict_on_batch(batch['inputs'])
            else:
                yield self.model.predict_activation_on_batch(batch['inputs'], layer=layer)

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
            if i == 0 and not self.dataloader_cls.output_schema.compatible_with_batch(batch):
                logger.warn("First batch of data is not compatible with the dataloader schema.")

            pred = self.model.input_grad(batch['inputs'], filter_idx, avg_func, layer, final_layer,
                                         selected_fwd_node, pre_nonlinearity, **kwargs)

            # store the predictions with the inputs, so that they can be analysed together afterwards.
            batch['grads'] = pred
            yield batch
