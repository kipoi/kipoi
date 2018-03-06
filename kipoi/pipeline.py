"""Whole model pipeline: extractor + model
"""
from __future__ import absolute_import
from __future__ import print_function

import os
from .utils import cd
import kipoi  # for .config module
from .data import numpy_collate_concat
# import h5py
# import six
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def install_model_requirements(model, source="kipoi", and_dataloaders=False):
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
    kipoi.get_source(source).get_model_descr(dataloader).dependencies.install()


class Pipeline(object):
    """Provides the predict_example, predict and predict_generator to the kipoi.Model
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
            preproc_kwargs: Keyword arguments passed to the pre-processor
            **kwargs: Further arguments passed to batch_iter

        :return: Predict the whole array
        """
        pred_list = [batch for batch in tqdm(self.predict_generator(dataloader_kwargs,
                                                                    batch_size, **kwargs))]
        return numpy_collate_concat(numpy_collate_concat(pred_list))

    def predict_generator(self, dataloader_kwargs, batch_size=32, **kwargs):
        """Prediction generator

        # Arguments
            preproc_kwargs: Keyword arguments passed to the pre-processor
            **kwargs: Further arguments passed to batch_iter

        # Yields
            model batch prediction
        """
        logger.info('Initialized data generator. Running batches...')

        it = self.dataloader_cls(**dataloader_kwargs).batch_iter(batch_size=batch_size, **kwargs)

        for i, batch in enumerate(it):
            if i == 0 and not self.dataloader_cls.output_schema.compatible_with_batch(batch):
                logger.warn("First batch of data is not compatible with the dataloader schema.")
            yield self.model.predict_on_batch(batch['inputs'])
