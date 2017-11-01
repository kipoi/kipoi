from __future__ import absolute_import
from __future__ import print_function

import os
import logging
import yaml
import kipoi  # for .config module
from .utils import load_module, cd
import abc
import six

from .components import ModelDescription
from .pipeline import Pipeline

_logger = logging.getLogger('kipoi')


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict_on_batch(self, x):
        raise NotImplementedError

    # TODO - define the .model attribute?


def get_model(model, source="kipoi", with_dataloader=True):
    """Load the `model` from `source`, as well as the
    default dataloder to model.default_dataloder.

    Args:
      model, str:  model name
      source, str:  source name
      with_dataloader, bool: if True, the default dataloader is
        loaded to model.default_dataloadera and the pipeline at model.pipeline enabled.
    """
    # TODO - model can be a yaml file or a directory
    source_name = source

    source = kipoi.config.get_source(source)

    # pull the model & get the model directory
    yaml_path = source.pull_model(model)
    source_dir = os.path.dirname(yaml_path)

    # Setup model description
    md = ModelDescription.load(yaml_path)
    # TODO - is there a way to prevent code duplication here?
    # TODO - possible to inherit from both classes and call the corresponding inits?
    # --------------------------------------------
    # TODO - load it into memory?

    # TODO - validate md.default_dataloader <-> model

    # attach the default dataloader already to the model
    if ":" in md.default_dataloader:
        dl_source, dl_path = md.default_dataloader.split(":")
    else:
        dl_source = source_name
        dl_path = md.default_dataloader

    if with_dataloader:
        # allow to use relative and absolute paths for referring to the dataloader
        default_dataloader_path = os.path.join("/" + model, dl_path)[1:]
        default_dataloader = kipoi.get_dataloader_factory(default_dataloader_path,
                                                          dl_source)
    else:
        default_dataloader = None

    # Read the Model - append methods, attributes to self
    with cd(source_dir):  # move to the model directory temporarily
        if md.type == 'custom':
            Mod = load_model_custom(**md.args)
            assert issubclass(Mod, BaseModel)  # it should inherit from Model
            mod = Mod()
        elif md.type in AVAILABLE_MODELS:
            # TODO - this doesn't seem to work
            mod = AVAILABLE_MODELS[md.type](**md.args)
        else:
            raise ValueError("Unsupported model type: {0}. " +
                             "Model type needs to be one of: {1}".
                             format(md.type,
                                    ['custom'] + list(AVAILABLE_MODELS.keys())))

    # populate the returned class
    mod.type = md.type
    mod.args = md.args
    mod.info = md.info
    mod.schema = md.schema
    mod.dependencies = md.dependencies
    mod.default_dataloader = default_dataloader
    mod.name = model
    mod.source = source
    mod.source_name = source_name
    mod.source_dir = source_dir
    mod.post_processing = md.post_processing
    if with_dataloader:
        mod.pipeline = Pipeline(model=mod, dataloader_cls=default_dataloader)
    else:
        mod.pipeline = None
    return mod


# ------ individual implementations ----
# each requires a special module to be installed (?)
# - TODO - where to specify those requirements?
#      model: model's relative path / name in the source.
#      2nd column in the `kipoi.list_models()` `pd.DataFrame`.


def load_model_custom(file, object):
    """Loads the custom Model

    # model.yml entry

        ```
        Model:
          type: custom
          args:
            file: model.py
            object: Model
        ```
    """
    return getattr(load_module(file), object)


class KerasModel(BaseModel):
    """Loads the serialized Keras model

    # Arguments
        weights: File path to the hdf5 weights or the hdf5 Keras model
        arhc: Architecture json model. If None, `weights` is
    assumed to speficy the whole model
        custom_objects: Python file defining the custom Keras objects
    in a `OBJECTS` dictionary


    # `model.yml` entry

        ```
        Model:
          type: Keras
          args:
            weights: model.h5
            arch: model.json
            custom_objects: custom_keras_objects.py
        ```
    """

    def __init__(self, weights, arch, custom_objects=None):
        # TODO - check that Keras is indeed installed + specific requirements?

        from keras.models import model_from_json, load_model

        if custom_objects is not None and os.path.exists(custom_objects):
            self.custom_objects = load_module(custom_objects).OBJECTS
        else:
            self.custom_objects = {}

        self.weights = weights
        self.arch = arch

        if arch is None:
            # load the whole model
            self.model = load_model(weights, custom_objects=self.custom_objects)
            _logger.info('successfully loaded the model from {}'.
                         format(weights))
        else:
            # load arch
            with open(arch, "r") as arch:
                self.model = model_from_json(arch.read(),
                                             custom_objects=self.custom_objects)
            _logger.info('successfully loaded model architecture from {}'.
                         format(arch))

            # load weights
            self.model.load_weights(weights)
            _logger.info('successfully loaded model weights from {}'.
                         format(weights))

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)


# def PyTorchModel(BaseModel):
# TODO - implement and test
#     def __init__(self, path):
#         import torch
#         self.model = torch.load(path)

#     def predict_on_batch(self, x):
#         return self.model(x)


class SklearnModel(BaseModel):
    """Loads the serialized scikit learn model

    # Arguments
        pkl_file: File path to the dumped sklearn file in the pickle format.

    # model.yml entry

        ```
        Model:
          type: sklearn
          args:
            pkl_file: asd.pkl
        ```
    """

    def __init__(self, pkl_file):
        self.pkl_file = pkl_file

        from sklearn.externals import joblib
        self.model = joblib.load(self.pkl_file)

    def predict_on_batch(self, x):
        # assert isinstance(x, dict)
        # assert len(x) == 1
        # x = x.popitem()[1]
        return self.model.predict(x)


AVAILABLE_MODELS = {"keras": KerasModel,
                    # "pytorch": PyTorchModel,
                    "sklearn": SklearnModel}
# "custom": load_model_custom}
