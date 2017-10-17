from __future__ import absolute_import
from __future__ import print_function

import os
import logging
import yaml
import kipoi  # for .config module
from .utils import load_module
import abc
import six

_logger = logging.getLogger('kipoi')


MODEL_FIELDS = ['inputs', 'targets']
DATA_TYPES = ['dna', 'bigwig', 'v-plot']


class Model(object):
    __metaclass__ = abc.ABCMeta

    def predict_on_batch(self, x):
        raise NotImplementedError

    # TODO - define the .model attribute?


def dir_model_info(model_dir):
    """Load model yaml file
    """
    # Parse the model.yaml
    with open(os.path.join(model_dir, 'model.yaml')) as ifh:
        description_yaml = yaml.load(ifh)
    return description_yaml


def dir_load_model(model_dir):
    """Load the model

    1. Parse the model.yml
    2. Load the Model
    3. Append yaml description to __doc__
    4. Return the Model
    """
    # TODO - handle different model specifications:
    # - [x] local directory
    #   - [ ] Also allow .yml path?
    # - [ ] local yaml file path
    # - [ ] <username>/<model>:<version>
    # - [ ] Remote directory URL (git repo)

    # Parse the model.yaml
    with open(os.path.join(model_dir, 'model.yaml')) as ifh:
        unparsed_yaml = ifh.read()
    description_yaml = yaml.load(unparsed_yaml)
    model_spec = description_yaml['model']
    validate_model_spec(model_spec)

    # load the model
    if model_spec["type"] == "custom":
        model = load_model_custom(py_path=os.path.join(model_dir, model_spec["args"]["file"]),
                                  object_name=model_spec["args"]["object"])

    elif model_spec["type"] == "keras":
        cobj = model_spec["args"].get("custom_objects", None)
        if cobj is not None:
            cobj = os.path.join(model_dir, cobj)

        model = KerasModel(weights_file=os.path.join(model_dir, model_spec["args"]["weights"]),
                           arch_file=os.path.join(model_dir, model_spec["args"]["arch"]),
                           custom_objects_file=cobj)

    elif model_spec["type"] == "sklearn":
        model = SklearnModel(pkl_flie=os.path.join(model_dir, model_spec["args"]["file"]))

    elif model_spec["type"] == "pytorch":
        raise NotImplementedError
    else:
        raise ValueError("Unsupported model type: {0}. " +
                         "Model type needs to be one of: ['custom', 'keras', 'sklearn', 'pytorch']".
                         format(model_spec["type"]))

    # Append yaml description to __doc__
    try:
        model.__doc__ = """Model instance

        # Methods
          predict_on_batch(x)

        # model.yaml

            {0}
        """.format((' ' * 8).join(unparsed_yaml.splitlines(True)))
    except AttributeError:
        _logger.warning("Unable to set the docstring")

    try:
        model.model_spec = model_spec
    except:
        _logger.warning("Unable to set model_spec")

    return model


def load_model(model, source="kipoi"):
    """Load the model

    source: source from which to pull the model
    """
    if source == "dir":
        return dir_load_model(model)
    else:
        return kipoi.config.get_source(source).load_extractor(model)


def model_info(model, source="kipoi"):
    """Get information about the model

    # Arguments
      model: model's relative path/name in the source. 2nd column in the `kipoi.list_models()` `pd.DataFrame`.
      source: Model source. 1st column in the `kipoi.list_models()` `pd.DataFrame`.
    """
    if source == "dir":
        return dir_model_info(model)
    else:
        return kipoi.config.get_source(source).get_model_info(model)


def validate_model_spec(model_spec):
    """Validate the model specification in the yaml file
    """
    # check model fields
    assert (all(field in model_spec for field in MODEL_FIELDS))

    # check input and target data types
    for data_name, data_spec in six.iteritems(model_spec['inputs']):
        if type in data_spec:
            assert data_spec['type'] in DATA_TYPES
    for data_name, data_spec in six.iteritems(model_spec['targets']):
        if type in data_spec:
            assert data_spec['type'] in DATA_TYPES


# ------ individual implementations ----
# each requires a special module to be installed (?)
# - TODO - where to specify those requirements?


def load_model_custom(py_path, object_name):
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
    return getattr(load_module(py_path), object_name)


class GradientMixin():
    def input_grad(self, x, layer, filter_ind):
        raise NotImplementedError


class KerasModel(Model, GradientMixin):
    """Loads the serialized Keras model

    # Arguments
        weights_file: File path to the hdf5 weights or the hdf5 Keras model
        arhc_file: Architecture json model. If None, `weights_file` is
    assumed to speficy the whole model
        custom_objects_file: Python file defining the custom Keras objects
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

    def __init__(self, weights_file, arch_file, custom_objects_file):
        # TODO - check that Keras is indeed installed + specific requirements?

        from keras.models import model_from_json, load_model

        if custom_objects_file is not None and os.path.exists(custom_objects_file):
            self.custom_objects = load_module(custom_objects_file).OBJECTS
        else:
            self.custom_objects = {}

        self.weights_file = weights_file
        self.arch_file = arch_file

        self.gradient_functions = {}

        if arch_file is None:
            # load the whole model
            self.model = load_model(weights_file, custom_objects=self.custom_objects)
            _logger.info('successfully loaded the model from {}'.
                         format(weights_file))
        else:
            # load arch
            with open(arch_file, "r") as arch:
                self.model = model_from_json(arch.read(),
                                             custom_objects=self.custom_objects)
            _logger.info('successfully loaded model architecture from {}'.
                         format(arch_file))

            # load weights
            self.model.load_weights(weights_file)
            _logger.info('successfully loaded model weights from {}'.
                         format(weights_file))

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)

    def __generate_direct_saliency_functions__(self, layer, filter_slices = None,
                                               filter_func = None, filter_func_kwargs = None):
        import copy
        from keras import backend as kB
        # Generate the gradient functions according to the layer / filter definition
        if layer not in self.gradient_functions:
            self.gradient_functions[layer]= {}
        filter_id = str(filter_slices)
        if filter_func is not None:
            filter_id = str(filter_func) + ":" + str(filter_func_kwargs)
        if filter_id not in self.gradient_functions[layer]:
            # Copy input so that the model definition is not altered
            inp = copy.copy(self.model.inputs)
            # Get selected layer outputs
            filters = self.model.layers[layer].output
            if filter_slices is not None:
                sel_filter = filters[filter_slices]
            elif filter_func is not None:
                if filter_func_kwargs is None:
                    filter_func_kwargs = {}
                sel_filter = filter_func(filters, **filter_func_kwargs)
            else:
                raise Exception("Either filter_slices or filter_func have to be set!")
            # TODO: does Theano really require "sel_filter.sum()" instead of "sel_filter" here?
            saliency = kB.gradients(sel_filter, inp)
            if self.model.uses_learning_phase and not isinstance(kB.learning_phase(), int):
                inp.append(kB.learning_phase())
            self.gradient_functions[layer][filter_id] = kB.function(inp, saliency)
        return self.gradient_functions[layer][filter_id]

    def input_grad(self, x, layer, filter_slices = None, filter_func = None, filter_func_kwargs = None):
        """Adapted from keras.engine.training.predict_on_batch. Returns gradients for a single batch of samples.
    
        # Arguments
            x: Input samples, as a Numpy array.
    
        # Returns
            Numpy array(s) of predictions.
        """
        from keras.engine.training import _standardize_input_data
        from keras import backend as K
        x = _standardize_input_data(x, self.model._feed_input_names,
                                    self.model._feed_input_shapes)
        if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins = x + [0.]
        else:
            ins = x
        gf = self.__generate_direct_saliency_functions__(layer, filter_slices, filter_func, filter_func_kwargs)
        outputs = gf(ins)
        if len(outputs) == 1:
            return outputs[0]
        return outputs


# def PyTorchModel(Model):
# TODO - implement and test
#     def __init__(self, path):
#         import torch
#         self.model = torch.load(path)

#     def predict_on_batch(self, x):
#         return self.model(x)


class SklearnModel(Model):
    """Loads the serialized scikit learn model

    # Arguments
        pkl_file: File path to the dumped sklearn file in the pickle format.

    # model.yml entry

        ```
        Model:
          type: sklearn
          args:
            file: asd.pkl
        ```
    """

    def __init__(self, pkl_file):
        self.pkl_file = pkl_file

        from sklearn.externals import joblib
        self.model = joblib.load(self.pkl_file)

    def predict_on_batch(self, x):
        return self.model.predict(x)
