from __future__ import absolute_import
from __future__ import print_function

import os
import yaml
import kipoi  # for .config module
from .utils import load_module, cd, merge_dicts, read_pickle
import abc
import six
import numpy as np
import json

from .components import ModelDescription
from .pipeline import Pipeline
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    with cd(source_dir):
        md = ModelDescription.load(os.path.basename(yaml_path))
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
    # parse the postprocessing module
    mod.postprocessing = md.postprocessing
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


class GradientMixin():

    def input_grad(self, x, layer, filter_ind):
        raise NotImplementedError


class KerasModel(BaseModel, GradientMixin):
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

    def __init__(self, weights, arch=None, custom_objects=None):
        # TODO - check that Keras is indeed installed + specific requirements?

        from keras.models import model_from_json, load_model

        if custom_objects is not None and os.path.exists(custom_objects):
            self.custom_objects = load_module(custom_objects).OBJECTS
        else:
            self.custom_objects = {}

        self.weights = weights
        self.arch = arch

        self.gradient_functions = {}
        if arch is None:
            # load the whole model
            self.model = load_model(weights, custom_objects=self.custom_objects)
            logger.info('successfully loaded the model from {}'.
                        format(weights))
        else:
            # load arch
            with open(arch, "r") as arch:
                self.model = model_from_json(arch.read(),
                                             custom_objects=self.custom_objects)
            logger.info('successfully loaded model architecture from {}'.
                        format(arch))

            # load weights
            self.model.load_weights(weights)
            logger.info('successfully loaded model weights from {}'.
                        format(weights))

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)

    def __generate_direct_saliency_functions__(self, layer, filter_slices=None,
                                               filter_func=None, filter_func_kwargs=None):
        import copy
        from keras import backend as K
        # Generate the gradient functions according to the layer / filter definition
        if layer not in self.gradient_functions:
            self.gradient_functions[layer] = {}
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
            saliency = K.gradients(sel_filter, inp)
            if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inp.append(K.learning_phase())
            self.gradient_functions[layer][filter_id] = K.function(inp, saliency)
        return self.gradient_functions[layer][filter_id]

    def _input_grad(self, x, layer, filter_slices=None, filter_func=None, filter_func_kwargs=None):
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

    def pred_grad(self, x, output_slice):
        return self._input_grad(x, -1, output_slice)


class PyTorchModel(BaseModel):
    """Loads a pytorch model. 

    """

    def __init__(self, file=None, build_fn=None, weights=None, auto_use_cuda=True):
        """
        Load model
        `weights`: Path to the where the weights are stored (may also contain model architecture, see below)
        `gen_fn`: Either callable or path to callable that returns a pytorch model object. If `weights` is not None
        then the model weights will be loaded from that file, otherwise it is assumed that the weights are already set
        after execution of `gen_fn()` or the function defined in `gen_fn`.  

        Models can be loaded in 2 ways:
        If the model was saved:

        * `torch.save(model, ...)` then the model will be loaded by calling `torch.load(weights)`
        * `torch.save(model.state_dict(), ...)` then another callable has to be passed to arch which returns the
        `model` object, on then `model.load_state_dict(torch.load(weights))` will then be called. 

        Where `weights` is the parameter of this function.
        Partly based on: https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
        """
        import torch
        if build_fn is not None:
            if callable(build_fn):
                gen_fn_callable = build_fn

            elif isinstance(build_fn, six.string_types):
                file_path = file
                obj_name = build_fn
                gen_fn_callable = getattr(load_module(file_path), obj_name)

            else:
                raise Exception("gen_fn has to be callable or a string pointing to the callable.")

            # Load model using generator function
            self.model = gen_fn_callable()

            # Load weights
            if weights is not None:
                self.model.load_state_dict(torch.load(weights))

        elif weights is not None:
            # Architecture is stored with the weights (not recommended)
            self.model = torch.load(weights)

        else:
            raise Exception("At least one of the arguments 'weights' or 'gen_fn' has to be set.")

        if auto_use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.use_cuda = True
        else:
            self.use_cuda = False

        # Assuming that model should be used for predictions only
        self.model.eval()

    @staticmethod
    def correct_neg_stride(x):
        if any([el < 0 for el in x.strides]):
            # pytorch doesn't support negative strides at the moment, copying the numpy array will create a new array
            # with positive strides.
            return x.copy()
        return x

    def _torch_var(self, input):
        from torch.autograd import Variable
        out = Variable(input)
        if self.use_cuda:
            out = out.cuda()
        return out

    def _torch_var_to_numpy(self, input):
        if input.is_cuda:
            input = input.cpu()
        return input.data.numpy()

    def _model_is_cuda(self):
        return next(self.model.parameters()).is_cuda

    def predict_on_batch(self, x):
        """
        Input dictionaries will be translated into **kwargs of the `model.forward(...)` call
        Input lists will be translated into *args of the `model.forward(...)` call
        Input np.ndarray will be used as the only argument in a `model.forward(...)` call
        """
        # TODO: Understand how pytorch models could return multiple outputs
        import torch
        from torch.autograd import Variable

        if isinstance(x, np.ndarray):
            # convert to a pytorch tensor and then to a pytorch variable
            input = self._torch_var(torch.from_numpy(self.correct_neg_stride(x)))
            pred = self.model(input)

        elif isinstance(x, dict):
            # convert all entries in the dict to pytorch variables
            input_dict = {k: self._torch_var(torch.from_numpy(self.correct_neg_stride(x[k]))) for k in x}
            pred = self.model(**input_dict)

        elif isinstance(x, list):
            # convert all entries in the list to pytorch variables
            input_list = [self._torch_var(torch.from_numpy(self.correct_neg_stride(el))) for el in x]
            pred = self.model(*input_list)

        else:
            raise Exception("Input not supported!")

        # convert results back to numpy arrays
        if isinstance(pred, Variable):
            pred_np = self._torch_var_to_numpy(pred)

        elif isinstance(pred, dict):
            pred_np = {k: self._torch_var_to_numpy(pred[k]) for k in pred}

        elif isinstance(pred, list) or isinstance(pred, tuple):
            pred_np = [self._torch_var_to_numpy(el) for el in pred]

        else:
            raise Exception("Model output format not supported!")

        return pred_np


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

# --------------------------------------------
# Tensorflow


def get_op_outputs(graph, node_names):
    """Query op names
    """
    if isinstance(node_names, dict):
        return {k: graph.get_operation_by_name(v).outputs[0]
                for k, v in six.iteritems(node_names)}
    elif isinstance(node_names, list):
        return [graph.get_operation_by_name(v).outputs[0]
                for v in node_names]
    elif isinstance(node_names, str):
        return graph.get_operation_by_name(node_names).outputs[0]
    else:
        raise ValueError("node_names has to be dict, list or str. Found: {0}".
                         format(type(node_names)))


class TensorFlowModel(BaseModel):

    def __init__(self,
                 input_nodes,
                 target_nodes,
                 checkpoint_path,
                 const_feed_dict_pkl=None
                 ):
        """Tensorflow graph

        Args:
          input_nodes: dict(str), list(str) or str: input node names.
            Keys correspond to the values in the feeded data (in schema)
          target_nodes: Same as input_nodes, but for the output node.
            If dict/list, the model will return a dict/list of np.arrays.
          checkpoint_path: Path to the saved model using:
            `saver = tf.train.Saver(); saver.save(checkpoint_path)`
          const_feed_dict_pkl: Constant feed dict stored as a pickle file.
            Values of this dict will get passed every time to feed_dict.
            Hence, const_feed_dict holds required values by the model not
            provided by the Dataloader.
        """
        import tensorflow as tf

        self.input_nodes = input_nodes
        self.target_nodes = target_nodes
        self.checkpoint_path = checkpoint_path
        self.graph = tf.Graph()  # use a fresh graph for the model
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(self.checkpoint_path + '.meta')
            saver.restore(self.sess, self.checkpoint_path)

        self.input_ops = get_op_outputs(self.graph, input_nodes)
        self.target_ops = get_op_outputs(self.graph, target_nodes)

        self.const_feed_dict_pkl = const_feed_dict_pkl
        if self.const_feed_dict_pkl is not None:
            # Load the feed dictionary from the pickle file
            const_feed_dict = read_pickle(self.const_feed_dict_pkl)
            self.const_feed_dict = {self.graph.get_operation_by_name(k).outputs[0]: v
                                    for k, v in six.iteritems(const_feed_dict)}
        else:
            self.const_feed_dict = {}

    def predict_on_batch(self, x):

        # build feed_dict
        if isinstance(self.input_nodes, dict):
            # dict
            assert isinstance(x, dict)
            feed_dict = {v: x[k] for k, v in six.iteritems(self.input_ops)}
        elif isinstance(self.input_nodes, list):
            # list
            assert isinstance(x, list)
            feed_dict = {v: x[i] for i, v in enumerate(self.input_ops)}
        elif isinstance(self.input_nodes, str):
            # single array
            feed_dict = {self.input_ops: x}
        else:
            raise ValueError

        return self.sess.run(self.target_ops,
                             feed_dict=merge_dicts(feed_dict, self.const_feed_dict))


AVAILABLE_MODELS = {"keras": KerasModel,
                    "pytorch": PyTorchModel,
                    "sklearn": SklearnModel,
                    "tensorflow": TensorFlowModel}
# "custom": load_model_custom}
