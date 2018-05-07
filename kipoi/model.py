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

    MODEL_PACKAGE = None

    @abc.abstractmethod
    def predict_on_batch(self, x):
        raise NotImplementedError

    @classmethod
    def _sufficient_deps(cls, deps):
        """Tests it the provided dependencies contain MODEL_PACKAGE

        Args:
          deps: instance of kipoi.components.Dependencies

        Returns:
          True if cls.MODEL_PACKAGE is listed in the depenencies and False otherwise
        """
        if cls.MODEL_PACKAGE is None:
            return True
        else:
            for d in deps.conda:
                if cls.MODEL_PACKAGE in d:
                    return True
            for d in deps.pip:
                if cls.MODEL_PACKAGE in d:
                    return True
            return False


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


class LayerActivationMixin():

    @abc.abstractmethod
    def predict_activation_on_batch(self, x, layer, pre_nonlinearity=False):
        """
        Get predictions based on layer output. 

        Arguments:
            x: model inputs from dataloader batch
            layer: layer identifier / name. can be integer or string.
            pre_nonlinearity: Assure that output is returned from before the non-linearity. This feature does
            not have to be implemented always (not possible). If not implemented and set to True either raise
            Error or at least warn! 
        """
        raise NotImplementedError

class KerasModel(BaseModel, GradientMixin, LayerActivationMixin):
    """Loads the serialized Keras model

    # Arguments
        weights: File path to the hdf5 weights or the hdf5 Keras model
        arch: Architecture json model. If None, `weights` is
    assumed to speficy the whole model
        custom_objects: Python file defining the custom Keras objects
    in a `OBJECTS` dictionary
        backend: Keras backend to use ('tensorflow', 'theano', ...)
        image_dim_ordering: 'tf' or 'th': Whether to use 'tf' ('channels_last')
            or 'th' ('cannels_first') dimension ordering.

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

    MODEL_PACKAGE = "keras"

    def __init__(self, weights, arch=None, custom_objects=None, backend=None, image_dim_ordering=None):
        self.backend = backend
        self.image_dim_ordering = image_dim_ordering
        if self.backend is not None and 'KERAS_BACKEND' not in os.environ:
            logger.info("Using Keras backend: {0}".format(self.backend))
            os.environ['KERAS_BACKEND'] = self.backend
        if self.image_dim_ordering is not None:
            import keras.backend as K
            logger.info("Using image_dim_ordering: {0}".format(self.image_dim_ordering))
            K.set_image_dim_ordering(self.image_dim_ordering)
        import keras
        from keras.models import model_from_json, load_model

        if self.backend is not None:
            if keras.backend.backend() != self.backend:
                logger.warn("Keras backend is {0} instead of {1}".
                            format(keras.backend.backend(), self.backend))

        if custom_objects is not None and os.path.exists(custom_objects):
            self.custom_objects = load_module(custom_objects).OBJECTS
        else:
            self.custom_objects = {}

        self.weights = weights
        self.arch = arch

        self.gradient_functions = {}  # contains dictionaries with string reps of filter functions / slices
        self.activation_functions = {}  # contains the activation functions
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

    def get_layers_and_outputs(self, layer=None, use_final_layer=False, pre_nonlinearity=False):
        """
        Get layers and outputs either by name / index or from the final layer(s).
        If the final layer should be used it has an activation function that is not Linear, then the input to the
        activation function is returned. This check is not performed when `use_final_layer` is False and `layer`
        is being used.

        Arguments:
            layer: layer index (int) or name (non-int)
            use_final_layer:  instead of using `layer` return the final model layer(s) + outputs
        """
        import keras
        sel_outputs = []
        sel_output_dims = []

        def output_sel(layer, output):
            """
            If pre_nonlinearity is true run get_pre_activation_output
            """
            if pre_nonlinearity:
                output = self.get_pre_activation_output(layer, output)[0]  # always has length 1
            return output

        # If the final layer should be used: (relevant for gradient)
        if use_final_layer:
            # Use outputs from the model output layer(s)
            # If the last layer should be selected automatically then:
            # get all outputs from the model output layers
            if isinstance(self.model, keras.models.Sequential):
                selected_layers = [self.model.layers[-1]]
            else:
                selected_layers = self.model.output_layers
            for l in selected_layers:
                for i in range(self.get_num_inbound_nodes(l)):
                    sel_output_dims.append(len(l.get_output_shape_at(i)))
                    sel_outputs.append(output_sel(l, l.get_output_at(i)))

        # If not the final layer then the get the layer by its name / index
        elif layer is not None:
            if isinstance(layer, int):
                selected_layer = self.model.get_layer(index=layer)
            elif isinstance(layer, six.string_types):
                selected_layer = self.model.get_layer(name=layer)
            selected_layers = [selected_layer]
            # get the outputs from all nodes of the selected layer (selecting output from individual output nodes
            # creates None entries when running K.gradients())
            if self.get_num_inbound_nodes(selected_layer) > 1:
                logger.warn("Layer %s has multiple input nodes. By default outputs from all nodes "
                            "are concatenated" % selected_layer.name)
                for i in range(self.get_num_inbound_nodes(selected_layer)):
                    sel_output_dims.append(len(selected_layer.get_output_shape_at(i)))
                    sel_outputs.append(output_sel(selected_layer, selected_layer.get_output_at(i)))
            else:
                sel_output_dims.append(len(selected_layer.output_shape))
                sel_outputs.append(output_sel(selected_layer, selected_layer.output))
        else:
            raise Exception("Either use_final_layer has to be set or a layer name has to be defined.")

        return selected_layers, sel_outputs, sel_output_dims

    @staticmethod
    def get_pre_activation_output(layer, output):
        import keras
        # if the current layer uses an activation function then grab the input to the activation function rather
        # than the output from the activation function.
        # This can lead to confusion if the activation function translates to backend operations that are not a
        # single operation. (Which would also be a misuse of the activation function itself.)
        # suggested here: https://stackoverflow.com/questions/45492318/keras-retrieve-value-of-node-before-activation-function
        if hasattr(layer, "activation") and not layer.activation == keras.activations.linear:
            new_output_ois = []
            if hasattr(output, "op"):
                # TF
                for inp_here in output.op.inputs:
                    new_output_ois.append(inp_here)
            else:
                # TH
                for inp_here in output.owner.inputs:
                    new_output_ois.append(inp_here)
            if len(new_output_ois) > 1:
                raise Exception("More than one input to activation function of selected layer. No general rule "
                                "implemented for handing those cases. Consider using a linear activation function + a "
                                "non-linear activation layer instead.")
            return new_output_ois
        else:
            return [output]

    @staticmethod
    def get_num_inbound_nodes(layer):
        if hasattr(layer, "_inbound_nodes"):
            # Keras 2.1.5
            return len(layer._inbound_nodes)
        elif hasattr(layer, "inbound_nodes"):
            # Keras 2.0.4
            return len(layer.inbound_nodes)
        else:
            raise Exception("No way to find out about number of inbound Nodes")

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
        import keras
        from keras import backend as K
        if keras.__version__[0] == '1':
            from keras.engine.training import standardize_input_data as _standardize_input_data
            x = _standardize_input_data(x, self.model.input_names,
                                        self.model.internal_input_shapes)
        else:
            from keras.engine.training import _standardize_input_data
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

    def _generate_activation_output_functions(self, layer, pre_nonlinearity):
        import copy
        layer_id = str(layer) + "_" + str(pre_nonlinearity)
        if layer_id in self.activation_functions:
            return self.activation_functions[layer_id]

        # get the selected layers
        selected_layers, sel_outputs, sel_output_dims = self.get_layers_and_outputs(layer=layer,
                                                                                    use_final_layer=False,
                                                                                    pre_nonlinearity=pre_nonlinearity)

        # copy the model input in case learning flag has to appended when using the activation function.
        inp = copy.copy(self.model.inputs)

        # Can't we have multiple outputs for the function?
        output_oi = sel_outputs  # list of outputs should work: https://keras.io/backend/#backend-functions -> backend.function

        from keras import backend as K

        if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inp.append(K.learning_phase())

        activation_function = K.function(inp, output_oi)

        # store the generated activation function:
        self.activation_functions[layer_id] = activation_function

        return activation_function

    def predict_activation_on_batch(self, x, layer, pre_nonlinearity=False):
        """Adapted from keras.engine.training.predict_on_batch. Returns gradients for a single batch of samples.

        Arguments
            x: Input samples, as a Numpy array.

        Returns
            Numpy array(s) of predictions.
        """
        import keras
        from keras import backend as K
        if keras.__version__[0] == '1':
            from keras.engine.training import standardize_input_data as _standardize_input_data
            x = _standardize_input_data(x, self.model.input_names,
                                        self.model.internal_input_shapes)
        else:
            from keras.engine.training import _standardize_input_data
            x = _standardize_input_data(x, self.model._feed_input_names,
                                        self.model._feed_input_shapes)
        if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins = x + [0.]
        else:
            ins = x
        af = self._generate_activation_output_functions(layer, pre_nonlinearity)
        outputs = af(ins)
        return outputs


class PytorchFwdHook(object):

    def __init__(self):
        self.forward_values = []

    def run_forward_hook(self, module, input, output):
        self.forward_values.append(output)


class PyTorchModel(BaseModel, LayerActivationMixin):
    """Loads a pytorch model. 

    """

    MODEL_PACKAGE = "pytorch"

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

        return self.pred_to_np(pred)

    def pred_to_np(self, pred):
        from torch.autograd import Variable
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

    def get_layer(self, index):
        """
        Get layer (module) based on index: index for sequentials is e.g.: '1.5.1', for models defined as sublcasses of
        nn.Module it's the class object variable names
        """
        # index for sequentials is e.g.: '1.5.1', for models defined as sublcasses of nn.Module it's the class object
        # variable names
        for idx, m in self.model.named_modules():
            if idx == index:
                return m

    def get_layer_id(self, layer):
        for idx, m in self.model.named_modules():
            if m == layer:
                return idx

    @staticmethod
    def extract_module_id(trace_node_obj):
        import re
        sqb_restr = r"\[([A-Za-z0-9_]+)\]"
        scopeName = trace_node_obj.scopeName()
        idx_name = ".".join([re.search(sqb_restr, grp).group(1) for grp in
                             scopeName.split("/") if re.search(sqb_restr, grp) is not None])
        return idx_name

    @staticmethod
    def _is_nonlinear_activation(layer):
        import torch
        import torch.nn.modules.activation as tact
        import inspect
        activation_modules = []
        activation_module_names = []
        for mod_name, mod in tact.__dict__.items():
            if inspect.isclass(mod) and issubclass(mod, torch.nn.Module):
                if mod != torch.nn.Module:
                    activation_modules.append(mod)
                    activation_module_names.append(mod_name)
        # This will also catch instances of subclasses
        return any([isinstance(layer, mod) for mod in activation_modules])

    def _get_trace(self, x):
        import torch
        trace, _ = torch.jit.trace(self.model, args=x)
        return trace

    def get_last_layers(self, x):
        """
        Returns the model output layers
        x must be a pytorch Variable compatible with the model input
        """
        trace = self._get_trace(x)
        layer_idxs = [self.extract_module_id(n) for n in trace.graph().outputs()]
        return [self.get_layer(i) for i in layer_idxs]

    def get_downstream_layers(self, x, layer_id):
        # layers that are only created in the forward call (e.g. activation layers?) cannot be referred to properly
        raise Exception("No safe graph taversal is implemented yet!")
        layer_output_unames = []  # unique names of layer outputs (data streams)
        trace = self._get_trace(x)

        # Iterate over all modules in the graph and remember the module outputs (so that they can be checked later)
        for mod in trace.graph().nodes():
            if self.extract_module_id(mod) == layer_id:
                layer_output_unames += [n.uniqueName() for n in mod.outputs()]

        # get the model output stream names to check is it is a leaf output
        model_output_unames = [n.uniqueName() for n in trace.graph().outputs()]

        # get the layer ids that receive data from layer_id
        next_layer_ids = []
        for mod in trace.graph().nodes():
            this_layer_inputs = [n.uniqueName() for n in mod.inputs()]
            if any([iuname in layer_output_unames for iuname in this_layer_inputs]):
                next_layer_ids.append(self.extract_module_id(mod))

        if "" in next_layer_ids:
            raise Exception("The model is not compatible with the current implementation")

        # some values are fed back to the layer itself.
        next_layer_ids = [lid for lid in next_layer_ids if lid != layer_id]

        # layers receiving from the given layer, is the layer (also) an output leaf node
        return next_layer_ids, [self.get_layer(i) for i in next_layer_ids], \
            any([iuname in layer_output_unames for iuname in model_output_unames])

    def get_upstream_layers(self, x, layer_id):
        # layers that are only created in the forward call (e.g. activation layers?) cannot be referred to properly
        raise Exception("No safe graph taversal is implemented yet!")
        layer_input_unames = []  # unique names of layer outputs (data streams)
        trace = self._get_trace(x)

        # Iterate over all modules in the graph and remember the module inputs (so that they can be checked later)
        for mod in trace.graph().nodes():
            if self.extract_module_id(mod) == layer_id:
                layer_input_unames += [n.uniqueName() for n in mod.inputs()]

        # get the model input stream names to check is it is a leaf input
        model_input_unames = [n.uniqueName() for n in trace.graph().inputs()]

        # get the layer ids that feed data into layer_id
        prev_layer_ids = []
        for mod in trace.graph().nodes():
            this_layer_outputs = [n.uniqueName() for n in mod.outputs()]
            if any([ouname in layer_input_unames for ouname in this_layer_outputs]):
                prev_layer_ids.append(self.extract_module_id(mod))

        # some values are fed back to the layer itself.
        prev_layer_ids = [lid for lid in prev_layer_ids if lid != layer_id]

        # layers feeding into the given layer, is the layer (also) an output leaf node
        return prev_layer_ids, [self.get_layer(i) for i in prev_layer_ids], \
            any([iuname in layer_input_unames for iuname in model_input_unames])

    def _register_fwd_hook(self, layer):
        """
        Install a forward hook on the given layer index
        Returns a PytorchFwdHook object that contains a 
        """
        fwd_hook_obj = PytorchFwdHook()
        removable_hook_obj = layer.register_forward_hook(fwd_hook_obj.run_forward_hook)
        return fwd_hook_obj, removable_hook_obj

    def predict_activation_on_batch(self, x, layer, pre_nonlinearity=False):
        """Adapted from keras.engine.training.predict_on_batch. Returns gradients for a single batch of samples.

        Arguments
            x: Input samples, as a Numpy array.
        Returns
            List of list of Numpy array(s) of predictions. First level is the hooks (= layers), second level is the
            calls of the hooks (if the layer or its upstream layer is called multiple times in a graph).
        """
        selected_layers = [self.get_layer(layer)]

        if selected_layers[0] is None:
            raise ValueError("Unable to get layer {}".format(layer))

        if pre_nonlinearity:
            raise Exception("pre_nonlinearity is not implemented for PyTorch models.")
            # Currently inactive code because graph traversal is not yet failsafe for all imaginable model architectures
            # if not self._is_nonlinear_activation(selected_layers[0]):
            #    selected_layer_ids, selected_layers, is_leaf = self.get_upstream_layers(x, selected_layers[0])
            #    if len(selected_layer_ids) ==0:
            #        if is_leaf:
            #            raise Exception("Layer '%s' is a nonlinear activation function and is an input leaf node - no "
            #                            "upstream layer could be found!")
            #        else:
            #            raise Exception("Layer '%s' is a nonlinear activation function and no upstream layer could be "
            #                            "found!")

        # Register hooks for the layers
        fwd_hook_objs = []
        removable_hook_objs = []
        for selected_layer in selected_layers:
            fwd_hook_obj, removable_hook_obj = self._register_fwd_hook(selected_layer)
            fwd_hook_objs.append(fwd_hook_obj)
            removable_hook_objs.append(removable_hook_obj)

        # Run full prediction to also
        self.predict_on_batch(x)
        # Remove hook to avoid future use
        [rho.remove() for rho in removable_hook_objs]
        # convert results back and return values.
        # First loop over the hook (= layers) then over the calls of the hooks (inner loop)
        return [[self.pred_to_np(fv) for fv in fho.forward_values] for fho in fwd_hook_objs]


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

    MODEL_PACKAGE = "scikit-learn"

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


class TensorFlowModel(BaseModel, LayerActivationMixin):

    MODEL_PACKAGE = "tensorflow"

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

    def _build_feed_dict(self, x):
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

        return feed_dict

    def predict_on_batch(self, x):
        feed_dict = self._build_feed_dict(x)
        return self.sess.run(self.target_ops,
                             feed_dict=merge_dicts(feed_dict, self.const_feed_dict))

    def predict_activation_on_batch(self, x, layer, pre_nonlinearity=False):
        """
        Get predictions based on layer output. 

        Arguments:
            x: model inputs from dataloader batch
            layer: layer identifier / name. can be integer or string.
            pre_nonlinearity: Not implemented.
        """
        feed_dict = self._build_feed_dict(x)
        new_target_ops = get_op_outputs(self.graph, layer)
        return self.sess.run(new_target_ops,
                             feed_dict=merge_dicts(feed_dict, self.const_feed_dict))


AVAILABLE_MODELS = {"keras": KerasModel,
                    "pytorch": PyTorchModel,
                    "sklearn": SklearnModel,
                    "tensorflow": TensorFlowModel}
# "custom": load_model_custom}
