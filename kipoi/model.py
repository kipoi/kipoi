from __future__ import absolute_import
from __future__ import print_function

from collections import OrderedDict, Mapping
import sys
import os
import yaml
import kipoi  # for .config module
from .utils import (load_module, cd, merge_dicts, read_pickle, override_default_kwargs,
                    load_obj, inherits_from, infer_parent_class, makedir_exist_ok)
import abc
import six
import numpy as np
import json
import yaml

from .specs import ModelDescription, RemoteFile, DataLoaderImport, download_default_args
from .pipeline import Pipeline
import logging
from distutils.version import LooseVersion

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
          deps: instance of kipoi.specs.Dependencies

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
    """Load the `model` from `source`, as well as the default dataloder to model.default_dataloder.

    # Arguments
      model (str): model name
      source (str):  source name
      with_dataloader (bool): if True, the default dataloader is
        loaded to `model.default_dataloader` and the pipeline at `model.pipeline` enabled.

    # Returns
    - Instance of class inheriting from `kipoi.models.BaseModel` (like `kipoi.models.KerasModel`)
       decorated with additional attributes.

    # Methods
    - **predict_on_batch(x)**: Make model predictions given a batch of data `x`

    # Appended attributes
    - **type** (`str`): model type (class name)
    - **args** (`dict`): model args used to instantiate the model class
    - **info** (`kipoi.specs.Info`): information about the author (etc)
    - **schema** (`kipoi.specs.ModelSchema`): information about the input/outputdata modalities
    - **dependencies** (`kipoi.specs.Dependencies`): class specifying the dependencies.
          (implements `install` method for running the installation)
    - **default_dataloader** (class inheriting from `kipoi.data.BaseDataLoader`): default
           dataloader. None if `with_dataloader=False` was used.
    - **name** (`str`): model name
    - **source** (`str`): model source
    - **source_dir** (`str`): local path to model source storage
    - **postprocessing** (`dict`): dictionary of loaded plugin specifications
    - **pipeline** (`kipoi.pipeline.Pipeline`): handle to a `Pipeline` object

    """
    # TODO - model can be a yaml file or a directory
    if isinstance(source, str):
        source_name = source
        source = kipoi.config.get_source(source)
    else:
        source_name = 'obj'
        source = source

    # pull the model
    source.pull_model(model)
    # get the model directory
    source_dir = source.get_model_dir(model)
    # get model description
    md = source.get_model_descr(model)

    # TODO - is there a way to prevent code duplication here?
    # TODO - possible to inherit from both classes and call the corresponding inits?
    # --------------------------------------------
    # TODO - load it into memory?

    # TODO - validate md.default_dataloader <-> model

    # Load the dataloader
    if with_dataloader:
        # load from python
        if isinstance(md.default_dataloader, DataLoaderImport):
            with cd(source_dir):
                default_dataloader = md.default_dataloader.get()
            default_dataloader.source_dir = source_dir
            # download util links if specified under default & override the default parameters
            override = download_default_args(default_dataloader.args, source.get_dataloader_download_dir(model))
            if override:
                # override default arguments specified under default
                override_default_kwargs(default_dataloader, override)
        else:
            # load from directory
            # attach the default dataloader already to the model
            if ":" in md.default_dataloader:
                dl_source, dl_path = md.default_dataloader.split(":")
            else:
                dl_source = source
                dl_path = md.default_dataloader

            # allow to use relative and absolute paths for referring to the dataloader
            default_dataloader_path = os.path.join("/" + model, dl_path)[1:]
            default_dataloader = kipoi.get_dataloader_factory(default_dataloader_path,
                                                              dl_source)
    else:
        default_dataloader = None

    model_download_dir = source.get_model_download_dir(model)
    # Read the Model - append methods, attributes to self
    with cd(source_dir):  # move to the model directory temporarily

        # explicitly handle downloading files for TensorFlowModel
        if md.type == 'tensorflow' or md.defined_as == 'kipoi.model.TensorFlowModel':
            output_dir = os.path.join(model_download_dir, "ckp")
            md.args['checkpoint_path'] = _parse_tensorflow_checkpoint_path(md.args['checkpoint_path'], output_dir)

        # download url links if specified under args
        for k in md.args:
            if isinstance(md.args[k], RemoteFile):
                output_dir = os.path.join(model_download_dir, k)
                logger.info("Downloading model arguments {} from {}".format(k, md.args[k].url))
                makedir_exist_ok(output_dir)

                if md.args[k].md5:
                    fname = md.args[k].md5
                else:
                    fname = "file"

                # download the parameters and override the model
                path = md.args[k].get_file(os.path.join(output_dir, fname))
                md.args[k] = path
        if md.type is not None:
            # old API
            if md.type == 'custom':
                Mod = load_model_custom(**md.args)
                assert issubclass(Mod, BaseModel)  # it should inherit from Model
                mod = Mod()
            elif md.type == "pytorch":
                mod = infer_pyt_class(md.args)(**md.args)
            elif md.type in AVAILABLE_MODELS:
                # TODO - this doesn't seem to work
                mod = AVAILABLE_MODELS[md.type](**md.args)
            else:
                raise ValueError("Unsupported model type: {0}. " +
                                 "Model type needs to be one of: {1}".
                                 format(md.type,
                                        ['custom'] + list(AVAILABLE_MODELS.keys())))
        else:
            # new API
            try:
                Mod = load_obj(md.defined_as)
            except ImportError:
                if md.defined_as.startswith("kipoi.model."):
                    # user tried importing some of the available models
                    logger.error("{} is not a valid kipoi model. Available models are: {}\n".format(
                        md.defined_as,
                        ", ".join(["kipoi.model." + str(AVAILABLE_MODELS[k].__name__) for k in AVAILABLE_MODELS])
                    ))
                raise ImportError("Unable to import {}".format(md.defined_as))
            if not inherits_from(Mod, BaseModel):
                raise ValueError("Model {} needs to inherit from kipoi.model.BaseModel".format(md.defined_as))
            mod = Mod(**md.args)
            for k, v in six.iteritems(AVAILABLE_MODELS):
                if isinstance(mod, v):
                    md.type = k
            if md.type is None:
                md.type = 'custom'

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


class GradientMixin(object):
    __metaclass__ = abc.ABCMeta
    allowed_functions = ["sum", "max", "min", "absmax"]

    @abc.abstractmethod
    def input_grad(self, x, filter_idx=None, avg_func=None, layer=None, final_layer=True,
                   selected_fwd_node=None, pre_nonlinearity=False):
        """
        Calculate the input-layer gradient for filter `filter_idx` in layer `layer` with respect to `x`. If avg_func
        is defined average over filters with the averaging function `avg_func`. If `filter_idx` and `avg_func` are both
        not None then `filter_idx` is first applied and then `avg_func` across the selected filters.

        # Arguments
            x: model input
            filter_idx: filter index of `layer` for which the gradient should be returned
            avg_func: String name of averaging function to be applied across filters in layer `layer`
            layer: layer from which backwards the gradient should be calculated
            final_layer: Use the final (classification) layer as `layer`
            selected_fwd_node: None - not supported by KerasModel at the moment
            pre_nonlinearity: Try to use the layer output prior to activation (will not always be possible in an
            automatic way)
        """
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
                # As of the latest version of Theano this feature is not supported - the activation layer is too
                # diffuse to be handeled here since Theano does not have objects for the activation.
                raise Exception("`get_pre_activation_output` is not supported for Theano models!")
                import theano
                for inp_here in output.owner.inputs:
                    if not isinstance(inp_here, theano.gof.Constant):
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

    @staticmethod
    def homogenize_filter_slices(filter_slices):
        if isinstance(filter_slices, int):
            filter_slices = (filter_slices,)
        if isinstance(filter_slices, slice):
            filter_slices = (filter_slices,)
        if isinstance(filter_slices, list):
            filter_slices = tuple(filter_slices)
        if isinstance(filter_slices, tuple):
            # Add a 0th dimension for samples if obviously missing, but no information about the actual dimensions
            # is known!
            if len(filter_slices) == 1:
                filter_slices = tuple([slice(None)] + list(filter_slices))
        return filter_slices

    def _get_gradient_function(self, layer=None, use_final_layer=False, pre_nonlinearity=False, filter_slices=None,
                               filter_func=None, filter_func_kwargs=None):
        """
        Get keras gradient function

        # Arguments:
            layer: Layer name or index with respect to which the input gradients should be returned
            use_final_layer: Alternative to `layer`, if the final layer should be used. In this case `layer` can be None.
            filter_slices: Selection of filters in `layer` that should be taken into consideration
            filter_func: Function to be applied on all filters of the selected layer. If both `filter_slices` and
                `filter_func` are defined, then `filter_slices` will be applied first and then `filter_func`.
            filter_func_kwargs: keyword argument dict passed on to `filter_func`
        """
        import keras
        import copy
        from keras.models import Model
        # Generate the gradient functions according to the layer / filter definition
        gradient_function = None

        layer_label = layer
        # Try to use a previously generated gradient function
        if use_final_layer:
            layer_label = "_KIPOI_FINAL_"

        if layer_label is None:
            raise Exception("Either `layer` must be defined or `use_final_layer` set to True.")

        # Cannot query the layer output shape, so only if the slice is an integer or a list of length 1 it is
        # clear that the batch dimension is missing
        if filter_slices is not None:
            filter_slices = self.homogenize_filter_slices(filter_slices)

        if layer_label not in self.gradient_functions:
            self.gradient_functions[layer_label] = {}
        filter_id = str(filter_slices) + "_PNL_" + str(pre_nonlinearity)
        if filter_func is not None:
            filter_id = str(filter_func) + ":" + str(filter_func_kwargs) + ":" + filter_id
        if filter_id in self.gradient_functions[layer_label]:
            gradient_function = self.gradient_functions[layer_label][filter_id]

        if gradient_function is None:
            # model layer outputs wrt which the gradient shall be calculated
            selected_layers, sel_outputs, sel_output_dims = self.get_layers_and_outputs(layer=layer,
                                                                                        use_final_layer=use_final_layer,
                                                                                        pre_nonlinearity=pre_nonlinearity)

            # copy the model input in case learning flag has to appended when using the gradient function.
            inp = copy.copy(self.model.inputs)

            # Now check if layer outputs have to be concatenated (multiple input nodes in the respective layer)
            has_concat_output = False
            if len(sel_outputs) > 1:
                has_concat_output = True
                # Flatten layers in case dimensions don't match
                all_filters_flat = [keras.layers.Flatten()(x) if dim > 2 else x for x, dim in
                                    zip(sel_outputs, sel_output_dims)]
                # A new model has to be generated in order for the concatenated layer output to have a defined layer output
                if hasattr(keras.layers, "Concatenate"):
                    # Keras 2
                    all_filters_merged = keras.layers.Concatenate(axis=-1)(all_filters_flat)
                    gradient_model = Model(inputs=inp, outputs=all_filters_merged)
                else:
                    # Keras 1
                    all_filters_merged = keras.layers.merge(all_filters_flat, mode='concat')
                    gradient_model = Model(input=inp, output=all_filters_merged)
                # TODO: find a different way to get layer outputs...
                # gradient_model.compile(optimizer=self.model.optimizer, loss=self.model.loss)
                gradient_model.compile(optimizer='rmsprop', loss='binary_crossentropy')
                # output of interest for a given gradient
                output_oi = gradient_model.output
            else:
                output_oi = sel_outputs[0]

            # Which subset of the selected layer outputs should be looked at?
            if filter_slices is not None:
                if has_concat_output:
                    logger.warn("Filter slices have been defined for output selection from layers %s, but "
                                "layer outputs of nodes had to be concatenated. This will potentially lead to undesired "
                                "output - please take this concatenation into consideration when "
                                "defining `filter_slices`." % str([l.name for l in selected_layers]))
                output_oi = output_oi[filter_slices]

            # Should a filter function be applied
            if filter_func is not None:
                if filter_func_kwargs is None:
                    filter_func_kwargs = {}
                output_oi = filter_func(output_oi, **filter_func_kwargs)

            if (filter_slices is None) and (filter_func is None):
                raise Exception("Either filter_slices or filter_func have to be set!")

            # generate the actual gradient function
            from keras import backend as K
            saliency = K.gradients(output_oi, inp)

            if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inp.append(K.learning_phase())

            gradient_function = K.function(inp, saliency)

            # store the generated gradient function:
            self.gradient_functions[layer_label][filter_id] = gradient_function

        return gradient_function

    def _get_feed_input_names(self):
        import keras
        from keras import backend as K
        feed_input_names = None
        if keras.__version__[0] == '1':
            feed_input_names = self.model.input_names
        elif hasattr(keras.engine.training, "_standardize_input_data"):
            from keras.engine.training import _standardize_input_data
            if not hasattr(self.model, "_feed_input_names"):
                if not self.model.built:
                    self.model.build()
            feed_input_names = self.model._feed_input_names
        return feed_input_names

    def _batch_to_list(self, x):
        import keras
        from keras import backend as K
        feed_input_names = self._get_feed_input_names()
        if keras.__version__[0] == '1':
            from keras.engine.training import standardize_input_data as _standardize_input_data
            if not self.model.built:
                self.model.build()
            iis = None
            if hasattr(self.model, "internal_input_shapes"):
                iis = self.model.internal_input_shapes
            elif hasattr(self.model, "model") and hasattr(self.model.model, "internal_input_shapes"):
                iis = self.model.model.internal_input_shapes
            x_standardized = _standardize_input_data(x, feed_input_names,
                                                     iis)
        elif hasattr(keras.engine.training, "_standardize_input_data"):
            from keras.engine.training import _standardize_input_data
            if not hasattr(self.model, "_feed_input_names"):
                if not self.model.built:
                    self.model.build()
            fis = None
            if hasattr(self.model, "_feed_input_shapes"):
                fis = self.model._feed_input_shapes
            x_standardized = _standardize_input_data(x, feed_input_names, fis)
        else:
            raise Exception("This Keras version is not supported!")
        return x_standardized

    def _match_to_input(self, to_match, input):
        feed_input_names = self._get_feed_input_names()
        if isinstance(input, np.ndarray):
            assert len(to_match) == 1
            outputs = to_match[0]
        elif isinstance(input, list):
            # Already in right format
            outputs = to_match
        elif isinstance(input, dict):
            from collections import OrderedDict
            outputs_dict = OrderedDict()
            for k, v in zip(feed_input_names, to_match):
                outputs_dict[k] = v
            outputs = outputs_dict
        return outputs

    def _input_grad(self, x, layer=None, use_final_layer=False, filter_slices=None,
                    filter_func=None, filter_func_kwargs=None, pre_nonlinearity=False):
        """Adapted from keras.engine.training.predict_on_batch. Returns gradients for a single batch of samples.

        # Arguments
            x: Input samples, as a Numpy array.

        # Returns
            Numpy array(s) of predictions.
        """
        import keras
        from keras import backend as K
        x_standardized = self._batch_to_list(x)
        if self.model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins = x_standardized + [0.]
        else:
            ins = x_standardized
        gf = self._get_gradient_function(layer, use_final_layer=use_final_layer, filter_slices=filter_slices,
                                         filter_func=filter_func, filter_func_kwargs=filter_func_kwargs,
                                         pre_nonlinearity=pre_nonlinearity)
        outputs = gf(ins)

        # re-format to how the input was:
        return self._match_to_input(outputs, x)

    def input_grad(self, x, filter_idx=None, avg_func=None, layer=None, final_layer=True,
                   selected_fwd_node=None, pre_nonlinearity=False):
        """
        Calculate the input-layer gradient for filter `filter_idx` in layer `layer` with respect to `x`. If avg_func
        is defined average over filters with the averaging function `avg_func`. If `filter_idx` and `avg_func` are both
        not None then `filter_idx` is first applied and then `avg_func` across the selected filters.

        # Arguments
            x: model input
            filter_idx: filter index of `layer` for which the gradient should be returned
            avg_func: String name of averaging function to be applied across filters in layer `layer`
            layer: layer from which backwards the gradient should be calculated
            final_layer: Use the final (classification) layer as `layer`
            selected_fwd_node: None - not supported by KerasModel at the moment
            pre_nonlinearity: Try to use the layer output prior to activation (will not always be possible in an
            automatic way)
        """
        import keras.backend as K
        _avg_funcs = {"sum": K.sum, "min": K.min, "max": K.max, "absmax": lambda x: K.max(K.abs(x))}
        if avg_func is not None:
            assert avg_func in _avg_funcs
            avg_func = _avg_funcs[avg_func]
        else:
            if K._BACKEND == "theano":
                avg_func = _avg_funcs["sum"]
        if selected_fwd_node is not None:
            raise Exception("'selected_fwd_node' is currently not supported for Keras models!")
        return self._input_grad(x, layer=layer, filter_slices=filter_idx, use_final_layer=final_layer,
                                filter_func=avg_func, pre_nonlinearity=pre_nonlinearity)

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


class PyTorchFwdHook(object):
    def __init__(self):
        self.forward_values = []

    def run_forward_hook(self, module, input, output):
        self.forward_values.append(output)


pt_type_conversions = {
    "FloatTensor": "float",
    "DoubleTensor": "double",
    "HalfTensor": "half",
    "ByteTensor": "byte",
    "CharTensor": "char",
    "ShortTensor": "short",
    "IntTensor": "int",
    "LongTensor": "long"}
pt_type_conversions = {pre + k: v for pre in ["torch.", "torch.cuda."] for k, v in pt_type_conversions.items()}

def infer_pyt_class(kwargs):
    minimum_kwargs_new = (("weights", "module_obj"), ("weights", "module_class"))
    given_kwargs = set(kwargs.keys())
    if any([set(kwargs_new).issubset(given_kwargs) for kwargs_new in minimum_kwargs_new]):
        return PyTorchModel
    else:
        return OldPyTorchModel

class PyTorchModel(BaseModel, GradientMixin, LayerActivationMixin):
    """Loads a pytorch model.

    """

    MODEL_PACKAGE = "pytorch"

    def __init__(self, weights, module_class=None, module_kwargs=None, module_obj=None, module_file=None,
                 auto_use_cuda=True):
        """
        Instantiate a PyTorchModel. The preferred way of instantiating PyTorch models is by using the `load_state_dict`
        method of the model class that specifies the PyTorch model.

        This is in agreement with:
         https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        Arguments:
          weights: file in which the weights are stored
          module_class: name of the PyTorch module class (model class) defined in the `module_file` file. Also the
           `my_module_file.MyModuleClass` is allowed where `my_module_file.py` resides in the same folder as the
           `model.yaml`.
          module_kwargs: If `module_class` is used then kwargs for the module initialisation can be defined here.
          module_obj: name of the PyTorch module object ("model") defined in the `module_file` file. Also
            the `my_module_file.MyModule` is allowed where `my_module_file.py` resides in the same folder as the
           `model.yaml`.
          module_file: path to the python file defining either `module_obj` or `module_class`
          auto_use_cuda: Automatically try to use CUDA if available
        """
        import torch
        from kipoi.utils import load_obj

        if (module_obj is None) and (module_class is None):
            raise Exception("Either 'module_obj' or 'module_class' have to be defined.")

        obj_name = module_class
        if module_obj is not None:
            obj_name = module_obj

        if module_file is not None:
            obj = getattr(load_module(module_file), obj_name)
        else:
            try:
                obj = load_obj(obj_name)
            except ValueError as e:
                raise ValueError("The module file either has to be defined explicitly in `module_file` or implicitly "
                                 "in the `module_class` or `module_obj` arguments. Loading the PyTorchModel failed "
                                 "with: %s" % e.message)

        self.model = obj
        if module_class is not None:
            kwargs = {}
            if module_kwargs is not None:
                if isinstance(module_kwargs, six.string_types):
                    kwargs = yaml.load(module_kwargs)
                else:
                    kwargs = module_kwargs
            self.model = obj(**kwargs)

        self.model.load_state_dict(torch.load(weights))

        if auto_use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.use_cuda = True
        else:
            self.use_cuda = False

        # Assuming that model should be used for predictions only
        self.model.eval()

        # Keep all gradient hooks in a list
        self.grad_hooks = []

    @staticmethod
    def correct_neg_stride(x):
        if any([el < 0 for el in x.strides]):
            # pytorch doesn't support negative strides at the moment, copying the numpy array will create a new array
            # with positive strides.
            return x.copy()
        return x

    def _torch_var(self, input, requires_grad=False):
        from torch.autograd import Variable
        if self.use_cuda:
            input = input.cuda()
        out = Variable(input, requires_grad=requires_grad)
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

        pred, _ = self.np_run_pred(x)
        return self.pred_to_np(pred)

    def numpy_to_torch(self, x, requires_grad=False):
        import torch
        if isinstance(x, np.ndarray):
            # convert to a pytorch tensor and then to a pytorch variable
            input = self._torch_var(torch.from_numpy(self.correct_neg_stride(x)), requires_grad)

        elif isinstance(x, dict):
            # convert all entries in the dict to pytorch variables
            input = {k: self._torch_var(torch.from_numpy(self.correct_neg_stride(x[k])), requires_grad) for k in x}

        elif isinstance(x, list):
            # convert all entries in the list to pytorch variables
            input = [self._torch_var(torch.from_numpy(self.correct_neg_stride(el)), requires_grad) for el in x]

        else:
            raise Exception("Input not supported!")

        return input

    def np_run_pred(self, x, requires_grad=False):
        """
        Input dictionaries will be translated into **kwargs of the `model.forward(...)` call
        Input lists will be translated into *args of the `model.forward(...)` call
        Input np.ndarray will be used as the only argument in a `model.forward(...)` call
        """
        input = self.numpy_to_torch(x, requires_grad=requires_grad)
        if isinstance(x, np.ndarray):
            # convert to a pytorch tensor and then to a pytorch variable
            pred = self.model(input)

        elif isinstance(x, dict):
            # convert all entries in the dict to pytorch variables
            pred = self.model(**input)

        elif isinstance(x, list):
            # convert all entries in the list to pytorch variables
            pred = self.model(*input)

        else:
            raise Exception("Input not supported!")

        return pred, input

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
        import torch
        sqb_restr = r"\[([A-Za-z0-9_]+)\]"
        if LooseVersion(torch.__version__) < LooseVersion('0.4.0'):
            scopeName = trace_node_obj.scopeName()
        else:
            scopeName = trace_node_obj.node().scopeName()
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
        # Versions of Pytorch prior to '0.4.0':
        if LooseVersion(torch.__version__) < LooseVersion('0.4.0'):
            trace_fn = torch.jit.trace
        else:
            trace_fn = torch.jit.get_trace_graph

        trace = None

        if isinstance(x, np.ndarray):
            trace, _ = trace_fn(self.model, args=(self.numpy_to_torch(x),))
        elif isinstance(x, dict):
            trace, _ = trace_fn(self.model, kwargs=self.numpy_to_torch(x))
        elif isinstance(x, list):
            trace, _ = trace_fn(self.model, args=tuple(self.numpy_to_torch(x)))

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

    @classmethod
    def _pt_type_match(self, in_data, like, match_cuda=False):
        from torch.autograd import Variable
        if isinstance(like, Variable):
            in_type = like.data.type()
        else:
            in_type = like.type()
        out_data = getattr(in_data, pt_type_conversions[in_type])()
        if match_cuda and like.is_cuda:
            out_data = out_data.cuda()
        return out_data

    @classmethod
    def get_grad_tens(self, forward_values, filter_slices, filter_func):
        import torch
        import six
        if (filter_slices is None) and (filter_func is None):
            raise Exception("Either filter slices or filter function have to be defined")
        if filter_func is not None:
            if not (isinstance(filter_func, six.string_types) and filter_func in self.allowed_functions):
                raise Exception("filter_func has to be a string within %s" % str(self.allowed_functions))

        # perform given operations in numpy, simpler implementation...
        if filter_slices is not None:
            pt_filt_slice = torch.from_numpy(get_filter_array(filter_slices, tuple(forward_values.size()))).byte()
        else:
            pt_filt_slice = torch.ByteTensor(*forward_values.size())
            pt_filt_slice[:] = 1
        if filter_func is not None:
            if filter_func == "sum":
                # don't do anything and keep the filter mask
                pass
            else:
                float_mask = self._pt_type_match(pt_filt_slice, forward_values)
                subset_dataset = forward_values.cpu().data * float_mask
                pt_filt_slice_new = torch.ByteTensor(*pt_filt_slice.size()).zero_()
                if filter_func == "max":
                    # reset so that the masked values are the minimum
                    subset_dataset = subset_dataset + (1 - float_mask) * subset_dataset.min()
                    # Find the maximum output value among the selected
                    for i in range(subset_dataset.shape[0]):
                        pt_filt_slice_new[i] = subset_dataset[i] == subset_dataset[i].max()
                elif filter_func == "min":
                    # reset so that the masked values are the maximum
                    subset_dataset = subset_dataset + (1 - float_mask) * subset_dataset.max()
                    # Find the minimum value among the selected
                    for i in range(subset_dataset.shape[0]):
                        pt_filt_slice_new[i] = subset_dataset[i] == subset_dataset[i].min()
                elif filter_func == "absmax":
                    # Find the absmax value among the selected
                    for i in range(subset_dataset.shape[0]):
                        pt_filt_slice_new[i] = subset_dataset[i] == subset_dataset[i].abs().max()
                pt_filt_slice = pt_filt_slice_new
        pt_filt_slice = self._pt_type_match(pt_filt_slice, forward_values, match_cuda=True)
        return pt_filt_slice

    def _register_fwd_hook(self, layer):
        """
        Install a forward hook on the given layer index
        Returns a PytorchFwdHook object that contains a
        """
        fwd_hook_obj = PyTorchFwdHook()
        removable_hook_obj = layer.register_forward_hook(fwd_hook_obj.run_forward_hook)
        return fwd_hook_obj, removable_hook_obj

    def _input_grad(self, x, layer=None, filter_slices=None, filter_func=None, selected_fwd_node=None):
        import torch
        # Register hooks for the layers
        fwd_hook_obj, removable_hook_obj = self._register_fwd_hook(layer)

        pred, x_in = self.np_run_pred(x, requires_grad=True)

        self.model.zero_grad()

        if selected_fwd_node is not None:
            if isinstance(selected_fwd_node, int):
                grad_concat = fwd_hook_obj.forward_values[selected_fwd_node]
            else:
                raise Exception("'selected_fwd_node' can either be None or an integer indicating the PyTorch model"
                                "forward-iteration of the sepcified layer / module.")
        else:
            # in Keras at the moment the tensors are flattened and then concatenated for every position.
            flat_fwds = [el.view(el.size(0), -1) for el in fwd_hook_obj.forward_values]
            grad_concat = torch.cat(flat_fwds, 1)

        replacement_grad = self.get_grad_tens(grad_concat, filter_slices=filter_slices, filter_func=filter_func)
        grad_concat.backward(gradient=replacement_grad)
        removable_hook_obj.remove()

        def extract_grad(variable_obj):
            vo = variable_obj
            if vo.grad is not None:
                return vo.grad.cpu().data.numpy()
            else:
                ret_arr = np.empty(vo.size())
                ret_arr[:] = np.nan
                return ret_arr

        if isinstance(x_in, torch.autograd.Variable) or isinstance(x_in, torch.Tensor):
            # make sure it is on the cpu, then extract the gradient data as numpy arrays
            grad_out = extract_grad(x_in)

        elif isinstance(x_in, dict):
            # extract gradient values for all dict entries
            from collections import OrderedDict
            grad_out = OrderedDict()
            for k in x:
                grad_out[k] = extract_grad(x_in[k])

        elif isinstance(x_in, list):
            # extract gradient values for all list entries
            grad_out = [extract_grad(el) for el in x_in]

        else:
            raise Exception("Gradient could not be extracted!")

        return grad_out

    def input_grad(self, x, filter_idx=None, avg_func=None, layer=None, final_layer=True,
                   selected_fwd_node=None, pre_nonlinearity=False):
        """
        Calculate the input-layer gradient for filter `filter_idx` in layer `layer` with respect to `x`. If avg_func
        is defined average over filters with the averaging function `avg_func`. If `filter_idx` and `avg_func` are both
        not None then `filter_idx` is first applied and then `avg_func` across the selected filters.

        # Arguments
            x: model input
            filter_idx: filter index of `layer` for which the gradient should be returned
            avg_func: String name of averaging function to be applied across filters in layer `layer`
            layer: layer from which backwards the gradient should be calculated
            final_layer: Use the final (classification) layer as `layer`
            selected_fwd_node: None or integer. If a layer is re-used models may support that the gradient is
            calculated only with respect to one of the incoming edges / nodes.
            pre_nonlinearity: Try to use the layer output prior to activation (will not always be possible in an
            automatic way)
        """

        if layer is not None:
            selected_layers = [self.get_layer(layer)]
        elif final_layer:
            selected_layers = self.get_last_layers(x)
        else:
            raise Exception("Either `layer` must be defined or `final_layer` set to True.")

        if pre_nonlinearity:
            raise Exception("pre_nonlinearity is not implemented for PyTorch models.")

        if len(selected_layers) > 1:
            # This could be implemented by sequentially looping over layers
            raise Exception("Only one layer may be selected at a time!")

        if selected_layers[0] is None:
            raise ValueError("Unable to get layer {}".format(layer))

        return self._input_grad(x, layer=selected_layers[0], filter_slices=filter_idx, filter_func=avg_func,
                                selected_fwd_node=selected_fwd_node)

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

class OldPyTorchModel(PyTorchModel):
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
        logger.warn("You are using the old initialisation of Kipoi's pytorch models! This feature will soon be "
                    "removed. Please convert your model to comply with the new definition of loading 'PyTorchModel's.")
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

        # Keep all gradient hooks in a list
        self.grad_hooks = []


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
            predict_method: Which prediction method to use. Available options:
               'predict', 'predict_proba' or 'predict_log_proba'.
        ```
    """

    MODEL_PACKAGE = "scikit-learn"

    def __init__(self, pkl_file, predict_method="predict"):
        self.pkl_file = pkl_file

        from sklearn.externals import joblib
        self.model = joblib.load(self.pkl_file)
        assert predict_method in ['predict_proba', 'predict', 'predict_log_proba']
        assert hasattr(self.model, predict_method)
        self.predict_method = predict_method

    def predict_on_batch(self, x):
        # assert isinstance(x, dict)
        # assert len(x) == 1
        # x = x.popitem()[1]
        return getattr(self.model, self.predict_method)(x)

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


def get_filter_array(filter_slices, input_shape):
    def index_is_none(index_el):
        if isinstance(index_el, slice):
            if all([getattr(index_el, att) is None for att in ["start", "stop", "step"]]):
                return True
        return False

    input_dim = len(input_shape)
    pt_filt_slice = np.zeros(input_shape)
    if isinstance(filter_slices, int):
        if input_dim != 2:
            raise Exception(
                "Integer filter slice can only be used on 1D filter, but dimension is: %d" % (input_dim - 1))
        pt_filt_slice[:, filter_slices] = 1
    else:
        if isinstance(filter_slices, slice):
            filter_slices = (filter_slices,)
        if isinstance(filter_slices, list):
            filter_slices = tuple(filter_slices)
        if isinstance(filter_slices, tuple):
            # Add a 0th dimension for samples if missing.
            if len(filter_slices) == input_dim - 1:
                filter_slices = tuple([slice(None)] + list(filter_slices))
            # If dimension is wrong complain
            elif len(filter_slices) != input_dim:
                raise Exception("Filter slice of length %d cannot be applied in a filter of dimension: %d" % (
                    len(filter_slices), input_dim - 1))
            # If sample dimension is not ":" complain
            if not index_is_none(filter_slices[0]):
                raise Exception(
                    "0th (sample) dimension always has to be None. Filter dimension without sample dimension: %d." % (
                        input_dim - 1))
            # Finally apply filter
            pt_filt_slice.__setitem__(filter_slices, 1)
        else:
            raise Exception(
                "filter_slices has to be None or of type integer, tuple or list. Tuples and lists have to contain compatible objects e.g.: slice()-objects.")
    return pt_filt_slice


class TensorFlowModel(BaseModel, GradientMixin, LayerActivationMixin):
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

    @classmethod
    def get_grad_tens(self, forward_values, filter_slices, filter_func):
        import six
        if (filter_slices is None) and (filter_func is None):
            raise Exception("Either filter slices or filter function have to be defined")
        if filter_func is not None:
            if not (isinstance(filter_func, six.string_types) and filter_func in self.allowed_functions):
                raise Exception("filter_func has to be a string within %s" % str(self.allowed_functions))
        # perform given operations in numpy, simpler implementation...
        if filter_slices is not None:
            pt_filt_slice = get_filter_array(filter_slices, forward_values.shape)
        else:
            pt_filt_slice = np.zeros_like(forward_values)
            pt_filt_slice[:] = 1
        if filter_func is not None:
            if filter_func == "sum":
                # don't do anything and keep the filter mask
                pass
            else:
                float_mask = pt_filt_slice
                subset_dataset = forward_values * float_mask
                pt_filt_slice_new = np.zeros_like(forward_values)
                if filter_func == "max":
                    # reset so that the masked values are the minimum
                    subset_dataset = subset_dataset + (1 - float_mask) * subset_dataset.min()
                    # Find the maximum output value among the selected
                    for i in range(subset_dataset.shape[0]):
                        pt_filt_slice_new[i, ...] = subset_dataset[i, ...] == subset_dataset[i, ...].max()
                elif filter_func == "min":
                    # reset so that the masked values are the maximum
                    subset_dataset = subset_dataset + (1 - float_mask) * subset_dataset.max()
                    # Find the minimum value among the selected
                    for i in range(subset_dataset.shape[0]):
                        pt_filt_slice_new[i, ...] = subset_dataset[i, ...] == subset_dataset[i, ...].min()
                elif filter_func == "absmax":
                    # Find the absmax value among the selected
                    for i in range(subset_dataset.shape[0]):
                        pt_filt_slice_new[i, ...] = subset_dataset[i, ...] == np.abs(subset_dataset[i, ...]).max()
                pt_filt_slice = pt_filt_slice_new * pt_filt_slice
        return pt_filt_slice

    def input_grad(self, x, filter_idx=None, avg_func=None, layer=None, final_layer=True,
                   selected_fwd_node=None, pre_nonlinearity=False):
        """
        Calculate the input-layer gradient for filter `filter_idx` in layer `layer` with respect to `x`. If avg_func
        is defined average over filters with the averaging function `avg_func`. If `filter_idx` and `avg_func` are both
        not None then `filter_idx` is first applied and then `avg_func` across the selected filters.

        # Arguments
            x: model input
            filter_idx: filter index of `layer` for which the gradient should be returned
            avg_func: String name of averaging function to be applied across filters in layer `layer`
            layer: layer from which backwards the gradient should be calculated
            final_layer: Use the final (classification) layer as `layer`
            selected_fwd_node: None - not supported by TensorFlowModel
            pre_nonlinearity: Try to use the layer output prior to activation (will not always be possible in an
            automatic way)
        """

        if selected_fwd_node is not None:
            raise Exception("TensorFlowModel does not support the use of selected_fwd_node!")

        if pre_nonlinearity:
            raise Exception("TensorFlowModel does not support the use of pre_nonlinearity as desired nodes can be "
                            "selected directly!")

        import tensorflow as tf

        feed_dict = self._build_feed_dict(x)

        # get the correct layer with respect to which the gradients should be calculated
        assert isinstance(layer, six.string_types)
        if not final_layer:
            new_target_ops = get_op_outputs(self.graph, layer)
        else:
            new_target_ops = self.target_ops
            if not isinstance(new_target_ops, tf.Tensor):
                raise Exception("Gradients can't be calculated with respect to mutliple layers in TensorFlowModel. "
                                "Please select a single target operation using 'layer'.")

        fwd_values = self.sess.run(new_target_ops,
                                   feed_dict=merge_dicts(feed_dict, self.const_feed_dict))

        # Make sure that input_ops is a list of tf.Tensor objects.
        if isinstance(self.input_ops, dict):
            input_ops = list(self.input_ops.values())
        elif isinstance(self.input_ops, list):
            input_ops = self.input_ops
        elif isinstance(self.input_ops, tf.Tensor):
            input_ops = [self.input_ops]
        else:
            raise ValueError

        # Run the gradient prediction
        # get_grad_tens avoids rebuilding the model which is necessary for TF models!
        grad_op = tf.gradients(new_target_ops, input_ops, name='gradient_%s' % str(layer),
                               grad_ys=self.get_grad_tens(fwd_values, filter_idx, avg_func))
        grad_pred = self.sess.run(grad_op, feed_dict=feed_dict)

        # Format the output so match the input
        if isinstance(self.input_nodes, dict):
            # dict
            grad_pred_formatted = {}
            for op, ret_val in zip(input_ops, grad_pred):
                for k, v in six.iteritems(self.input_ops):
                    if v == op:
                        grad_pred_formatted[k] = ret_val
        elif isinstance(self.input_nodes, list):
            # list
            grad_pred_formatted = []
            for op, ret_val in zip(input_ops, grad_pred):
                for v in self.input_ops:
                    if v == op:
                        grad_pred_formatted.append(ret_val)
        elif isinstance(self.input_nodes, str):
            # single array
            grad_pred_formatted = grad_pred[0]
        else:
            raise ValueError

        return grad_pred_formatted

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


def _parse_tensorflow_checkpoint_path(ckp_path, output_dir):
    """Parse and download tensorflow's checkpoint_path
    """
    if not isinstance(ckp_path, str):
        # need to handle the special case
        error_message = "checkpoint_path needs to be either a string " + \
                        "checkpoint_path needs to be either or a dictionary with " + \
                        "keys: meta, index, data"
        if not isinstance(ckp_path, Mapping):
            raise ValueError(error_message + "\n detected class {}".format(type(ckp_path)))
        if set(ckp_path.keys()) != {'meta', 'index', 'data'}:
            raise ValueError(error_message + "\n detected keys {}".format(set(ckp_path.keys())))

        # either all are string or all are remote paths
        types = {type(ckp_path[k]) for k in ckp_path}
        if len(types) != 1:
            raise ValueError("All types in checkpoint_path need to be the same. Found: {}".format(types))
        if not (isinstance(ckp_path['meta'], str) or isinstance(ckp_path['meta'], RemoteFile)):
            raise ValueError("Values of the checkpoint_path ckp_path need to be either "
                             "str or RemoteFile. Found: {}".format(ckp_path['meta']))
        if isinstance(ckp_path['meta'], RemoteFile):
            # download files
            makedir_exist_ok(output_dir)
            ckp_path['meta'] = ckp_path['meta'].get_file(os.path.join(output_dir, "model.meta"))
            ckp_path['index'] = ckp_path['index'].get_file(os.path.join(output_dir, "model.index"))
            ckp_path['data'] = ckp_path['data'].get_file(os.path.join(output_dir, "model.data-00000-of-00001"))

        if isinstance(ckp_path['meta'], str):
            assert ckp_path['meta'].endswith(".meta")
            assert ckp_path['index'].endswith(".index")
            assert ckp_path['data'].endswith(".data-00000-of-00001")
            # figure out the prefix
            ckp_path_prefix = ckp_path['meta'][:-5]
            assert ckp_path['data'].startswith(ckp_path_prefix)
            assert ckp_path['index'].startswith(ckp_path_prefix)

            return ckp_path_prefix
        else:
            raise ValueError("Values of the checkpoint_path ckp_path need to be either "
                             "str or RemoteFile. Found: {}".format(ckp_path['meta']))
    else:
        return ckp_path

# --------------------------------------------


AVAILABLE_MODELS = OrderedDict([("keras", KerasModel),
                                ("pytorch", PyTorchModel),
                                ("sklearn", SklearnModel),
                                ("tensorflow", TensorFlowModel)])
# "custom": load_model_custom}
