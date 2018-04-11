import warnings
warnings.filterwarnings('ignore')
import pytest
import sys
import os
import yaml
from contextlib import contextmanager
import kipoi
from kipoi.pipeline import install_model_requirements
from kipoi.utils import Slice_conv
import config

# TODO: Implement automatic switching of backends to test on Theano model!

INSTALL_REQ = config.install_req


EXAMPLES_TO_RUN = ["rbp", "extended_coda"]  # "pyt" not used as gradients are not yet supported for pytorch model.


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def read_json_yaml(filepath):
    with open(filepath) as ifh:
        return yaml.load(ifh)


def get_extractor_cfg(model_dir):
    return read_json_yaml(os.path.join(model_dir, 'extractor.yaml'))


def get_test_kwargs(model_dir):
    return read_json_yaml(os.path.join(model_dir, 'example_files/test.json'))

def get_sample_functional_model_input():
    import numpy as np
    return [np.random.rand(1,280,256), np.random.rand(1,280,256)]

def get_sample_functional_model():
    # Keras ["2.0.4", "2.1.5"]
    import keras
    if keras.__version__.startswith("2."):
        # from the Keras Docs - functional models
        import keras
        from keras.layers import Input, LSTM, Dense
        from keras.models import Model
        #
        tweet_a = Input(shape=(280, 256))
        tweet_b = Input(shape=(280, 256))
        #
        # This layer can take as input a matrix
        # and will return a vector of size 64
        shared_lstm = LSTM(64, name = "shared_lstm")
        #
        # When we reuse the same layer instance
        # multiple times, the weights of the layer
        # are also being reused
        # (it is effectively *the same* layer)
        encoded_a = shared_lstm(tweet_a)
        encoded_b = shared_lstm(tweet_b)
        # We can then concatenate the two vectors:
        merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1, name = "concatenation")
        # And add a logistic regression on top
        predictions = Dense(1, activation='sigmoid', name = "final_layer")(merged_vector)
        # We define a trainable model linking the
        # tweet inputs to the predictions
        model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)
        #
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    elif keras.__version__.startswith("1."):
        # from the Keras Docs - functional models
        import keras
        from keras.layers import Input, LSTM, Dense, merge
        from keras.models import Model
        #
        tweet_a = Input(shape=(280, 256))
        tweet_b = Input(shape=(280, 256))
        #
        # This layer can take as input a matrix
        # and will return a vector of size 64
        shared_lstm = LSTM(64, name = "shared_lstm")
        # When we reuse the same layer instance
        # multiple times, the weights of the layer
        # are also being reused
        # (it is effectively *the same* layer)
        encoded_a = shared_lstm(tweet_a)
        encoded_b = shared_lstm(tweet_b)
        # We can then concatenate the two vectors:
        # merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)
        merged_vector = merge([encoded_a, encoded_b], mode='concat', name = "concatenation")
        # And add a logistic regression on top
        predictions = Dense(1, activation='sigmoid', name = "final_layer")(merged_vector)
        # We define a trainable model linking the
        # tweet inputs to the predictions
        model = Model(input=[tweet_a, tweet_b], output=predictions)
        #
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    import tempfile
    temp_weights = tempfile.mkstemp()[1]
    temp_arch = tempfile.mkstemp()[1]
    with open(temp_arch, "w") as ofh:
        ofh.write(model.to_json())
    model.save_weights(temp_weights)
    return temp_weights, temp_arch

def get_sample_sequential_model_input():
    import numpy as np
    return [np.random.rand(1,784)]

def get_sample_sequential_model():
    # from the Keras Docs - sequential models
    from keras.layers import Dense
    from keras.models import Sequential
    model = Sequential([
        Dense(32, input_shape=(784,), activation='relu', name = "first"),
        Dense(32, activation='relu', name = "hidden"),
        Dense(10, activation = 'softmax', name = "final"),
    ])
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    import tempfile
    temp_weights = tempfile.mkstemp()[1]
    temp_arch = tempfile.mkstemp()[1]
    with open(temp_arch, "w") as ofh:
        ofh.write(model.to_json())
    model.save_weights(temp_weights)
    return temp_weights, temp_arch

@pytest.mark.parametrize("example", EXAMPLES_TO_RUN)
def test_activation_function_model(example):
    """Test extractor
    """
    if example == "rbp" and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")
    #
    example_dir = "examples/{0}".format(example)
    # install the dependencies
    # - TODO maybe put it implicitly in load_dataloader?
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)
    #
    Dl = kipoi.get_dataloader_factory(example_dir, source="dir")
    #
    test_kwargs = get_test_kwargs(example_dir)
    #
    # install the dependencies
    # - TODO maybe put it implicitly in load_extractor?
    if INSTALL_REQ:
        install_model_requirements(example_dir, source="dir")
    #
    # get model
    model = kipoi.get_model(example_dir, source="dir")
    #
    with cd(example_dir + "/example_files"):
        # initialize the dataloader
        dataloader = Dl(**test_kwargs)
        #
        # sample a batch of data
        it = dataloader.batch_iter()
        batch = next(it)
        # predict with a model
        model.predict_on_batch(batch["inputs"])
        model.predict_activation_on_batch(batch["inputs"], layer=len(model.model.layers)-2)
        if example == "rbp":
            model.predict_activation_on_batch(batch["inputs"], layer="flatten_6")



def test_keras_get_layers_and_outputs():
    model = kipoi.model.KerasModel(*get_sample_functional_model())
    selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs("shared_lstm")
    assert len(selected_layers) == 1
    assert selected_layers[0].name == "shared_lstm"
    assert len(sel_outputs) == 2
    assert len(sel_output_dims) == 2
    with pytest.raises(Exception):  # expect exception
        # LSTM activation layer has non-trivial input
        selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs("shared_lstm",
                                                                                     pre_nonlinearity=True)
    selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs(use_final_layer=True,
                                                                                 pre_nonlinearity=True)
    assert len(selected_layers) == 1
    assert selected_layers[0].name == "final_layer"
    assert len(sel_outputs) == 1
    assert sel_outputs[0] != selected_layers[0].output
    assert len(sel_output_dims) == 1
    selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs(use_final_layer=True,
                                                                                 pre_nonlinearity=False)
    assert len(selected_layers) == 1
    assert selected_layers[0].name == "final_layer"
    assert len(sel_outputs) == 1
    assert sel_outputs[0] == selected_layers[0].output
    assert len(sel_output_dims) == 1
    # using the sequential model
    model = kipoi.model.KerasModel(*get_sample_sequential_model())
    selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs(2)
    assert len(selected_layers) == 1
    assert selected_layers[0].name == "hidden"
    assert len(sel_outputs) == 1
    assert len(sel_output_dims) == 1
    selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs(use_final_layer=True,
                                                                                 pre_nonlinearity=True)
    assert len(selected_layers) == 1
    assert selected_layers[0].name == "final"
    assert len(sel_outputs) == 1
    assert sel_outputs[0] != selected_layers[0].output
    assert len(sel_output_dims) == 1
    selected_layers, sel_outputs, sel_output_dims = model.get_layers_and_outputs(use_final_layer=True,
                                                                                 pre_nonlinearity=False)
    assert len(selected_layers) == 1
    assert selected_layers[0].name == "final"
    assert len(sel_outputs) == 1
    assert sel_outputs[0] == selected_layers[0].output
    assert len(sel_output_dims) == 1


def test_generate_activation_output_functions():
    model = kipoi.model.KerasModel(*get_sample_functional_model())
    sample_input = get_sample_functional_model_input()
    act_fn = model._generate_activation_output_functions(layer="shared_lstm", pre_nonlinearity=False)
    act_fn(sample_input)
    act_fn_nl = model._generate_activation_output_functions(layer="final_layer", pre_nonlinearity=False)
    act_fn_nl(sample_input)
    act_fn_l = model._generate_activation_output_functions(layer="final_layer", pre_nonlinearity=True)
    act_fn_l(sample_input)

    # sequential model:
    model = kipoi.model.KerasModel(*get_sample_sequential_model())
    sample_input=get_sample_sequential_model_input()
    act_fn_nl = model._generate_activation_output_functions(layer="hidden", pre_nonlinearity=False)
    act_fn_nl(sample_input)
    act_fn_l = model._generate_activation_output_functions(layer="hidden", pre_nonlinearity=True)
    act_fn_l(sample_input)