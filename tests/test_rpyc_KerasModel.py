import warnings

warnings.filterwarnings('ignore')
import pytest
import sys
import os
import yaml
from contextlib import contextmanager
import kipoi
from kipoi.pipeline import install_model_requirements
from kipoi_utils.utils import Slice_conv
import config
import numpy as np
from kipoi_utils.utils import cd
from kipoi.rpyc_model import *

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


def get_sample_functional_model_input(kind="list"):
    import numpy as np
    if kind == "list":
        return [np.random.rand(1, 280, 256), np.random.rand(1, 280, 256)]
    if kind == "dict":
        return {"FirstInput": np.random.rand(1, 280, 256), "SecondInput": np.random.rand(1, 280, 256)}


def get_single_layer_model():
    from keras.layers import Dense
    from keras.models import Sequential
    model = Sequential()
    model.add(Dense(1, input_shape=(1,)))
    model.layers[0].set_weights([np.array([[0.5]]), np.array([0.0])])
    import tempfile
    temp_weights = tempfile.mkstemp()[1]
    temp_arch = tempfile.mkstemp()[1]
    with open(temp_arch, "w") as ofh:
        ofh.write(model.to_json())
    model.save_weights(temp_weights)
    return temp_weights, temp_arch


def get_sample_functional_model():
    # Keras ["2.0.4", "2.1.5"]
    import keras
    if keras.__version__.startswith("2."):
        # from the Keras Docs - functional models
        import keras
        from keras.layers import Input, LSTM, Dense
        from keras.models import Model
        #
        tweet_a = Input(shape=(280, 256), name="FirstInput")
        tweet_b = Input(shape=(280, 256), name="SecondInput")
        #
        # This layer can take as input a matrix
        # and will return a vector of size 64
        shared_lstm = LSTM(64, name="shared_lstm")
        #
        # When we reuse the same layer instance
        # multiple times, the weights of the layer
        # are also being reused
        # (it is effectively *the same* layer)
        encoded_a = shared_lstm(tweet_a)
        encoded_b = shared_lstm(tweet_b)
        # We can then concatenate the two vectors:
        merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1, name="concatenation")
        # And add a logistic regression on top
        predictions = Dense(1, activation='sigmoid', name="final_layer")(merged_vector)
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
        tweet_a = Input(shape=(280, 256), name="FirstInput")
        tweet_b = Input(shape=(280, 256), name="SecondInput")
        #
        # This layer can take as input a matrix
        # and will return a vector of size 64
        shared_lstm = LSTM(64, name="shared_lstm")
        # When we reuse the same layer instance
        # multiple times, the weights of the layer
        # are also being reused
        # (it is effectively *the same* layer)
        encoded_a = shared_lstm(tweet_a)
        encoded_b = shared_lstm(tweet_b)
        # We can then concatenate the two vectors:
        # merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)
        merged_vector = merge([encoded_a, encoded_b], mode='concat', name="concatenation")
        # And add a logistic regression on top
        predictions = Dense(1, activation='sigmoid', name="final_layer")(merged_vector)
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
    return [np.random.rand(1, 784)]


def get_sample_sequential_model():
    # from the Keras Docs - sequential models
    from keras.layers import Dense
    from keras.models import Sequential
    model = Sequential([
        Dense(32, input_shape=(784,), activation='relu', name="first"),
        Dense(32, activation='relu', name="hidden"),
        Dense(10, activation='softmax', name="final"),
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
@pytest.mark.parametrize("port", [18838])#,18839,18840,18838,18839,18840])
def test_predict_on_batch(example, port):
    """Test extractor
    """

    import keras
    backend = keras.backend._BACKEND

    if example == "rbp":
      pytest.skip("")


    if backend == 'theano' and example == "rbp":
        pytest.skip("extended_coda example not with theano ")
    #
    example_dir = "example/models/{0}".format(example)
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
    


    # get model
    model = kipoi.get_model(example_dir, source="dir")

    # get remote model
    s = kipoi.rpyc_model.ServerArgs(env_name=None,  address='localhost', port=port, logging_level=0)
    remote_model = kipoi.get_model(example_dir, source="dir", server_settings=s)
    #
    with cd(example_dir + "/example_files"):
        # initialize the dataloader
        dataloader = Dl(**test_kwargs)
        #
        # sample a batch of data
        it = dataloader.batch_iter()
        batch = next(it)
        # predict with a model
        res = model.predict_on_batch(batch["inputs"])
        remote_res = remote_model.predict_on_batch(batch["inputs"])

        numpy.testing.assert_allclose(res, remote_res)