## Other possible models

As seen in [../nbs/contributing_models.ipynb](../nbs/contributing_models.ipynb), we need to provide the serialized model 
and write the `model.yaml` file. This document describes all the supported model types.

### Keras

Save the model architecture to `model_files/model.json` and model weights to `model_files/model.h5`.
Top of the `model.yaml` file should then look like this.

```yaml
type: Keras
args:
  weights: model_files/model.h5 # - File path to the hdf5 weights or the hdf5 Keras model
  arch: model_files/model.json # - Architecture json model. If None, `weights_file` is assumed to speficy the whole model
  custom_objects: custom_keras_objects.py # - Python file defining the custom Keras objects
```

#### `custom_objects: custom_keras_objects.py`

With Keras, you can also provide custom layer implementations in `custom_keras_objects.py`. 
This file defines a dictionary containing custom Keras components called `OBJECTS`.
These object will be added to `custom_objects` when loading the model with `keras.models.load_model`.

```python
from concise.layers import SplineT

OBJECTS = {"SplineT": SplineT}
```


### Sci-kit learn

```yaml
type: sklearn
args:
  file: asd.pkl  # File path to the dumped sklearn file in the pickle format.
```

### Custom model 

```yaml
type: custom
args:
  file: model.py
  object: Model
```

The defined class `Model` in `model.py` needs to implement the following methods:

- `def predict_on_batch(self, x)` - takes a batch of samples from extractor's returned `['inputs']`
field and predicts the target variable.
