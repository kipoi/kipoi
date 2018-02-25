# model.yaml

The model.yaml file describes the individual model in the model zoo. It defines its dependencies, framework, architecture, input / output schema, general information and more. Correct defintions in the model.yaml enable to make full use of Kipoi features and make sure that a model can be executed at any point in future.

To help understand the synthax of YAML please take a look at: [YAML Synthax Basics](http://docs.ansible.com/ansible/latest/YAMLSyntax.html#yaml-basics)

Here is an example `model.yaml`:

```yaml
type: keras  # use `kipoi.model.KerasModel`
args:  # arguments of `kipoi.model.KerasModel`
    arch: model_files/model.json
    weights: model_files/weights.h5
default_dataloader: . # path to the dataloader directory. Here it's defined in the same directory
info: # General information about the model
    authors: 
        - name: Your Name
          github: your_github_username
          email: your_email@host.org
    doc: Model predicting the Iris species
    version: 0.1  # optional 
    cite_as: https://doi.org:/... # preferably a doi url to the paper
    trained_on: Iris species dataset (http://archive.ics.uci.edu/ml/datasets/Iris) # short dataset description
    license: MIT # Software License - defaults to MIT
dependencies:
    conda: # install via conda
      - python=3.5
      - h5py
      # - soumith::pytorch  # specify packages from other channels via <channel>::<package>      
    pip:   # install via pip
      - keras>=2.0.4
      - tensorflow>=1.0
schema:  # Model schema
    inputs:
        features:
            shape: (4,)  # array shape of a single sample (omitting the batch dimension)
            doc: "Features in cm: sepal length, sepal width, petal length, petal width."
    targets:
        shape: (3,)
        doc: "One-hot encoded array of classes: setosa, versicolor, virginica."
```

The model.yaml file has the following mandatory fields:

## type

The model type refers to base framework which the model was defined in. Kipoi comes with a support for Keras, PyTorch, SciKit-learn and tensorflow models. To indicate which kind of model will be used the following values for `type` are allowed: `keras`, `pytorch`, `sklearn`, `tensorflow`, and `custom`.

The model type is required to find the right internal prepresentation of a model within Kipoi, which enables loading weights and architecture correctly and offers to have a unified API across frameworks.

In the model.yaml file the definition of a Keras model would like this:

```yaml
type: keras
```

## args
Model arguments define where the files are files and functions are located to instantiate the model. Most entries of `args` will contain paths to files, those paths are relative to the location of the model.yaml file. The correct definition of `args` depends on the `type` that was selected:


### `keras` models

For Keras models the following args are available: `weights`, `arch`, `custom_objects`. The Keras framework offers different ways to store model architecture and weights:

* Architecture and weights can be stored separately:

```yaml
type: keras
args:
    arch: model_files/model.json
    weights: model_files/weights.h5
```

* The architecture can be stored together with the weights:

```yaml
type: keras
args:
    weights: model_files/model.h5
```

In Keras models can have custom layers, which then have to be available at the instantiation of the Keras model, those should be stored in one python file that comes with the model architecture and weights. This file defines a dictionary containing custom Keras components called `OBJECTS`. These objects will be added to `custom_objects` when loading the model with `keras.models.load_model`.

Example of a `custom_keras_objects.py`:
```python
from concise.layers import SplineT

OBJECTS = {"SplineT": SplineT}
```


Example of the corresponding model.yaml entry:
```yaml
type: keras
args:
    ...
    custom_objects: model_files/custom_keras_objects.py
```
Here all the objects present in `model_files/custom_keras_objects.py` will be made available to Keras when loading the model.

### `pytorch` models

Pytorch offers much freedom as to how the model is stored. In Kipoi a pytorch model has the following `args`: `file`, `build_fn`, `weights`. If `cuda` is available the model will automatically be switched to cuda mode, so the user does not have to take care of that and the `build_fn` should not attempt to do this conversion. The following ways of instantiating a model are supported:

* Build function: In the example below Kipoi expects that when calling `get_full_model()` (which is defined in `model_files/model_def.py`) A pytorch model is returned that for which the weights have already been loaded.

```yaml
type: pytorch
args:
    file: model_files/model_def.py
    build_fn: get_full_model
```

* Build function + weights: In the example below the model is instantiated by calling `get_model()` which can be found in `model_files/model_def.py`. After that the weights will be loaded by executing `model.load_state_dict(torch.load(weights))`.

```yaml
type: pytorch
args:
    file: model_files/model_def.py
    build_fn: get_model
    weights: model_files/weights.pth
```

* Architecture and weights in one file: In this case Kipoi assumes that `model = torch.load(weights)` will be a valid pytorch model. Care has to be taken when storing the architecture a model this way as only standard pytorch layers will be loaded correctly, please see the pytorch documentation for details.

```yaml
type: pytorch
args:
    weights: model_files/model.pth
```

### `sklearn` models

SciKit-learn models can be loaded from a pickle file as defined below. The command used for loading is: `joblib.load(pkl_file)`

```yaml
type: sklearn
args:
  pkl_file: model_files/model.pkl
```

### `tensorflow` models
Tensorflow models are expected to be stored by calling `saver = tf.train.Saver(); saver.save(checkpoint_path)`. The `input_nodes` argument is then a string, list of strings or dictionary of strings that define the input node names. The `target_nodes` argument is a string, list of strings or dictionary of strings that define the model target node names.

```yaml
type: tensorflow
args:
  input_nodes: "inputs"
  target_nodes: "preds"
  checkpoint_path: "model_files/model.tf"
```

If a model requires a constant feed of data which is not provided by the dataloader the `const_feed_dict_pkl` argument can be defined additionally to the above. Values given in the pickle file will be added to the batch samples created by the dataloader. If values with identical keys have been created by the dataloader they will be overwritten with what is given in `const_feed_dict_pkl`.

```yaml

type: tensorflow
args:
  ...
  const_feed_dict_pkl: "model_files/const_feed_dict.pkl"
```

### `custom` models

It is possible to defined a model class independent of the ones which are made available in Kipoi. In that case the contributor-defined `Model` class must be a subclass of `BaseModel` defined in `kipoi.model`. Custom models should never deviate from using only numpy arrays, lists thereof, or dictionaries thereof as input for the `predict_on_batch` function. This is essential to maintain a homogeneous and clear interface between dataloaders and models in the Kipoi zoo!

If for example a custom model class definition (`MyModel`) lies in a file `my_model.py`, then the model.yaml will contain:

```yaml
type: custom
args:
  file: my_model.py
  object: MyModel
```

Kipoi will then use an instance of MyModel as a model. Keep in mind that MyModel has to be subclass of `BaseModel`, which in other words means that `def predict_on_batch(self, x)` has to be implemented. So if `batch` is for example what the dataloader returns for a batch then `predict_on_batch(batch['inputs'])` has to work.




## info
The `info` field of a model.yaml file contains general information about the model.

* `authors`: a list of authors with the field: `name`, and the optional fields: `github` and `email`. Where the `github` name is the github user id of the respective author
* `doc`: Free text documentation of the model. A short description of what it does and what it is designed for.
* `version`: Model version
* `license`: String indicating the license, if not defined it defaults to `MIT`
* `tags`: A list of key words describing the model and its use cases
* `cite_as`: Link to the journal, arXiv, ...
* `trained_on`: Description of the training dataset
* `training_procedure`: Description of the training procedure

A dummy example could look like this:

```yaml
info:
  authors:
    - name: My Name
      github: myGithubName
      email: my@email.com
    - name: Second Author
  doc: My fancy model description
  version: 1.0
  license: GNU
  tags:
    - TFBS
    - tag2
  cite_as: http://www.the_journal.com/mypublication
  trained_on: The XXXX dataset from YYYY
  training_procedure: 10-fold cross validation
```


## default_dataloader
The `default_dataloader` points to the location of the dataloader.yaml file. By default this will be in the same folder as the model.yaml file, in which case `default_dataloader` doesn't have to be defined.

If dataloader.yaml lies in different subfolder then `default_dataloader: path/to/folder` would be used where dataloader.yaml would lie in `folder`.

## schema

Schema defines what the model inputs and outputs are, what they consist in and what the dimensions are.

`schema` contains two categories `inputs` and `targets` which each specify the shapes of the model input and model output.

In general model inputs and outputs can either be a numpy array, a list of numpy arrays or a dictionary (or `OrderedDict`) of numpy arrays. Whatever format is defined in the schema is expected to be produced by the dataloader and is expected to be accepted as input by the model. The three different kinds are represented by the single entries, lists or dictionaries in the yaml definition:

* A single numpy array as input or target:

```yaml
schema:
    inputs:
       name: seq
       shape: (1000,4)
```

* A list of numpy arrays as inputs or targets:

```yaml
schema:
    targets:
       - name: seq
         shape: (1000,4)
       - name: inp2
         shape: (10)
```

* A dictionary of numpy arrays as inputs or targets:

```yaml
schema:
    inputs:
       seq:
         shape: (1000,4)
       inp2:
         shape: (10)
```

### `inputs`
The `inputs` fields of `schema` may be lists, dictionaries or single occurences of the following entries:

* `shape`: Required: A tuple defining the shape of a single input sample. E.g. for a model that predicts a batch of `(1000, 4)` inputs `shape: (1000, 4)` should be set. If a dimension is of variable size then the numerical should be replaced by `None`.
* `doc`: A free text description of the model input
* `name`: Name of model input , not required if input is a dictionary.
* `special_type`: Possibility to flag that respective input is a 1-hot encoded DNA sequence (`special_type: DNASeq`) or a string DNA sequence (`special_type: DNAStringSeq`), which is important for variant effect prediction.

### `targets`

The `targets` fields of `schema` may be lists, dictionaries or single occurences of the following entries:

* `shape`: Required: Details see in `input`
* `doc`: A free text description of the model input
* `name`: Name of model  target, not required if target is a dictionary.
* `column_labels`: Labels for the tasks of a multitask matrix output. Can be the file name of a text file containing the task labels (one label per line).



### How model types handle schemas
The different model types handle those three different encapsulations of numpy arrays differently:

#### `keras` type models
##### Input
In case a Keras model is used the batch produced by the dataloader is passed on as it is to the `model.predict_on_batch()` function. So if for example a dictionary is defined in the model.yaml and that is produced by the dataloader then this dicationary is passed on to `model.predict_on_batch()`.
##### Output
The model is expected to return the schema that is defined in model.yaml. If for example a model returns a list of numpy arrays then that has to be defined correctly in the model.yaml schema.

#### `pytorch` type models
Pytorch needs `torch.autograd.Variable` instances to work. Hence all inputs are automatically converted into `Variable` objects and results are converted back into numpy arrays transparently. If `cuda` is available the model will automatically be used in cuda mode and also the input variables will be switched to `cuda`.
##### Input
For prediction the following will happen to the tree different encapsulations of input arrays:

* A single array: Will be passed directly as the only argument to model call: `model(Variable(from_numpy(x)))`
* A list of arrays: The model will be called with the list of converted array as args (e.g.: `model(*list_of_variables)`)
* A dictionary of arrays: The model will be called with the dictionary of converted array as kwargs (e.g.: `model(**dict_of_variables)`)

##### Output
The model return values will be converted back into encapsulations of numpy arrays, where:

* a single `Variable` object will be converted into a numpy arrays
* lists of `Variable` objects will be converted into a list of numpy arrays in the same order and

#### `sklearn` type models
The batch generated by the dataloader will be passed on directly to the SciKit-learn model using `model.predict(x)`

#### `tensorflow` type models
##### Input
The `feed_dict` for running a tensorflow session is generated by converting the batch samples into the `feed_dict` using `input_nodes` defined in the `args` section of the model.yaml. For prediction the following will happen to the tree different encapsulations of input arrays:

* If `input_nodes` is a single string the model will be fed with a dictionary `{input_ops: x}`
* If `input_nodes` is a list then the batch is also exptected to be a list in the corresponding order and the feed dict will be created from that.
* If `input_nodes` is a dictionary then the batch is also exptected to be a dictionary with the same keys and the feed dict will be created from that.

##### Output
The return value of the tensorflow model is returned without further transformations and the model outpu schema defined in the `schema` field of model.yaml has to match that.


## dependencies
One of the core elements of ensuring functionality of a model is to define software dependencies correctly and strictly. Dependencies can be defined for conda and for pip using the `conda` and `pip` sections respectively.

Both can either be defined as a list of packages or as a text file (ending in `.txt`) which lists the dependencies.

Conda as well as pip dependencies can and should be defined with exact versions of the required packages, as defining a package version using e.g.: `package>=1.0` is very likely to break at some point in future.

###conda
Conda dependencies can be defined as lists or if the dependencies are defined in a text file then the path of the text must be given (ending in `.txt`).

If conda packages need to be loaded from a channel then the nomenclature `channel_name::package_name` can be used.

###pip
Pip dependencies can be defined as lists or if the dependencies are defined in a text file then the path of the text must be given (ending in `.txt`).

## postprocessing

The postprocessing section of a model.yaml is necessary to indicate that a model is compatible with a certain kind of postprocessing feature available in Kipoi. At the moment only variant effect prediction is available for postprocessing. To understand how to set your model up for variant effect prediction, please take a look at the documentation of variant effect prediction.
