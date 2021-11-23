# model.yaml

The model.yaml file describes the individual model in the model zoo. It defines its dependencies, framework, 
architecture, input / output schema, general information and more. Correct definitions in the model.yaml enable to make 
full use of Kipoi features and make sure that a model can be executed at any point in future.

To help understand the syntax of YAML please take a look at: 
[YAML Syntax Basics](http://docs.ansible.com/ansible/latest/YAMLSyntax.html#yaml-basics)

Here is an example `model.yaml`:

```yaml
defined_as: kipoi.model.KerasModel  
args:  # arguments of `kipoi.model.KerasModel`
    arch:
        url: https://zenodo.org/path/to/my/architecture/file
        md5: 1234567890abc
    weights:
        url: https://zenodo.org/path/to/my/model/weights.h5
        md5: 1234567890abc
default_dataloader: . # path to the dataloader directory. Or to the dataloader class, e.g.: `kipoiseq.dataloaders.SeqIntervalDl
info: # General information about the model
    authors: 
        - name: Your Name
          github: your_github_username
          email: your_email@host.org
    doc: Model predicting the Iris species
    cite_as: https://doi.org:/... # preferably a doi url to the paper
    trained_on: Iris species dataset (http://archive.ics.uci.edu/ml/datasets/Iris) # short dataset description
    license: MIT # Software License - defaults to MIT
dependencies:
    conda: # install via conda
      - python
      - h5py
      - pip
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

## defined_as

The model type refers to base framework which the model was defined in. Kipoi comes with a support for Keras, PyTorch, 
SciKit-learn and tensorflow models. To indicate which kind of model will be used the respective class name in Kipoi
 has to be used. Therefore `defined_as` can be one of the followinf values: `kipoi.model.KerasModel`, 
 `kipoi.model.PyTorchModel`, `kipoi.model.SklearnModel`, and `kipoi.model.TensorFlowModel`. If you wrote your own Kipoi
  model class, you called it `MyModel`, and you defined it in the file `my_model.py`, then the `type` field would be:
 `my_model.MyModel`.

The model type is required to find the right internal prepresentation of a model within Kipoi, which enables loading 
weights and architecture correctly and offers to have a unified API across frameworks.

In the model.yaml file the definition of a Keras model would like this:

```yaml
defined_as: kipoi.model.KerasModel  
```

## args
Model arguments define where the files are files and functions are located to instantiate the model. Most entries 
of `args` will contain links to zenodo or figshare downloads. The correct 
definition of `args` depends on the model `defined_as` that was selected:


### `kipoi.model.KerasModel` models

For Keras models the following args are available:

- `weights`: URL and md5 of the hdf5 weights or the hdf5 Keras model.
- `arch`: Architecture json model. If None, `weights` is assumed to speficy the whole model
- `custom_objects`: URL and md5 of python file defining the custom Keras objects in a `OBJECTS` dictionary
- `backend`: Keras backend to use ('tensorflow', 'theano', 'cntk')
- `image_dim_ordering`: `'tf'` or `'th'`: Whether the model was trained with using 'tf' ('channels_last') or 
'th' ('cannels_first') dimension ordering.

The Keras framework offers different ways to store model architecture and weights:

* Architecture and weights can be stored separately:

```yaml
defined_as: kipoi.model.KerasModel
args:
    arch: 
        url: https://zenodo.org/path/to/my/architecture/file
        md5: 1234567890abc
    weights: 
        url: https://zenodo.org/path/to/my/model/weights.h5
        md5: 1234567890abc
```

* The architecture can be stored together with the weights:

```yaml
defined_as: kipoi.model.KerasModel
args:
    weights:
        url: https://zenodo.org/path/to/my/model/weights.h5
        md5: 1234567890abc
```

In Keras models can have custom layers, which then have to be available at the instantiation of the Keras model,
those should be stored in one python file that is uploaded with the model architecture and weights. This file defines a
dictionary containing custom Keras components called `OBJECTS`. These objects will be added to `custom_objects` 
when loading the model with `keras.models.load_model`.

Example of a `custom_keras_objects.py`:
```python
from concise.layers import SplineT

OBJECTS = {"SplineT": SplineT}
```

Example of the corresponding model.yaml entry:
```yaml
defined_as: kipoi.model.KerasModel
args:
    ...
    custom_objects: custom_keras_objects.py
```
Here all the objects present in `custom_keras_objects.py` will be made available to Keras when loading the model.


### `kipoi.model.PyTorchModel` models

Pytorch offers much freedom as to how the model is stored. In Kipoi a pytorch model has the following `args`: 
`weights`, `module_class`, `module_kwargs`, `module_obj`.
PyTorch models require python code in which the model is defined. The code that defines the model should **not** 
attempt load the weights, as this is done inside the `PyTorchModel` class in Kipoi using the 
`model.load_state_dict(torch.load(weights))` command.

For example the pytorch model definition could be in a file `my_pytorch_model.py`:
```python
from torch import nn

class DummyModel(nn.Module):
    def __init__(self, x, y, z):
        super(DummyModel, self).__init__()
        # Some code here

    def forward(self, x):
        # some code here
        return x
```

Assuming that the `my_pytorch_model.py` file lies in the same folder as the `model.yaml`, the default way for loading 
this model in Kiopi is then as follows:

```yaml
defined_as: kipoi.model.PyTorchModel
args:
    module_class: my_pytorch_model.DummyModel
    module_kwargs: 
      x: 1
      y: 2
      z: 3
    weights: 
	    url: https://zenodo.org/path/to/my/model/weights.pth
	    md5: 1234567890abc
```

If the module class does not have any arguments then `module_kwargs` can be omitted.
 
If you use Sequential models (`torch.nn.Sequential`) or you generate a module instance in your `my_sequential.py` file, 
then you can use the `module_obj` in `model.yaml` to load that module:
  
```yaml
defined_as: kipoi.model.PyTorchModel
args:
    module_obj: my_sequential.sequential_model
	weights: 
        url: https://zenodo.org/path/to/my/model/weights.pth
        md5: 1234567890abc
```

where `my_sequential.py` for example contains:

```python
import torch
sequential_model = torch.nn.Sequential(...)
```

If you have trouble with the imports or if you would like to
import a module from a parent directory you can explicitly specify the python file path:

```python
defined_as: kipoi.model.PyTorchModel
args:
    module_file: ./my_sequential.py
    module_obj: sequential_model
	weights: 
        url: https://zenodo.org/path/to/my/model/weights.pth
        md5: 1234567890abc
```

If `cuda` is available on the system then the model will automatically be switched to cuda mode, so the 
user does not have to take care of that. 

### `kipoi.model.SklearnModel` models

SciKit-learn models can be loaded from a pickle file as defined below. The command used for loading is: 
`joblib.load(pkl_file)`

```yaml
defined_as: kipoi.model.SklearnModel
args:
  pkl_file: 
      url: https://zenodo.org/path/to/my/model.pkl
      md5: 1234567890abc
  predict_method: predict_proba  # Optional. predict by default. Available: predict, predict_proba, predict_log_proba
```

### `kipoi.model.TensorFlowModel` models
Tensorflow models are expected to be stored by calling `saver = tf.train.Saver(); saver.save(checkpoint_path)`. The 
`input_nodes` argument is then a string, list of strings or dictionary of strings that define the input node names. 
The `target_nodes` argument is a string, list of strings or dictionary of strings that define the model target node 
names.

```yaml
defined_as: kipoi.model.TensorFlowModel
args:
  input_nodes: "inputs"
  target_nodes: "preds"
  checkpoint_path: 
      url: https://zenodo.org/path/to/my/model.tf
      md5: 1234567890abc
```

If a model requires a constant feed of data which is not provided by the dataloader the `const_feed_dict_pkl` argument 
can be defined additionally to the above. Values given in the pickle file will be added to the batch samples created 
by the dataloader. If values with identical keys have been created by the dataloader they will be overwritten with 
what is given in `const_feed_dict_pkl`.

```yaml

defined_as: kipoi.model.TensorFlowModel
args:
  ...
  const_feed_dict_pkl:
      url: https://zenodo.org/path/to/my/const_feed_dict.pkl
      md5: 1234567890abc
```

### custom models

It is possible to defined a model class independent of the ones which are made available in Kipoi. In that case the 
contributor-defined `Model` class must be a subclass of `BaseModel` defined in `kipoi.model`. Custom models should 
never deviate from using only numpy arrays, lists thereof, or dictionaries thereof as input for the `predict_on_batch` 
function. This is essential to maintain a homogeneous and clear interface between dataloaders and models in the 
Kipoi zoo!

If for example a custom model class definition (`MyModel`) is in a file `my_model.py`, then the model.yaml will contain:

```yaml
defined_as: my_model.MyModel
```

Kipoi will then use an instance of MyModel as a model. Keep in mind that MyModel has to be subclass of `BaseModel`, 
which in other words means that `def predict_on_batch(self, x)` has to be implemented. So if `batch` is for example 
what the dataloader returns for a batch then `predict_on_batch(batch['inputs'])` has to work.

It is likely that `MyModel` will require additional files to work. The Kipoi way of using such files is by defining 
Model in the following way:

```python
from kipoi.model import BaseModel

class MyModel(BaseModel):
    def __init__(self, external_file):
        self.data = read_my_file(external_file)
        #...
```

The file will be downloaded from zenodo or figshare automatically and assigned to the `external_file` argument if the
`model.yaml` contains:

```yaml
defined_as: my_model.MyModel
args:
  external_file:
    default:
        url: https://zenodo.org/path/to/my/data
        md5: 1234567890abc
```





## info
The `info` field of a model.yaml file contains general information about the model.

* `authors`: a list of authors with the field: `name`, and the optional fields: `github` and `email`. Where the 
`github` name is the github user id of the respective author
* `doc`: Free text documentation of the model. A short description of what it does and what it is designed for.
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
  doc: My model description
  license: GNU
  tags:
    - TFBS
    - tag2
  cite_as: http://www.the_journal.com/mypublication
  trained_on: The XXXX dataset from YYYY
  training_procedure: 10-fold cross validation
```


## default_dataloader
The `default_dataloader` defines the dataloader that should be used for the given model. It can either be defined by 
a package like [kipoiseq](https://github.com/kipoi/kipoiseq) or it can be defined by the contributor.

#### Using a pre-defined dataloader

If one of the ready-made dataloaders on [kipoiseq](https://github.com/kipoi/kipoiseq) fits the needs of your model, 
then please follow the instructions on [kipoiseq](https://github.com/kipoi/kipoiseq). The `default_dataloader` in the 
`model.yaml` would then for example be:

```yaml
default_dataloader:
  defined_as: kipoiseq.dataloaders.SeqIntervalDl
  default_args:
    auto_resize_len: 1000
    alphabet_axis: 0
    dummy_axis: 1
    dtype: float32
```

#### Using a custom dataloader
If you need a specialised dataloader you are encouraged to used as many methods and classes from within 
[kipoiseq](https://github.com/kipoi/kipoiseq) as possible as their functionality is tested. See more information on 
writing a [dataloader](./04_Writing_dataloader.py.md) and its [companion](./03_Writing_dataloader.yaml.md) yaml. Both 
of those files should lie in the same folder as the `model.yaml`. Then the `default_dataloader` entry in `model.yaml` 
is:

```yaml
default_dataloader: .
```

It points to the location of the dataloader.yaml file. 
If dataloader.yaml lies in different subfolder then `default_dataloader: path/to/folder` would be used where 
dataloader.yaml would lie in `folder`.

## schema

Schema defines what the model inputs and outputs are, what they consist in and what the dimensions are.

`schema` contains two categories `inputs` and `targets` which each specify the shapes of the model input and model output.

In general model inputs and outputs can either be a numpy array, a list of numpy arrays or a dictionary (or 
`OrderedDict`) of numpy arrays. Whatever format is defined in the schema is expected to be produced by the dataloader 
and is expected to be accepted as input by the model. The three different kinds are represented by the single entries, 
lists or dictionaries in the yaml definition:

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
* `special_type`: Possibility to flag that respective input is a 1-hot encoded DNA sequence (`special_type: DNASeq`) or a string DNA sequence (`special_type: DNAStringSeq`).

### `targets`

The `targets` fields of `schema` may be lists, dictionaries or single occurences of the following entries:

* `shape`: Required: Details see in `input`
* `doc`: A free text description of the model input
* `name`: Name of model  target, not required if target is a dictionary.
* `column_labels`: Labels for the tasks of a multitask matrix output. Can be the file name of a text file containing the task labels (one label per line).



### How model types handle schemas
The different model types handle those three different encapsulations of numpy arrays differently:

#### `KerasModel` models
##### Input
In case a Keras model is used the batch produced by the dataloader is passed on as it is to the 
`model.predict_on_batch()` function. So if for example a dictionary is defined in the model.yaml and that is produced 
by the dataloader then this dicationary is passed on to `model.predict_on_batch()`.
##### Output
The model is expected to return the schema that is defined in model.yaml. If for example a model returns a list of 
numpy arrays then that has to be defined correctly in the model.yaml schema.

#### `PyTorchModel` models
Pytorch needs `torch.autograd.Variable` instances to work. Hence all inputs are automatically converted into `Variable` 
objects and results are converted back into numpy arrays transparently. If `cuda` is available the model will 
automatically be used in cuda mode and also the input variables will be switched to `cuda`.
##### Input
For prediction the following will happen to the tree different encapsulations of input arrays:

* A single array: Will be passed directly as the only argument to model call: `model(Variable(from_numpy(x)))`
* A list of arrays: The model will be called with the list of converted array as args (e.g.: `model(*list_of_variables)`)
* A dictionary of arrays: The model will be called with the dictionary of converted array as kwargs 
(e.g.: `model(**dict_of_variables)`)

##### Output
The model return values will be converted back into encapsulations of numpy arrays, where:

* a single `Variable` object will be converted into a numpy arrays
* lists of `Variable` objects will be converted into a list of numpy arrays in the same order and

#### `SklearnModel` models
The batch generated by the dataloader will be passed on directly to the SciKit-learn model using `model.predict(x)`, 
`model.predict_proba(x)` or `model.predict_log_proba` (depending on the `predict_method` argument).

#### `TensorFlowModel` models
##### Input
The `feed_dict` for running a tensorflow session is generated by converting the batch samples into the `feed_dict` 
using `input_nodes` defined in the `args` section of the model.yaml. For prediction the following will happen to the 
tree different encapsulations of input arrays:

* If `input_nodes` is a single string the model will be fed with a dictionary `{input_ops: x}`
* If `input_nodes` is a list then the batch is also exptected to be a list in the corresponding order and the feed 
dict will be created from that.
* If `input_nodes` is a dictionary then the batch is also exptected to be a dictionary with the same keys and the feed 
dict will be created from that.

##### Output
The return value of the tensorflow model is returned without further transformations and the model outpu schema defined 
in the `schema` field of model.yaml has to match that.


## dependencies
One of the core elements of ensuring functionality of a model is to define software dependencies correctly and 
strictly. Dependencies can be defined for conda and for pip using the `conda` and `pip` sections respectively.

Both can either be defined as a list of packages or as a text file (ending in `.txt`) which lists the dependencies.

Conda as well as pip dependencies can and should be defined with exact versions of the required packages, as defining 
a package version using e.g.: `package>=1.0` is very likely to break at some point in future.

If your model is a python-based model and you have not tested whether your model works in python 2 and python 3, then 
make sure that you also add the correct python version as a dependency e.g.: `python=2.7`.

###conda
Conda dependencies can be defined as lists or if the dependencies are defined in a text file then the path of the text 
must be given (ending in `.txt`).

If conda packages need to be loaded from a channel then the nomenclature `channel_name::package_name` can be used.

###pip
Pip dependencies can be defined as lists or if the dependencies are defined in a text file then the path of the text 
must be given (ending in `.txt`).