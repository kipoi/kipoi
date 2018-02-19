# Kipoi

[![CircleCI](https://circleci.com/gh/kipoi/kipoi.svg?style=svg&circle-token=afc949457e09baf22e3b3cc3f5ffebb4e140b1f9)](https://circleci.com/gh/kipoi/kipoi)

Kipoi defines a common 'model-zoo' API for predictive models. This repository implements a 
command-line interface (CLI) and a python SDK to query (and use) the Kipoi models. The Kipoi models can be hosted in 
public models sources like [github.com/kipoi/models](https://github.com/kipoi/models) or in your own private
model sources.

![img](docs/img/kipoi-workflow.png)

## Installation

### 1. Install miniconda/anaconda

Kipoi requires [conda](https://conda.io/) to manage model dependencies.
Make sure you have either anaconda ([download page](https://conda.io/miniconda.html)) or miniconda ([download page](https://www.anaconda.com/download/)) installed.

### 2. Install Git LFS

For downloading models, Kipoi uses [Git Large File Storage](https://git-lfs.github.com/) (LFS). To install it on Ubuntu, run:

```bash
# on Ubuntu
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git-lfs
git-lfs install
```

Alternatively, install it through conda:

```bash
conda install -c conda-forge git-lfs 
```


### 3. Install Kipoi

Next, install Kipoi using [pip](https://pip.pypa.io/en/stable/):

```
git clone https://github.com/kipoi/kipoi.git
pip install kipoi/
```

## Development

If you wish to develop `kipoi`, run instead:

```
conda install pytorch-cpu
pip install -e '.[develop]'
```

This will install some additional packages like `pytest`. You can test the package by running `py.test`. 

If you wish to run tests in parallel, run `py.test -n 6`.

## Quick start

### Using Kipoi models from python

```python
import kipoi

# list all the available models
kipoi.list_models()  

# Model ----------------------
# Load the model from github.com/kipoi/models/rbp
model = kipoi.get_model("rbp", source="kipoi") # source="kipoi" is the default

# Load the model from a local directory
model = kipoi.get_model("~/mymodels/rbp", source="dir")  
# Note: Custom model sources are defined in ~/.kipoi/config.yaml

# See the information about the author:
model.info

# Access the default dataloader
model.default_dataloader

# Access the Keras model
model.model

# Predict on batch - implemented by all the models regardless of the framework
# (i.e. works with sklearn, Keras, tensorflow, ...)
model.predict_on_batch(x)

# Get predictions for the raw files
# Kipoi runs: raw files -[dataloader]-> numpy arrays -[model]-> predictions 
model.pipeline.predict({"dataloader_arg1": "inputs.csv"})

# Dataloader -------------------
Dl = kipoi.get_dataloader_factory("rbp") # returns a class that needs to be instantiated
dl = Dl(dataloader_arg1="inputs.csv")  # Create/instantiate an object

# batch_iter - common to all dataloaders
# Returns an iterator generating batches of model-ready numpy.arrays
it = dl.batch_iter(batch_size=32)
out = next(it)  # {"inputs": np.array, (optional) "targets": np.arrays.., "metadata": np.arrays...}

# load the whole dataset into memory
dl.load_all()

# re-train the Keras model
dl = Dl(dataloader_arg1="inputs.csv", targets_file="mytargets.csv")
it_train = dl.batch_train_iter(batch_size=32)  
# batch_train_iter is a convenience wrapper of batch_iter
# yielding (inputs, targets) tuples indefinitely
model.model.fit_generator(it_train, steps_per_epoch=len(dl)//32, epochs=10)
```

For more information see: [nbs/python-sdk.ipynb](nbs/python-sdk.ipynb)

### Using Kipoi models from the command-line

```
$ kipoi
usage: kipoi <command> [-h] ...

    # Kipoi model-zoo command line tool. Available sub-commands:
    ls               List all the available models
    predict          Run the model prediction.
    pull             Downloads the directory associated with the model
    preproc          Returns an hdf5 array.
    test             Runs a set of unit-tests for the model

    # Further sub-commands:
    postproc         Tools for model postprocessing like variant effect prediction
    env              Tools to work with kipoi conda environments
```

Explore the CLI usage by running `kipoi <command> -h`. Also, see [docs/cli.md](docs/cli.md) for more information.

### Configure Kipoi in `.kipoi/config.yaml`

Setup your preference in: `.kipoi/config.yaml`

You can add your own model sources. See [docs/model_sources.md](docs/model_sources.md) for more information.

### Contributing models

See [nbs/contributing_models.ipynb](nbs/contributing_models.ipynb).

## Postprocessing

### SNV effect prediction

Functionality to predict the effect of SNVs is available in the API as well as in the command line interface. The input
is a VCF which can then be and returned in the process. For more details on the requirements for the models and
 dataloaders please check the documentation mentioned below.


## Documentation

To get started, please read the following ipynb's:

- [nbs/contributing_models.ipynb](nbs/contributing_models.ipynb)
- [nbs/python-sdk.ipynb](nbs/python-sdk.ipynb)
- [nbs/variant_effect_prediction.ipynb](nbs/variant_effect_prediction.ipynb)

This will provide pointers to the rest of the documentation in [docs/](docs/):

- [docs/cli.md](docs/cli.md) - command-line interface
- [docs/writing_dataloaders.md](docs/writing_dataloaders.md)- describes all supported dataloader types
- [docs/writing_models.md](docs/writing_models.md) - describes all supported model types
- [docs/model_sources.md](docs/model_sources.md) - describes how to setup your own model zoo
