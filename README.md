# Model-zoo


![Build Status](https://travis-ci.com/kipoi/model-zoo.svg?token=EQhjUezyCnoyp9tzNxc3&branch=master)

## Install

After cloning the repository `git clone https://github.com/kipoi/model-zoo.git`, run:

```
pip install -U .
```

## Usage


```
usage: modelzoo <command> [-h] ...

    Kipoi model-zoo command line tool. Available sub-commands:

    # Using the models
    predict          Run the model prediction.
    score_variants   Run prediction on a list of regions
    pull             Downloads the directory associated with the model
    preproc          Returns an hdf5 array.
    test             Runs a set of unit-tests for the model

    # Uploading your model
    push             Push

```

## Running the examples

Try out running the examples in `examples/`

```
modelzoo test examples/extended_coda
```

## Configure `model_zoo`

Setup your preference in: `.model_zoo/config.yaml`

```
cache_dir: .model_zoo/models/
add_model_dirs: [] # additional model directories, file_paths
```

## Python SDK

- Load the model
- Predict
- Run the pre-processor
- List all available models

## Documentation

Explore the markdown files in [docs/](docs/):
- Command-line interface [docs/cli.md](docs/cli.md/)
- Python interface [docs/py-interface.md](docs/py-interface.md)

---------------------------------------------------------------

## TODO

- [x] Refactor current repo into a proper python package - modelzoo
  - [x] Setup the command-line interface
- Setup unit-tests
  - [ ] Run the examples on TravisCI
- [x] Setup Wiki for documentation
  - Uploading models
	- Leave for later
  - Using models
	- CLI
	- In python
- [ ] Figure out where to upload the models for now
  - Need to have a table  listing all the models
    - model_name, repo URL
      - Users contribute by making a PR to that URL
		- TravisCI tests only models that were changed in the PR
          - test by saving a version + meta-info in a DB
	    - Model+version = one row
  - Use Synapse to host models at the beginning?
- Refactor:
  - have a class:
    - Model
      - KerasModel - inherits from Model?
  	- self.predict()
      - cls.load_model()
        - could also contain custom python code?
  		- use with other packages...
      - .save_model() ? 
    - Preprocessor (see pytorch)
  	- next()
      - length
        - Maybe enhance it with parallel processing
  		- the preprocessor runs in a separate thread within a custom python environment?
  - model and preprocessor factory methods

## Dev ideas


## Issues

How to start a pre-processor in a separate environment?
- Enforce using conda?
- Use a separate process?


What are we missing?
- preprocessors from other languages?
  - maybe put it on hold for now?
- Using multiple preprocessors in parallel, each running in a separate environment?

How to check for malicious software?
  - running preprocessors
    - make the preprocessor code well available


## Link collection

- [Kipoi Google drive](https://drive.google.com/drive/folders/0B9fJIVHGqt20b05GMzBZUVQzRVU)


### Useful links

- https://github.com/deepchem/deepchem
- https://developer.apple.com/documentation/coreml
