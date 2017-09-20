# Model-zoo


![Build Status](https://travis-ci.com/kipoi/model-zoo.svg?token=EQhjUezyCnoyp9tzNxc3&branch=master)

## Install

After cloning the repository `git clone https://github.com/kipoi/model-zoo.git`, run:

```
pip install -U .
```

If you wish to develop, run instead:

```
pip install -e '.[develop]'
```

This will install some additional packages like `pytest`.

## Usage


```
$ modelzoo
usage: modelzoo <command> [-h] ...

    Kipoi model-zoo command line tool. Available sub-commands:

    # Using the models
    predict          Run the model prediction.
    score_variants   Run prediction on a list of regions
    pull             Downloads the directory associated with the model
    preproc          Returns an hdf5 array.
    test             Runs a set of unit-tests for the model
```

## Running the examples

Try out running the examples in `examples/`

```
modelzoo test examples/extended_coda
```

## Configure `model_zoo`

Setup your preference in: `.kipoi/config.yaml`

You can add your own model sources. See [docs/model_sources.md](docs/model_sources.md) for more information.

## Python SDK

Provides functionality to:
- Load the model
- Predict
- Run the pre-processor
- List all available models

## Documentation

Explore the markdown files in [docs/](docs/):
- Command-line interface [docs/cli.md](docs/cli.md)
- Python interface [docs/py-interface.md](docs/py-interface.md)
- Model sources configuration [docs/model_sources.md](docs/model_sources.md)
  - setup your own model zoo
- Contributing models [docs/contributing_models.md](docs/contributing_models.md)
- **Examples**
  - Python interface [nbs/python-sdk.ipynb](nbs/python-sdk.ipynb)


---------------------------------------------------------------

## TODO

- [ ] Figure out where to upload the models for now
  - Need to have a table  listing all the models
    - model_name, repo URL
      - Users contribute by making a PR to that URL
		- TravisCI tests only models that were changed in the PR
          - test by saving a version + meta-info in a DB
	    - Model+version = one row

## Issues

What are we missing?
- virtual-env setup
- preprocessors from other languages?
  - maybe put it on hold for now?
  
How to start a pre-processor in a separate environment?
- Enforce using conda?
- Use a separate process?
- this could allow running preprocessor from multiple versions in parallel

How to check for malicious software?
  - running preprocessors
    - make the preprocessor code well available

## Link collection

- [Kipoi Google drive](https://drive.google.com/drive/folders/0B9fJIVHGqt20b05GMzBZUVQzRVU)


### Useful links

- https://github.com/deepchem/deepchem
- https://developer.apple.com/documentation/coreml
