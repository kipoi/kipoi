# Model-zoo


## Usage

```
python test_model_submission_api.py <example-directory>
```

Available examples:
- rbp
- extended_coda

---------------------------------------------------------------
## Usage

### CLI definition

Main arguments

- `<model>`: can be a git repository, local file path, remote file path or a short model string.

#### Pre-process

Returns an hdf5 array.

```
model_zoo preproc <model> <preprocessor inputs...> -o <output.h5>
```

#### Predict

```
model_zoo predict <model> <preprocessor inputs...> -o <output>
```

where `<model>` can be the directory containing the required files

#### Score variants

```
model_zoo score_variants <model> <preprocessor inputs...> <vcf file> -o <output>
```

#### Pull the model

Downloads the directory associated with the model

```
model_zoo pull <model> -o path
```

#### Push the model

- Maybe use the tools from docker?

```
model_zoo push <model-dir>
```

#### Test the model

Runs a set of unit-tests for the model

```
model_zoo test <model>
```

### Configure `model_zoo`

Setup your preference in: `.model_zoo/config.yaml`

```
cache_dir: .model_zoo/models/
add_model_dirs: [] # additional model directories
```

### Python SDK

- Load the model
- Predict
- Run the pre-processor
- List all available models


---------------------------------------------------------------

## TODO

- Refactor current repo into a proper python package
  - call it modelzoo ? 
- Setup unit-tests
  - Run the examples on TravisCI
- Figure out where to upload the models
  - Or host it yourself?
- Setup Wiki for documentation
  - Uploading models
  - Using models
	- CLI
	- In python


## Dev ideas

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
- cache models in .model_zoo/
  - config in .model_zoo/config

- Use Synapse to host models at the beginning?
  - 

## Issues

What are we missing?
- preprocessors from other languages?
  - maybe put it on hold for now?
- Using multiple preprocessors in parallel

How to check for malicious software?
  - running preprocessors
    - make the preprocessor code well available

How to setup docs?
  - have a wiki?


## Link collection

- [Kipoi Google drive](https://drive.google.com/drive/folders/0B9fJIVHGqt20b05GMzBZUVQzRVU)


### Useful links

- https://github.com/deepchem/deepchem
- https://developer.apple.com/documentation/coreml
