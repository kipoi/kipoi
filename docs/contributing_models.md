This document describes how to contribute a model.


## Folder structure

```
.
├── extractor.py
├── extractor.yaml
├── extractor_files
│   └── encodeSplines.pkl
├── model.yaml
├── model
│   ├── custom_keras_objects.py
│   ├── model.json
│   └── weights.h5
├── readme.md
├── requirements.txt
└── test_files
    ├── gencode_v25_chr22.gtf.pkl.gz
    ├── hg38_chr22.fa
    ├── hg38_chr22.fa.fai
    ├── intervals.tsv
    ├── targets.tsv
    └── test.json
```


## Model definition

### `model.yml`

**TODO** - There is a duplicated entry with preprocessor inputs...

```yaml
author: The Author
name: rbp_eclip
version: 0.1
description: RBP binding prediction
model:
    type: keras
    args:
      arch: model/model.json
      weights: model/weights.h5
      custom_objects: model/custom_keras_objects.py
    inputs:
        seq:
            type: DNA
            provide_ranges: True
        dist_polya_st:
            shape: (None, 1, 10)
            description: Distance to poly-a site transformed with B-splines
    targets:
        binding_site:
            shape: (None, 1)
```

### `type: `

Defines the serialized model type.

#### Keras

```yaml
Model:
  type: Keras
  args:
    weights: model.h5 # - File path to the hdf5 weights or the hdf5 Keras model
    arch: model.json # - Architecture json model. If None, `weights_file` is assumed to speficy the whole model
    custom_objects: custom_keras_objects.py # - Python file defining the custom Keras objects
```

#### Sci-kit learn

```yaml
Model:
  type: sklearn
  args:
    file: asd.pkl  # File path to the dumped sklearn file in the pickle format.
```

#### Custom model 

```yaml
Model:
  type: custom
  args:
    file: model.py
    object: Model
```			

The defined class `Model` in `model.py` needs to implement the following methods:

- `def predict_on_batch(self, x)` - takes a batch of samples from extractor's returned `['inputs']`
field and predicts the target variable.

### `custom_objects: custom_keras_objects.py`

Defines a dictionary containing custom Keras components called `OBJECTS`.
This will added to `custom_objects` when loading the model with `keras.models.load_model`.

```python
from concise.layers import SplineT

OBJECTS = {"SplineT": SplineT}
```

### Model files `weights: model/weights.h5, arch: model/model.json`

## Preprocessor

### `preprocessor.yml`

```
author: The Author
name: rbp_eclip
version: 0.1
description: RBP binding prediction
extractor:
    type: Dataset
    defined_as: SeqDistDataset
    arguments:
        intervals_file: "string; tsv file with `chrom start end id score strand`"
        fasta_file: "string: Reference genome sequence"
        gtf_file: "file path; Genome annotation GTF file pickled using pandas."
        preproc_transformer: "file path; tranformer used for pre-processing."
        target: # TODO - doesn't fit with the testing framework...
          target_file: "file path; path to the targets (txt) file"
    output:
        inputs:
            seq:
                type: DNA
                provide_ranges: True
            dist_polya_st:
                shape: (1, 10)
                description: Distance to poly-a site transformed with B-splines
        targets: binding_site
```

### `preprocessor.py`

Defines a class inheriting from `kipoi.data.Dataset`, i.e. you have to implement two methods:

- `def __getitem__(self, index):` - get the item with index `index`` from your dataset
- `def __len__(self):` - return the number of elements in your dataset

## `test_files/`

**TODO** - agree on the definition...

## `requirements.txt`

**TODO** - move to conda environment (which also contains a python version)?

## `readme.md`

Optional
