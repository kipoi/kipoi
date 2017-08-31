This document describes how to contribute a model.


## Folder structure

**TODO** - show an example project folder structure here

```
├── custom_keras_objects.py
├── model
│   ├── model.json
│   └── weights.h5
├── model.json -> model/model.json
├── model.yaml
├── preprocessor
│   └── encodeSplines.pkl
├── preprocessor.py
├── preprocessor_test_kwargs.json
├── preprocessor.yaml
├── __pycache__
│   ├── custom_keras_objects.cpython-35.pyc
│   └── preprocessor.cpython-35.pyc
├── readme.md
├── requirements.txt
├── test_files
│   ├── gencode_v25_chr22.gtf.pkl.gz
│   ├── hg38_chr22.fa
│   ├── hg38_chr22.fa.fai
│   ├── intervals.tsv
│   ├── targets.tsv
│   └── test.json
├── train_model.ipynb
└── weights.h5 -> model/weights.h5
```


## Individual files

### Model

#### `model.yml`

**TODO** - There is a duplicated entry with preprocessor inputs...

```yaml
author: The Author
name: rbp_eclip
version: 0.1
description: RBP binding prediction
model:
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

#### `custom_keras_objects.py`

Defines a dictionary containing custom Keras components called `OBJECTS`.
This will added to `custom_objects` when loading the model with `keras.models.load_model`.

```python
from concise.layers import SplineT

OBJECTS = {"SplineT": SplineT}
```

#### Model files `model/weights.h5, model/model.json`

- **TODO** - enforce them to be in a single folder

### Preprocessor

#### `preprocessor.yml`

```
author: The Author
name: rbp_eclip
version: 0.1
description: RBP binding prediction
preprocessor:
    function_name: preprocessor
    type: generator
    arguments:
        intervals_file: "string; tsv file with `chrom start end id score strand`"
        fasta_file: "string: Reference genome sequence"
        gtf_file: "file path; Genome annotation GTF file pickled using pandas."
        preproc_transformer: "file path; tranformer used for pre-processing."
        target: # TODO - doesn't fit with the testing framework...
          target_file: "file path; path to the targets (txt) file"
        batch_size: 4
    output:
        inputs:
            seq:
                type: DNA
                provide_ranges: True
            dist_polya_st:
                shape: (None, 1, 10)
                description: Distance to poly-a site transformed with B-splines
        targets: binding_site
```

#### `preprocessor.py`

Defines a function `preprocessor` - as specified in the `preprocessor.yml` file. 

**TODO** - describe the preprocessor output [Preprocessor output schema #2](https://github.com/kipoi/model-zoo/issues/2)

**TODO** - what if we already explicitly required it to be of class say `modelzoo.Preprocessor`?

### `test_files/`

**TODO** - agree on the definition...

### `requirements.txt`

**TODO** - move to conda environment (which also contains a python version)?

### `readme.md`

Optional
