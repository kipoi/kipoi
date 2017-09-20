This docs shows how our API should ideally look like.

## Load the library

```python
import modelzoo
```

## List all available models

Returns a pandas data.frame

```python
>>> df = modelzoo.list_models()
>>> df
  source          model           name  version          author  \
0  kipoi            rbp      rbp_eclip      0.1      Ziga Avsec   
1  kipoi  extended_coda  extended CODA      0.1  Johnny Israeli   

                               description   type                inputs  \
0                   RBP binding prediction  keras  [seq, dist_polya_st]   
1  Single bp resolution ChIP-seq denoising  keras  [H3K27ac_subsampled]   

          targets tags
0  [binding_site]   []
1       [H3K27ac]   []
```

Get all available information about some particular model (as python dictionary)

```python
>>> modelzoo.model_info("rbp", source="kipoi")
{'author': 'Ziga Avsec',
 'description': 'RBP binding prediction',
 'model': {'args': {'arch': 'model_files/model.json',
                    'custom_objects': 'custom_keras_objects.py',
                    'weights': 'model_files/weights.h5'},
           'inputs': {'dist_polya_st': {'description': 'Distance to poly-a '
                                                       'site transformed with '
                                                       'B-splines',
                                        'shape': '(None, 1, 10)'},
                      'seq': {'provide_ranges': True, 'type': 'DNA'}},
           'targets': {'binding_site': {'shape': '(None, 1)'}},
           'type': 'keras'},
 'name': 'rbp_eclip',
 'version': 0.1}
```

## Extracor

### Load the extractor

```python
Extractor = modelzoo.load_extractor("username/model1")
```
### Get information

```python
print(Extractor.__doc__)
```

### Initialize it

```python
extractor = Extractor(infile="~/file.txt", seq_len=10)
```

### Draw a few batches

```python
from torch.utils.data import DataLoader
from modelzoo.data import numpy_collate
batch_iter = iter(DataLoader(extractor,
                             batch_size=3,
                             collate_fn=numpy_collate,
                             num_workers=3))
x = next(batch_iter)
print(x)
```

## Model

### Load the model

Refer by `username/model`

```python
# Kipoi model
model = modelzoo.load_model("model1", source="kipoi")
model = modelzoo.load_model("model1") # source defaults to "kipoi"

# Most recent model - TODO - implement the versioning logic...
model = modelzoo.load_model("username/model1")
# Specific version
model = modelzoo.load_model("username/model1/v0.1")

# From directory
model = modelzoo.load_model("~/mymodels/model1", source="dir")

# From github
# Add source in ~/.kipoi/config.yaml or run
modelzoo.config.add_source("my_github_zoo",
                           modelzoo.remote.GitModelSource(remote_url="git@github.com/username/repo",
						                                  local_path="/tmp/models/"))
model = modelzoo.load_model("mymodel", source="my_github_zoo")
```

### Run predictions

```python
y = model.predict_on_batch(x["inputs"])
```

## Extractor + Model bundle

```python
Me = modelzoo.pieline.ModelExtractor("username/model1")

## Get info
print(Me.__doc__)

## Setup
me = Me(infile="~/file.txt", seq_len=10)

y = me.predict()
batch_iter = me.predict_generator(batch_size=3)
```
