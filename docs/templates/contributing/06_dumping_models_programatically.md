## Contributing multiple very similar models

To easily contribute model groups with multiple models of the same kind, you can specify two files describing all the models:

- `model-template.yaml` - template for model.yaml
- `models.tsv` - tab-separated files holding custom model variables

First few lines of `model-template.yaml`:

```yaml
type: keras
args:
  arch:
    url: {{ args_arch_url }}
    md5: {{ args_arch_md5 }}
  weights:
    url: {{ args_weights_url }}
    md5: {{ args_weights_md5 }}	
```	



First few lines of `models.tsv`:

```tsv
model	args_arch	args_weights	args_arch_md5	args_weights_md5	args_arch_url	args_weights_url
A549_ENCSR000DDI	model_files/A549_ENCSR000DDI.json	model_files/A549_ENCSR000DDI.h5	6d3a971ce766128ca444dd70ef76df70	f23198b146ad8e4d6755cb215fe75e0f	https://zenodo.org/record/1466073/files/A549_ENCSR000DDI?download=1	https://zenodo.org/record/1466073/files/A549_ENCSR000DDI.h5?download=1
BE2C_ENCSR000DEB	model_files/BE2C_ENCSR000DEB.json	model_files/BE2C_ENCSR000DEB.h5	919b2f7f675bebb9217d95021d92af74	159ea3cb7985c08eab8f64151eb1799e	https://zenodo.org/record/1466073/files/BE2C_ENCSR000DEB?download=1	https://zenodo.org/record/1466073/files/BE2C_ENCSR000DEB.h5?download=1
BJ_ENCSR000DEA	model_files/BJ_ENCSR000DEA.json	model_files/BJ_ENCSR000DEA.h5	6d3a971ce766128ca444dd70ef76df70	9ad8797caff0dd0e8274de6befded4e7	https://zenodo.org/record/1466073/files/A549_ENCSR000DDI?download=1	https://zenodo.org/record/1466073/files/BJ_ENCSR000DEA.h5?download=1
CMK_ENCSR000DGJ	model_files/CMK_ENCSR000DGJ.json	model_files/CMK_ENCSR000DGJ.h5	6d3a971ce766128ca444dd70ef76df70	d5c0c9dd55f1056036cc300ec1f61e1d	https://zenodo.org/record/1466073/files/A549_ENCSR000DDI?download=1	https://zenodo.org/record/1466073/files/CMK_ENCSR000DGJ.h5?download=1
```

One row in `models.tsv` will represent a single model and will be used to populate `model-template.yaml` and construct `model.yaml` using [Jinja2 templating language](http://jinja.pocoo.org/docs/2.10/). This allows you to even write if statements in `model-template.yaml`. See [CpGenie model](https://github.com/kipoi/models/tree/master/CpGenie) as an example.
