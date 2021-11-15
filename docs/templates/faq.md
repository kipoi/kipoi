### What type of models are suitable for the Kipoi model repository?

The Kipoi model repository is restricted to trained models with application in genomics,. Specifically, we request at least one input data modality to be derived from DNA sequence (which includes amino acid sequences) or from a functional genomics assay such as ChIP-seq or protein mass-spectrometry. Moreover, models must be able to satisfy the specifications of the API ([model](http://kipoi.org/docs/contributing/05_Writing_model.py) and [dataloader](http://kipoi.org/docs/contributing/04_Writing_dataloader.py)).
Please contact us if the model you would like to share a model that doesn’t fit the scope. We would be happy to help you instantiate a new model repository for a different domain (say imaging).


### What models don't go into Kipoi?

Basically, models that don't fit [the above requirements](#what-models-go-to-kipoi). These are for example models that require training before they can be used (say imputation models that need to be trained on the specific dataset prior to application).


### What licenses are allowed?

Any license that allows the redistribution model of files uploaded to file-sharing services like Zenodo or Figshare. We encourage users to use one of the standard open-source software licenses such as MIT, BSD License, GNU Public License or Apache License ([Comparison of free and open-source software licenses](https://en.wikipedia.org/wiki/Comparison_of_free_and_open-source_software_licenses)). Please contact us if you would like to host the files on your own servers. We note that it is the users' responsibility not to break copy rights when (re-)using models that are available in the Kipoi model zoo. License is specified either in the LICENSE file present in the model directory or the license type is specified in [model.yaml](http://kipoi.org/docs/contributing/02_Writing_model.yaml).

### Versioning of models

We do not version models in the model.yaml. Instead, if substantial changes to the model were made we 
encourage the author/contributor to create a new model. For example, if `KipoiModel/model.yaml` gets extended, the author
should could create new models: `KipoiModel/v1/model.yaml` and `KipoiModel/v2/model.yaml` and keep a softlink to the 
most recent version in `KipoiModel/v2/model.yaml -> KipoiModel/model.yaml`. 
Micro-changes like updating the model description are also tracked using Git, hence a particular model version can be 
always referred to using the commit hash.

### Can the model be a binary executable?

Yes if the binary is compiled and distributed through Bioconda or Conda-Forge conda channels. Lsgkm-SVM is one such example. See its [model.py](https://github.com/kipoi/models/blob/master/lsgkm-SVM/model.py) for the implementation details.

### Trouble with system-wide libraries?

If you have trouble executing kipoi because of system-wide installed libraries you can execute the commands within our singularity containers: After installing [singularity], just add the `--singularity` argument to your kipoi predict command.

### Is it possible to perform transfer learning across machine learning frameworks?

It depends. Kipoi allows you to pre-compute the activations of the frozen part of the network and save them to a file. These activations can be used as input features for a model written in an arbitrary framework. See [this](https://github.com/kipoi/manuscript/blob/master/src/transfer_learning/pre-computed-tlearn.ipynb) notebook on how to do this. If you wish to fine-tune the the whole model in a different framework you would need to convert the model parameters yourself (we recommend using [ONNX](https://onnx.ai/) to do so). In the future, we plan to convert all the models to 
the ONNX format which will allow porting models across different frameworks. If you are interested or would like to get involved - here is the issue tracking this feature: https://github.com/kipoi/kipoi/issues/405.

### Does Kipoi support Windows?

For Windows users we suggest to use the docker container: https://hub.docker.com/r/kipoi/models/. Installing conda environments for the Kipoi models is not fully supported on Windows due to unavailability of certain conda packages for Windows. Specifically, cyvcf2 is not readily available for windows via Bioconda, neither is htslib/tabix. htslib has been enabled for Windows from version 1.6, but not on conda. We hope that we will be able to enable support for Windows when those packages become Windows-compatible.
