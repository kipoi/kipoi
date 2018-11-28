### What models go to Kipoi?

  - **Trained** machine learning models from the area of **genomics**. Those can be models of any kind including deep learning, HMMs, trees, etc.
  - Models for which the model contributor has the license to redistribute

### What models don't go into Kipoi?

Basically, models that don't fit [these requirements](#what-models-go-to-kipoi). In particular, models that require 
training before their prediction becomes sensible. These are for example models for imputation that need to be trained 
on the specific dataset prior to application.

### Why is training not supported in Kipoi?

Kipoi strives to generalise and simplify tasks in a `one-command-fits-all` fashion. This enables the handling and use 
of models in a transparent way. Training a machine learning model is a complex task and the workload of a model 
contributor to convert their model training code into code that is compatible with a general API would be very high. 
Additionally, since all our models and code are tested nightly, this would also be required for training, which would 
increase the computation time for testing immensely. 

### What licenses are allowed?

Any license that allows the redistribution of model(weights). It is the users' responsibility not to break copy rights 
when (re-)using models that are available in the Kipoi model zoo.

### Versioning of models

We do not version models in the model.yaml. Instead, if substantial changes to the model were made we 
encourage the author/contributor to create a new model. For example, if `KipoiModel/model.yaml` gets extended, the author
should could create new models: `KipoiModel/v1/model.yaml` and `KipoiModel/v2/model.yaml` and keep a softlink to the 
most recent version in `KipoiModel/v2/model.yaml -> KipoiModel/model.yaml`. 
Micro-changes like updating the model description are also tracked using Git, hence a particular model version can be 
always referred to using the commit hash.


### Trouble with system-wide libraries?

If you have trouble executing kipoi because of system-wide installed libraries you can use our singularity container
to run calculations. After installing singularity, just add the `--singularity` argument to your kipoi command.

### Is it possible to perform transfer learning between different frameworks or different versions of the same framework?

At the moment we don’t offer framework conversion, therefore transfer learning can only be performed within the same 
framework and version in which the original model is available. In the future, we plan to convert all the models to 
the ONNX format which will allow porting models across different frameworks. If you are insterested or keen to help - 
here is the issue tracking this feature: https://github.com/kipoi/kipoi/issues/405

### Do you support Windows?

For Windows users we suggest to use the docker container: https://hub.docker.com/r/kipoi/models/. Installing conda environments for the Kipoi models is not fully supported on Windows due to unavailability of certain conda packages for Windows. Specifically, cyvcf2 is not readily available for windows via Bioconda, neither is htslib/tabix. htslib has been enabled for Windows from version 1.6, but not on conda. We hope that we will be able to enable support for Windows when those packages become Windows-compatible.
