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
