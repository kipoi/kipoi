## Python API

See the ipython notebook [tutorials/python-api](../tutorials/python-api.html) for additional information and a working example of the API. Here is a list of most useful python commands.

```python
import kipoi
```

### List all models

```python
kipoi.list_models()
```

### Get the model
Before getting started with models take a short look what a Kipoi model actually is. Kipoi model have to have the following folder structure in which all the relevant files have their assigned places:
```
├── model.yaml        # describes the model
├── dataloader.yaml   # (optional) describes the dataloader
└── dataloader.py     # (optional) implements the dataloader
```
The core file that defines a model is `model.yaml`, for more details please look at the docs for [contributing models](../contributing/01_Getting_started.html).

Now let's get started with the model:

```python
model = kipoi.get_model("Basset")
```
---
** Aside: `get_model` and models versus model groups**: 

>`get_model` expects to receive a path to a directory containing a `model.yaml` file.  This file specifies the underlying model, data loader, and other model attributes.  
>If instead you provide `get_model` a path to a model *group* (e.g "lsgkm-SVM/Tfbs/Ap2alpha/"), rather than one model (e.g "lsgkm-SVM/Tfbs/Ap2alpha/Helas3/Sydh_Std"), or any other directory without a `model.yaml` file, `get_model` will throw a `ValueError`.

---
If you want to access a model that is not part of the Kipoi model zoo, please use:
 
```python
model = kipoi.get_model("path/to/my/model", source="dir")
```

If you wish to access the model for a particular commit, use the github permalink:

```python
model = kipoi.get_model("https://github.com/kipoi/models/tree/7d3ea7800184de414aac16811deba6c8eefef2b6/pwm_HOCOMOCO/human/CTCF", source='github-permalink')
```


### Access information about the model
In the following commands a few properties of the model will be shown:

```python
model.info # Information about the author:

model.default_dataloader # Access the default dataloader

model.model # Access the underlying Keras model
```

### Test the model
Every Kipoi model comes with a small test dataset, which is used to assert its functionality in the nightly tests. This model test function can be accessed by:

```python
pred = model.pipeline.predict_example()
```

### Get predictions for the raw files

For any generation of the model output the dataloader has to be executed first. A dataloader will require input arguments in which the input files are defined, for example input fasta files or bed files, based on which the model input is generated. One way to display the keyword arguments a dataloader accepts is the following:

```python
model.default_dataloader.print_args()
```

The output of the function above will tell you which arguments you can use when running the following command. Alternatively, you can view the dataloader arguments on the model's website (`http://kipoi.org/models/<model>`). Let's assume that `model.default_dataloder.print_args()` has informed us that the dataloader accepts the arguments `dataloader_arg1` and `targets_file`. You can get the model prediction using `kipoi.pipeline.predict`:


```python
model.pipeline.predict({"dataloader_arg1": "...", "targets_file": "..."})
```

Specifically, for the `Basset` model, you would run the following:

```python
dl_kwargs = model.default_dataloader.download_example('example')

# Run the prediction
pred = model.pipeline.predict(dl_kwargs, batch_size=4)
```

### Setup the dataloader
If you don't want to use the `model.pipeline.predict` function, but you would rather execute the dataloader yourself then you can do the following:

```python
dl = model.default_dataloader(dataloader_arg1="...", targets_file="...")
```

This generates a dataloader object `dl`.

Note: `kipoi.get_model("<mymodel>").default_dataloader` is the same as `kipoi.get_dataloader_factory("<mymodel>")`

### Predict for a single batch
Data can be requested from the dataloader through its iterator functionality, which can then be used to perform model predictions. 


```python
# Get the batch iterator
it = dl.batch_iter(batch_size=32)

# get a single batch
single_batch = next(it)

```

It is important to note that the dataloader can also annotate model inputs with additional metadata. The `model.pipeline` command therefore selects the values in the `inputs` key as it is shown in the example:

```python
# Make a prediction
predictions = model.predict_on_batch(single_batch['inputs'])
```


### Re-train the model

```python
it_train = dl.batch_train_iter(batch_size=32)  # will yield tuples (inputs, targets) indefinitely

# Since we are using a Keras model, run:
model.model.fit_generator(it_train, steps_per_epoch=len(dl)//32, epochs=10)
```
