## Using Kipoi - Getting started

### Steps

#### 1. Install and setup Kipoi

First install Kipoi and select a model you want to work with using the [instructions](http://kipoi.org/docs/).

#### 2. Use the model

You can use the model from:

- Python
- Command-line interface
- R (via the [reticulate](https://github.com/rstudio/reticulate) package)

-----------------------------------------

### Python - quick start

See the ipython notebook [tutorials/python-api](../../tutorials/python-api/) for additional information and a working example of the API. Here is a list of most useful python commands.

```python
import kipoi
```

#### List all models

```python
kipoi.list_models()
```

#### Get the model
Before getting started with models take a short look what a Kipoi model actually is. Kipoi model have to have the following folder structure in which all the relevant files have their assigned places:

```
├── model.yaml        # describes the model
├── dataloader.yaml   # (optional) describes the dataloader
└── dataloader.py     # (optional) implements the dataloader
```
The core file that defines a model is `model.yaml`, for more details please look at the docs for [contributing models](../contributing/01_Getting_started/).

Now let's get started with the model:

```python
model = kipoi.get_model("rbp_eclip/AARS")
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


#### Access information about the model
In the following commands a few properties of the model will be shown:

```python
model.info # Information about the author

model.info.license # Display the model license

model.default_dataloader # Access the default dataloader

model.model # Access the underlying Keras model
```

#### License
As you have seen in the code snippet above every model has a `license` field. If you use or aim to redistribute the 
model please always make sure that the license gives you the permission to do so.

#### Test the model
Every Kipoi model comes with a small test dataset, which is used to assert its functionality in the nightly tests. This model test function can be accessed by:

```python
pred = model.pipeline.predict_example()
```

#### Get predictions for the raw files

For any generation of the model output the dataloader has to be executed first. A dataloader will require input arguments in which the input files are defined, for example input fasta files or bed files, based on which the model input is generated. One way to display the keyword arguments a dataloader accepts is the following:

```python
model.default_dataloader.print_args()
```

The output of the function above will tell you which arguments you can use when running the following command. Alternatively, you can view the dataloader arguments on the model's website (`http://kipoi.org/models/<model>`). Let's assume that `model.default_dataloder.print_args()` has informed us that the dataloader accepts the arguments `dataloader_arg1` and `targets_file`. You can get the model prediction using `kipoi.pipeline.predict`:


```python
model.pipeline.predict({"dataloader_arg1": "...", "targets_file": "..."})
```

Specifically, for the `rbp_eclip/AARS` model, you would run the following:

```python
# Make sure we are in the directory containing the example files
import os
os.chdir(os.path.expanduser('~/.kipoi/models/rbp_eclip/AARS'))

# Run the prediction
model.pipeline.predict({'intervals_file': 'example_files/intervals.bed', 
                        'fasta_file': 'example_files/hg38_chr22.fa', 
	                'gtf_file': 'example_files/gencode.v24.annotation_chr22.gtf', 
	                'filter_protein_coding': True, 
	                'target_file': 'example_files/targets.tsv'})
```

#### Setup the dataloader
If you don't want to use the `model.pipeline.predict` function, but you would rather execute the dataloader yourself then you can do the following:

```python
dl = model.default_dataloader(dataloader_arg1="...", targets_file="...")
```

This generates a dataloader object `dl`.

Note: `kipoi.get_model("<mymodel>").default_dataloader` is the same as `kipoi.get_dataloader_factory("<mymodel>")`

#### Predict for a single batch
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


#### Re-train the model

```python
it_train = dl.batch_train_iter(batch_size=32)  # will yield tuples (inputs, targets) indefinitely

# Since we are using a Keras model, run:
model.model.fit_generator(it_train, steps_per_epoch=len(dl)//32, epochs=10)
```

-----------------------------------------

### Command-line interface - quick start

#### Show help

For the command line interface the help command should explain most functionality

```bash
kipoi -h
```

#### List all models

```bash
kipoi ls
```
#### Get information on how the required dataloader keyword arguments
```bash
kipoi info -i --source kipoi rbp_eclip/AARS
```

#### Run model prediction

```bash
cd ~/.kipoi/models/rbp_eclip/AARS/example_files

kipoi predict rbp_eclip/AARS \
  --dataloader_args='{'intervals_file': 'intervals.bed', 'fasta_file': 'hg38_chr22.fa', 'gtf_file': 'gencode.v24.annotation_chr22.gtf'}' \
  -o '/tmp/rbp_eclip__AARS.example_pred.tsv'

# check the results
head '/tmp/rbp_eclip__AARS.example_pred.tsv'
```

#### Test a model

Test whether a model is defined correctly and whether is execution using the example files is successful.

```bash
kipoi test ~/.kipoi/models/rbp_eclip/AARS/example_files
```

#### Install all model dependencies

```bash
kipoi env install rbp_eclip/AARS
```

#### Create a new conda environment for the model

```bash
kipoi env create rbp_eclip/AARS
source activate kipoi-rbp_eclip__UPF
```

#### List all Kipoi environments

```bash
kipoi env list
```

Use `source activate <env>` or `conda activate <env>` to activate the environment.


#### Score variants


```bash
kipoi veff score_variant rbp_eclip/AARS \
	--batch_size=16 \
	-v input.vcf \
	-o output.vcf
```

### R - quick start

See [tutorials/R-api](../../tutorials/R-api/).
