## Using Kipoi - Getting started

### Steps

#### 1. Install Kipoi

1. Install git-lfs
    - `conda install -c conda-forge git-lfs` (alternatively see <https://git-lfs.github.com/>)
2. Install kipoi
    - `pip install kipoi`
	  - if you wish to use variant effect prediction, run `pip install kipoi[vep]`

#### 2. Choose the model from <http://kipoi.org/models>

1. Navigate to <http://kipoi.org/models> and select a model of your interest
2. On model's page (`http://kipoi.org/models/<model>`), check the required arguments in the dataloader section

#### 3. Use the model

You can use the model from:

- Python
- Crommand-line interface
- R (via the [reticulate](https://github.com/rstudio/reticulate) package)


-----------------------------------------


### Python - quick start

See the ipython notebook [tutorials/python-api](../../tutorials/python-api/) for additional information. Here is a list of most useful python commands

```python
import kipoi
```

#### List all models

```python
kipoi.list_models()
```

#### Get the model

```python
model = kipoi.get_model("rbp_eclip/UPF1")
```

#### Access information about the model

```python
model.info # Information about the author:

model.default_dataloader # Access the default dataloader

model.model # Access the underlying Keras model
```

#### Test the model

```python
pred = model.pipeline.predict_example()
```

#### Get predictions for the raw files

```python
model.pipeline.predict({"dataloader_arg1": "inputs.csv"})
```

#### Setup the dataloader

```python
dl = model.default_dataloader(dataloader_arg1="inputs.csv", targets_file="targets.csv")
```

Note: `model.default_dataloader` is the same as `kipoi.get_dataloader_factory("rbp_eclip/UPF1")`

#### Predict for a single batch

```python
# Get the batch iterator
it = dl.batch_iter(batch_size=32)

# get a single batch
single_batch = next(it)

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

```bash
kipoi -h
```

#### List all models

```bash
kipoi ls
```

#### Run model prediction

```bash
cd ~/.kipoi/models/rbp_eclip/UPF1/example_files

kipoi predict rbp_eclip/UPF1 \
  --dataloader_args='{'intervals_file': 'intervals.bed', 'fasta_file': 'hg38_chr22.fa', 'gtf_file': 'gencode.v24.annotation_chr22.gtf'}' \
  -o '/tmp/rbp_eclip__UPF1.example_pred.tsv'

# check the results
head '/tmp/rbp_eclip__UPF1.example_pred.tsv'
```

#### Test a model

```bash
kipoi test ~/.kipoi/models/rbp_eclip/UPF1/example_files
```

#### Install all model dependencies

```bash
kipoi env install rbp_eclip/UPF1
```

#### Create a new conda environment for the model

```bash
kipoi env create rbp_eclip/UPF1
source activate kipoi-rbp_eclip__UPF
```

#### Score variants

```bash
kipoi postproc score_variant rbp_eclip/UPF1 \
	--batch_size=16 \
	-v input.vcf \
	-o output.vcf
```
