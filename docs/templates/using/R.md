## R

You can use Kipoi from R via the [reticulate](https://github.com/rstudio/reticulate) package. For a more complete example, see <https://github.com/kipoi/kipoi/blob/master/notebooks/R-api.ipynb>.

### Installation

1. Install Kipoi. See [how](https://kipoi.org/docs/)
1. Install R
1. Install the `reticulate` package. From R, run: `install.packages("reticulate")`
  
Make sure reticulate is using python from the miniconda/anaconda installation (same as Kipoi):

```R
library(reticulate)
reticulate::py_config()
```

### Usage

Use a specific conda environment

```R
library(reticulate)
reticulate::use_condaenv("kipoi-Basset)
```

or install the dependencies from R:

```R
kipoi$install_model_requirements("Basset")
```

Get the model:

```R
kipoi <- import('kipoi')
model <- kipoi$get_model('Basset')
```

Make a prediction for example files

```R
predictions <- model$pipeline$predict_example()
```

Use dataloader and model separately

```R
# Get the dataloader
setwd('~/.kipoi/models/Basset')
dl <- model$default_dataloader(intervals_file='example_files/intervals.bed', fasta_file='example_files/hg38_chr22.fa')
# get a batch iterator
it <- dl$batch_iter(batch_size=4)
# predict for a batch
batch <- iter_next(it)
model$predict_on_batch(batch$inputs)
```

Make predictions for custom files directly:

```R
pred <- model$pipeline$predict(dl_kwargs, batch_size=4)
```
