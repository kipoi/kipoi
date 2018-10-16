# Kipoi: Model zoo for genomics

<a href='https://circleci.com/gh/kipoi/kipoi'>
	<img alt='CircleCI' src='https://circleci.com/gh/kipoi/kipoi.svg?style=svg' style="max-height:20px;width:auto">
</a>
<a href=https://coveralls.io/github/kipoi/kipoi?branch=master>
	<img alt='Coverage status' src=https://coveralls.io/repos/github/kipoi/kipoi/badge.svg?branch=master style="max-height:20px;width:auto;">
</a>
<a href=https://gitter.im/kipoi>
	<img alt='Gitter' src=https://badges.gitter.im/kipoi/kipoi.svg style="max-height:20px;width:auto;">
</a>

This repository implements a python package and a command-line interface (CLI) to access and use models from Kipoi-compatible model zoo's.

<img src="http://kipoi.org/static/img/fig1_v8_hires.png" width=600>


## Links

- [kipoi.org](http://kipoi.org) - Main website
- [kipoi.org/docs](http://kipoi.org/docs) - Documentation
- [github.com/kipoi/models](https://github.com/kipoi/models) - Model zoo for genomics maintained by the Kipoi team
- [bioarxiv preprint](https://doi.org/10.1101/375345) - Kipoi: accelerating the community exchange and reuse of predictive models for genomics
  
## Installation

### 1. Install miniconda/anaconda

Kipoi requires [conda](https://conda.io/) to manage model dependencies.
Make sure you have either anaconda ([download page](https://conda.io/miniconda.html)) or miniconda ([download page](https://www.anaconda.com/download/)) installed. If you are using OSX, see [Installing python on OSX](http://kipoi.org/docs/using/04_Installing_on_OSX/).

### 2. Install Git LFS

For downloading models, Kipoi uses git and [Git Large File Storage](https://git-lfs.github.com/) (LFS). See how to install git [here](https://www.atlassian.com/git/tutorials/install-git). To install git-lfs on Ubuntu, run:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install -y git git-lfs
git-lfs install
```

Alternatively, you can install git-lfs through conda:

```bash
conda install -c conda-forge git-lfs && git lfs install
```

### 3. Install Kipoi

Next, install Kipoi using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install kipoi
```

## Quick start

The following diagram gives a short overview over Kipoi's workflow:
<img src="http://kipoi.org/docs/img/kipoi-workflow.png" height=400>

If you want to check which models are available in Kipoi you can use the [website](http://kipoi.org/groups/), where you can also see example commands for how to use the models on the CLI, python and R.
Alternatively you can run `kipoi ls` in the command line or in python:
```python
import kipoi

kipoi.list_models()
```

Once a model has been selected (here: `rbp_eclip/UPF1`), the model environments have to be installed. To do so use one of the `kipoi env` commands.
For example to install an environment for model `rbp_eclip/UPF1`:

```bash
kipoi env create rbp_eclip/UPF1
```

---
** Aside: models versus model groups**: 

>A Kipoi `model` is a path to a directory containing a `model.yaml` file.  This file specifies the underlying model, data loader, and other model attributes.  
>If instead you provide a path to a model *group* (e.g  "rbp_eclip" or "lsgkm-SVM/Tfbs/Ap2alpha/"), rather than one model (e.g "rbp_eclip/UPF1" or "lsgkm-SVM/Tfbs/Ap2alpha/Helas3/Sydh_Std"), or any other directory without a `model.yaml` file, a `ValueError` will be thrown.

---

If you are working on a machine that has GPUs, you will want to add the `--gpu` flag to the command. And if you want to make use of the `kipoi-veff` plug-in then add `--vep`. For more options please run `kipoi env create --help`.


The command will tell you how the execution environment for the model is called, e.g.:

```INFO [kipoi.cli.env] Environment name: kipoi-rbp_eclip__UPF1```

Before using a model in any way, make sure that you have activated its environment, e.g.: prior to executing `kipoi` or `python` or `R` in the attempt to use Kipoi with the model. To activate the model environment run for example:
```bash
source activate kipoi-rbp_eclip__UPF1
``` 

### Command-line
Once the model environment is activated Kipoi's API can be accessed from the commandline using:

```
$ kipoi
usage: kipoi <command> [-h] ...

    # Kipoi model-zoo command line tool. Available sub-commands:
    # - using models:
    ls               List all the available models
    list_plugins     List all the available plugins
    info             Print dataloader keyword argument info
    predict          Run the model prediction
    pull             Download the directory associated with the model
    preproc          Run the dataloader and save the results to an hdf5 array
    env              Tools for managing Kipoi conda environments

    # - contributing models:
    init             Initialize a new Kipoi model
    test             Runs a set of unit-tests for the model
    test-source      Runs a set of unit-tests for many/all models in a source
    
    # - plugin commands:
    veff             Variant effect prediction
    interpret        Model interpretation using feature importance scores like ISM, grad*input or DeepLIFT.
```

Explore the CLI usage by running `kipoi <command> -h`. Also, see [docs/using/getting started cli](http://kipoi.org/docs/using/01_Getting_started/#command-line-interface-quick-start) for more information.

### Python
Once the model environment is activated (`source activate kipoi-rbp_eclip__UPF1`) Kipoi's python API can be used to:

The following commands give a short overview. For details please take a look at the python API documentation.
Load the model from model the source:
```python
import kipoi
model = kipoi.get_model("rbp_eclip/UPF1") # load the model
```

To get model predictions using the dataloader we can run:
```python
model.pipeline.predict(dict(fasta_file="hg19.fa",
                            intervals_file="intervals.bed",
                            gtf_file="gencode.v24.annotation_chr22.gtf"))
# runs: raw files -[dataloader]-> numpy arrays -[model]-> predictions 
```

To predict the values of a model input `x`, which for example was generated by dataloader, we can use:
 
```python
model.predict_on_batch(x) # implemented by all the models regardless of the framework
```

Here `x` has to be a `numpy.ndarray` or a list or a dict of a `numpy.ndarray`, depending on the model requirements, for details please see the documentation of the API or of `datalaoder.yaml` and `model.yaml`.



For more information see: [notebooks/python-api.ipynb](notebooks/python-api.ipynb) and [docs/using getting started](http://kipoi.org/docs/using/01_Getting_started/)


### Configure Kipoi in `.kipoi/config.yaml`

You can add your own (private) model sources. See [docs/using/03_Model_sources/](http://kipoi.org/docs/using/03_Model_sources/).

### Contributing models

See [docs/contributing getting started](http://kipoi.org/docs/contributing/01_Getting_started/) and [docs/tutorials/contributing/models](http://kipoi.org/docs/tutorials/contributing_models/) for more information.

## Plugins
Kipoi supports plug-ins which are published as additional python packages. Two plug-ins that are available are:

### [kipoi_veff](https://github.com/kipoi/kipoi-veff)

Variant effect prediction plugin compatible with (DNA) sequence based models. It allows to annotate a vcf file using model predictions for the reference and alternative alleles. The output is written to a new VCF file. For more information see <https://kipoi.org/veff-docs/>.

```bash
pip install kipoi_veff
```


### [kipoi_interpret](https://github.com/kipoi/kipoi-interpret)

Model interpretation plugin for Kipoi. Allows to use feature importance scores like in-silico mutagenesis (ISM), saliency maps or DeepLift with a wide range of Kipoi models. [example notebook](https://github.com/kipoi/kipoi-interpret/blob/master/notebooks/1-DNA-seq-model-example.ipynb)

```bash
pip install kipoi_interpret
```

## Documentation

Documentation can be found here: [kipoi.org/docs](http://kipoi.org/docs)

## Citing Kipoi

If you use Kipoi for your research, please cite the publication of the model you are using (see model's `cite_as` entry) and our Bioarxiv preprint: https://doi.org/10.1101/375345.

```bibtex
@article {kipoi,
	author = {Avsec, Ziga and Kreuzhuber, Roman and Israeli, Johnny and Xu, Nancy and Cheng, Jun and Shrikumar, Avanti and Banerjee, Abhimanyu and Kim, Daniel S and Urban, Lara and Kundaje, Anshul and Stegle, Oliver and Gagneur, Julien},
	title = {Kipoi: accelerating the community exchange and reuse of predictive models for genomics},
	year = {2018},
	doi = {10.1101/375345},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2018/07/24/375345},
	eprint = {https://www.biorxiv.org/content/early/2018/07/24/375345.full.pdf},
	journal = {bioRxiv}
}
```

## Development

If you want to help with the development of Kipoi, you are more than welcome to join in! 

For the local setup for development, you should install `kipoi` using:

```bash
conda install pytorch-cpu
pip install -e '.[develop]'
```

This will install some additional packages like `pytest`. You can test the package by running `py.test`. 

If you wish to run tests in parallel, run `py.test -n 6`.
