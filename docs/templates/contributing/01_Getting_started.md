## Contributing models - Getting started

Kipoi stores models (descriptions, parameter files, dataloader code, ...) as folders in the 
[kipoi/models](https://github.com/kipoi/models) github repository. The minimum requirement for a model is that a 
[`model.yaml`](./02_Writing_model.yaml.md) file is available in the model folder, which defines the type of the model, 
file paths / URLs, the dataloader, description, software dependencies, etc.

We have compiled some of the standard use-cases of model contribution here:

*the selection container div goes here*

1.	Choose model input
	1.	DNA-sequence based model
	2.	DNA-sequence based model with additional tracks
	3.	Splicing model
2.	Choose model type
        1.	Keras
        2.	TensorFlow
        3.	PyTorch
        4.	Other model
3.	Choose model group type 
        1.	Single model
                1. 	Divergent421
        2.	Set of highly similar models
                1.	Ex: CpGenie
        3.	Set of different models
                1.	Ex: FactorNet

<span class="cond forking">
Before you start, make sure you have installed `kipoi`. Now that you are all set change to your kipoi models directory,
in the default configuration this will be `cd ~/.kipoi/models`. Do all the following steps in this folder.
</span>

<span class="cond dna-ktp-single">


For this example let's assume the model you want to submit is called `MyModel`. To submit your model with DNA seqeunce
input you will have to:

- Create a new local folder named after your model, e.g.: `mkdir MyModel`
- In the `MyModel` folder you will have to crate a `model.yaml` file:
    The `model.yaml` files acts as a configuration file for Kipoi. For an example take a look at 
    [Divergent421/model.yaml](https://github.com/kipoi/models/blob/master/Divergent421/model.yaml).
</span>
<span class="cond dna-ktp-setSim">


For this example let's assume you have trained one model architecture on multiple similar datasets and can use the 
 same preprocessing code for all models. Let's assume you want to call the 
model-group `MyModel`. To submit your model with DNA seqeunce input you will have to:

- Create a new local folder named after your model, e.g.: `mkdir MyModel`
- In the `MyModel` folder you will have to crate a `model-template.yaml` file:
    The `model-template.yaml` files acts as a configuration file for Kipoi. For an example take a look at 
    [CpGenie/model-template.yaml](https://github.com/kipoi/models/blob/master/CpGenie/model-template.yaml).
- As you can see instead of putting urls and parameters directly in the `.yaml` file you need to put 
`{{ parameter_name }}` in the yaml file. The values are then automatically loaded from a `tab`-delimited
file called `models.tsv` that you also have to provide. For the previous example this would be: 
[CpGenie/models.tsv](https://github.com/kipoi/models/blob/master/CpGenie/models.tsv). Using kipoi those models are
then accessible by the model group name and the model name defined in the `models.tsv`. Model names may contain `/`s.


</span>
<span class="cond dna-ktp-setDiff">


If you have trained multiple models that logically belong into one model-group as they are similar in function, but 
they individually require different preprocessing code then you are right here.

- Create a new local folder named after your model, e.g.: `mkdir MyModel` and within this folder create a folder
structure so that every individual trained model has its own folder. Every folder that contains a `model.yaml` is then
intrepreted as an individual model by Kipoi.
- To make this clearer take a look at how `FactorNet` is structured: 
[FactorNet](https://github.com/kipoi/models/tree/master/FactorNet). If you have files that are re-used in multiple 
models you can use symbolic links (`ln -s`) relative within the folder structure of your model group.
- In the following the contents of the `model.yaml` files will be explained:


</span>
<span class="cond keras">


- In the model definition yaml file you see the `defined_as` keyword: Since your model is a Keras model set it to
 `kipoi.model.KerasModel`.
- In the model definition yaml file you see the `args` keyword, which can be set the following way: 
[KerasModel definition](../02_Writing_model.yaml/#kipoimodelkerasmodel-models)


</span>
<span class="cond tensorflow">


- In the model definition yaml file you see the `defined_as` keyword: Since your model is a Keras model set it to
 `kipoi.model.TensorFlowModel`.
- In the model definition yaml file you see the `args` keyword, which can be set the following way: 
[TensorFlowModel definition](../02_Writing_model.yaml/#kipoimodeltensorflowmodel-models)


</span>
<span class="cond pytorch">


- In the model definition yaml file you see the `defined_as` keyword: Since your model is a Keras model set it to
 `kipoi.model.PyTorchModel`.
- In the model definition yaml file you see the `args` keyword, which can be set the following way: 
[PyTorchModel definition](../02_Writing_model.yaml/#kipoimodelpytorchmodel-models)


</span>
<span class="cond dna-ktp-single">

As you have seen in the presented example and in the model definition links it is necessary that prior to model 
contribution you have published all model files (except for python scripts and other configuration files) on 
[zenodo](https://zenodo.org/) or [figshare](https://figshare.com/) to ensure functionality and versioning of models.

If you want to test your model(s) locally before publishing them on [zenodo](https://zenodo.org/) or
 [figshare](https://figshare.com/) you can replace the pair of `url` and `md5` tags in the model definition yaml by the 
local path on your filesystem, e.g.:
```yaml
args:
    arch: path/to/my/arch.json
```
But keep in mind that local paths are only good for testing and for models that you want to keep only locally.

</span>
<span class="cond dna-ktp-single dna-ktp-setSim dna-ktp-setDiff">

For models with DNA sequence input (1-hot encoded or string input) it is recommended to use dataloaders from the 
[kipoiseq](https://github.com/kipoi/kipoiseq) package. Notice that you can specify the requirements regarding of
the model input format in the `dataloader` model definition yaml: 
    
</span>
<span class="cond dna-ktp-single">

```yaml
default_dataloader:
    defined_as: kipoiseq.dataloaders.SeqIntervalDl
    default_args:
        auto_resize_len: 1000
```
    
</span>
<span class="cond dna-ktp-setSim">

```yaml
default_dataloader:
  defined_as: kipoiseq.dataloaders.SeqIntervalDl
  default_args:
    auto_resize_len: 1001
    alphabet_axis: 0
    dummy_axis: 1
```

</span>
<span class="cond dna-ktp-setDiff">

--> Have to take a look at the other dataloaders above because FactorNet is not a good example...
    
</span>

<span class="cond model-deps">

Don't forget to set the software requirements correctly. This happens in the `dependencies` section of the model 
`.yaml` file. As you can see in the example the dependencies are split by `conda` and `pip`. Ideally you define the 
ranges of the versions of packages your model supports - otherwise it may fail at some point in future. If you need 
to specify a conda channel use the `<channel>::<package>` notation for conda dependencies.

</span>

<span class="cond forking">

- Make sure your model repository is up to date: 
    - `git pull`
- Commit your changes
	- `git add MyModel/`
	- `git commit -m "Added <MyModel>"`
- [Fork](https://guides.github.com/activities/forking/) the <https://github.com/kipoi/models> repo on github (click on 
the Fork button)
- Add your fork as a git remote to `~/.kipoi/models`
    - `git remote add fork https://github.com/<username>/models.git`
- Push to your fork
    - `git push fork master`
- Submit a pull-request
    - On github click the [New pull request](https://help.github.com/articles/creating-a-pull-request/) button on your 
    github fork - `https://github.com/<username>/models>`

</span>
