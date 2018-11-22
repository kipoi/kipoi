## Contributing models - Getting started

Kipoi stores models (descriptions, parameter files, dataloader code, ...) as folders in the 
[kipoi/models](https://github.com/kipoi/models) github repository. The minimum requirement for a model is that a 
[`model.yaml`](./02_Writing_model.yaml.md) file is available in the model folder, which defines the type of the model, 
file paths / URLs, the dataloader, description, software dependencies, etc.

All files necessary for a model to be executed have to be published on [zenodo](https://zenodo.org/) or 
[figshare](https://figshare.com/) to insure functionality and versioning of models.


One key element for a model to be contributed to Kipoi is its dataloader. The main aim of a dataloader is to generate 
batches of data with which a model can be run. Its inputs should be files in the most common formats of the respective 
field, such as .bed and .fasta files for genomic sequences and regions. 

#### Pre-defined datalaoders
To simplify the process of contributing models to Kipoi we have created [kipoiseq](https://github.com/kipoi/kipoiseq), 
a repository that offers pre-defined dataloaders for common applications.

If you can use one of the dataloaders in kipoiseq for you model then the Kipoi model will consist solely in a folder
and one `model.yaml` file inside it:

```
MyModel
└── model.yaml         # describes the model
```
 

#### Non-default dataloaders

If the model you want to contribute requires different input from what is available out of the box in 
[kipoiseq](https://github.com/kipoi/kipoiseq), you are encouraged to use the tested tools available in 
[kipoiseq](https://github.com/kipoi/kipoiseq) to write your own [dataloader](./04_Writing_dataloader.py.md) and its 
companion [yaml-file](./03_Writing_dataloader.yaml.md). If you do so you should keep the standard Kipoi way of
defining models with all the files and their assigned places:
```
MyModel
├── dataloader.py     # implements the dataloader
├── dataloader.yaml   # describes the dataloader
└── model.yaml         # describes the model
```

### Required steps for contribution

#### Using pre-defined datalaoders

If the dataloaders offered in [kipoiseq](https://github.com/kipoi/kipoiseq) are what your model needs then submitting 
a new model can even be done online using github or locally as explained 
[here](#setting-up-kipoi-for-model-contribution).
 
###### Contribute model online
You can contribute a model online on github by clicking `Create new file` in the 
[models repository](https://github.com/kipoi/models). The filename would then be `MyModel/model.yaml`. The 
name of folder (here: `MyModel`) which contains the [`model.yaml`](./02_Writing_model.yaml.md) 
file. You can then select `Create a new branch for this commit and start a pull request` to attempt adding your model
to Kipoi. If you want to test your model locally you have to make sure that kipoi and git are installed locally. 
You can test your models as described [here](#how-to-test-the-model).


#### Defining your own dataloader
If the pre-defined dataloaders don't cover your use-case you will have to define your own. You therefore have to set up 
kipoi as described [here](#setting-up-kipoi-for-model-contribution).

#### Setting up Kipoi for model contribution

Here is a list of steps required to contribute a model to [kipoi/models](https://github.com/kipoi/models):

##### 1. Install Kipoi

1. Install git
    - There are many ways to do so and on many systems git is already installed. If not you can follow 
    [this](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) guide.
2. Install kipoi
    - `pip install kipoi`
3. Run `kipoi ls` (this will checkout the `kipoi/models` repo to `~/.kipoi/models`)

##### 2. Add the model

0. `cd ~/.kipoi/models`
1. [Write the model](#how-to-write-the-model): Create a new folder `<my new model>` containing all the required files. 
 The required files can be created by doing one of the following three options:
    - Option 1: Copy the existing model: `cp -R <existing model> <my new model>`, edit/replace/add the copied files 
    until they fit your new model.
	- Option 2: Run `kipoi init`, answer the questions, edit/replace the created files until they fit your new model.
	- Option 3: `mkdir <my new model>` & write all the files from scratch
2. [Test the model](#how-to-test-the-model)
    - Step 1: `kipoi test ~/.kipoi/models/my_new_model`
	- Step 2: `kipoi test-source kipoi --all -k my_new_model`

##### 3. Submit the pull-request

###### Option 1: Fork the repository 

0. Make sure you have all the recent changes locally
    - `cd ~/.kipoi/models`
    - `export GIT_LFS_SKIP_SMUDGE=1 && git pull` - pulls all the changes but doesn't download the files tracked by git-lfs.
1. Commit your changes
	- `git add my_new_model/`
	- `git commit -m "Added <my new model>"`
1. [Fork](https://guides.github.com/activities/forking/) the <https://github.com/kipoi/models> repo on github (click on the Fork button)
1. Add your fork as a git remote to `~/.kipoi/models`
    - `git remote add fork https://github.com/<username>/models.git`
1. Push to your fork
    - `git push fork master`
1. Submit a pull-request
    - click the [New pull request](https://help.github.com/articles/creating-a-pull-request/) button on your github fork - `https://github.com/<username>/models>`

###### Option 2: Create a new branch on kipoi/models

If you wish to contribute models more frequently, please [join the team](https://github.com/kipoi/models/issues/55). 
You will be added to the Kipoi organization. This will allow you to push to branches of the `kipoi/models` github repo directly.

1. Make sure you have all the recent changes locally
    - `cd ~/.kipoi/models`
    - `export GIT_LFS_SKIP_SMUDGE=1 && git pull` - pulls all the changes but doesn't download the files tracked by git-lfs.
1. Create a new branch in `~/.kipoi/models`
    - `git stash` - this will store/stash all local changes in [git stash](https://git-scm.com/book/en/v1/Git-Tools-Stashing)
    - `git checkout -b my_new_model` - create a new branch
    - `git stash pop` - get the stashed files back
2. Commit changes
    - `git add my_new_model/`
    - `git commit -m "Added <my new model>"`
3. Push changes to `my_new_model` branch
    - `git push -u origin my_new_model`
4. Submit a pull-request
    - click the [New pull request](https://help.github.com/articles/creating-a-pull-request/) button on `my_new_model` branch of repo <https://github.com/kipoi/models>.

Rest of this document will go more into the details about steps writing the model and testing the model.

#### How to write the model

Best place to start figuring out which files you need to contribute is to look at some of the existing models. Explore the <https://github.com/kipoi/models> repository and see if there are any models similar to yours (in terms of the dependencies, framework, input-output data modalities). See [tutorials/contributing_models](../../tutorials/contributing_models) for a step-by-step procedure for contributing models.

In terms of what to include in your model: The information in these pages here are the minimum requirement. The more
information you can share with other users the better! If you have converted the model from using a script, please add that.
If you have additional test and validation scripts that you wrote while verifying the Kipoi model, etc. , please add them.
You will make future users happy.

Hint: If you want to take a look at a specific model that is already in the zoo, but instead of the content of the model files there is just a hash entry, then use `kipoi pull <model_name>` to download the model data.

##### Option #1: Copy existing model

Once you have found the closest match, simply copy the directory and start editing/replacing the files. Edit the files in this order:

- model.yaml
- dataloader.yaml (If you wrote your own dataloader)
- dataloader.py (If you wrote your own dataloader)
- LICENSE

##### Option #2: Use `kipoi init`

Alternatively, you can use `kipoi init` instead of copying the existing model:

```bash
cd ~/.kipoi/models && kipoi init
```

This will ask you a few questions and create a new model folder.

```bash
$ kipoi init
INFO [kipoi.cli.main] Initializing a new Kipoi model

Please answer the questions below. Defaults are shown in square brackets.

You might find the following links useful: 
- (model_type) https://github.com/kipoi/kipoi/blob/master/docs/writing_models.md
- (dataloader_type) https://github.com/kipoi/kipoi/blob/master/docs/writing_dataloaders.md
--------------------------------------------

model_name [my_model]: my_new_model
author_name [Your name]: Ziga Avsec
author_github [Your github username]: avsecz
author_email [Your email(optional)]: 
model_doc [Model description]: Model predicting iris species
Select model_license:
1 - MIT
2 - BSD
3 - ISCL
4 - Apache Software License 2.0
5 - Not open source
Choose from 1, 2, 3, 4, 5 [1]:  
Select model_type:
1 - keras
2 - custom
3 - sklearn
Choose from 1, 2, 3 [1]: 1
Select model_input_type:
1 - np.array
2 - list of np.arrays
3 - dict of np.arrays
Choose from 1, 2, 3 [1]: 2
Select model_output_type:
1 - np.array
2 - list of np.arrays
3 - dict of np.arrays
Choose from 1, 2, 3 [1]: 3
Select dataloader_type:
1 - Dataset
2 - PreloadedDataset
3 - BatchDataset
4 - SampleIterator
5 - SampleGenerator
6 - BatchIterator
7 - BatchGenerator
Choose from 1, 2, 3, 4, 5, 6, 7 [1]: 1
--------------------------------------------
INFO [kipoi.cli.main] Done!
Created the following folder into the current working directory: my_new_model
```

The created folder contains a model and a dataloader for predicting the Iris species. 
You will now have to [edit the model.yaml](./02_Writing_model.yaml.md) and to 
[edit the dataloader.yaml](./03_Writing_dataloader.yaml.md) files according to your model.
 You can check whether you have succeeded and your model is setup correctly with the commands below.

#### How to test the model
Be aware that the test functions will only check whether the definition side of things 
(model.yaml, dataloader.yaml, syntax errors, etc.) is setup correctly, you will have to validate yourself whether 
the outputs created by using the predict function produce the desired model output!

##### Step 1: Run `kipoi test ~/.kipoi/models/my_new_model`

<!-- To make sure this work as you expect, test your model by running: -->

<!-- ```bash -->
<!-- kipoi test ~/.kipoi/models/my_new_model -->
<!-- ``` -->

This checks the yaml files and runs `kipoi predict` for the example files (specified in `dataloader.yaml > args > my_arg > example`). Once this command returns no errors or warnings proceed to the next step.

##### Step 2: Run `kipoi test-source kipoi --all -k my_new_model`

<!-- To also test that the conda environment for the model can be installed correctly, run: -->

<!-- ```bash -->
<!-- kipoi test-source kipoi --all -k my_new_model -->
<!-- ``` -->

This will run `kipoi test` in a new conda environment with dependencies specified in `model.yaml` and `dataloader.yaml`.

#### Removing or updating models

To remove, rename or update an existing model, send a pull-request (as when contributing models, see [3. Submit the pull-request](#3-submit-the-pull-request)).
