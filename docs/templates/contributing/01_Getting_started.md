## Contributing models - Getting started

Kipoi stores models (descriptions, parameter files, dataloader code, ...) as folders in the [kipoi/models](https://github.com/kipoi/models) github repository. Files residing in folders with a suffix of `_files` are tracked via Git Large File Storage (LFS). New models are added by simply submitting a pull-request to <https://github.com/kipoi/models>.

### Required steps

Here is a list of steps required to contribute a model to [kipoi/models](https://github.com/kipoi/models):

#### 1. Install Kipoi

1. Install git-lfs
    - `conda install -c conda-forge git-lfs` (alternatively see <https://git-lfs.github.com/>)
2. Install kipoi
    - `pip install kipoi`
3. Run `kipoi ls` (this will checkout the `kipoi/models` repo to `~/.kipoi/models`)

#### 2. Add the model

0. `cd ~/.kipoi/models`
1. [Write the model](#how-to-write-the-model): Create a new folder `<my new model>` containing all the required files
    - Option 1: Copy the existing model: `cp -R <existing model> <my new model>` & edit the copied files
	- Option 2: Run `kipoi init`, answer the questions & edit the created files
	- Option 3: `mkdir <my new model>` & write all the files from scratch
2. [Test the model](#how-to-test-the-model)
    - Step 1: `kipoi test ~/.kipoi/models/my_new_model`
	- Step 2: `kipoi test-source kipoi --all -k my_new_model`
3. Commit your changes
    - `cd ~/.kipoi/models && git commit -m "Added <my new model>"`

#### 3. Submit the pull-request

1. [Fork](https://guides.github.com/activities/forking/) the <https://github.com/kipoi/models> repo on github (click on the Fork button)
2. Add your fork as a git remote to `~/.kipoi/models`
    - `cd ~/.kipoi/models && git remote add fork https://github.com/<username>/models.git`
3. Push to your fork
    - `git push fork master`
4. Submit a pull-request (click the [New pull request](https://help.github.com/articles/creating-a-pull-request/) button on your github fork - `https://github.com/<username>/models>`)


Rest of this document will go more into the details about steps writing the model and testing the model.

### How to write the model

Best place to start figuring out which files you need to contribute is to look at some of the existing models. Explore the <https://github.com/kipoi/models> repository and see if there are any models similar to yours (in terms of the dependencies, framework, input-output data modalities). See [tutorials/contributing_models](../tutorials/contributing_models) for a step-by-step procedure for contributing models.

#### Option #1: Copy existing model

Once you have found the closest match, simply copy the directory and start editing/replacing the files. Edit the files in this order:

- model.yaml
- dataloader.yaml
- dataloader.py
- overwrite files in `model_files/`
- `example_files/`
- LICENSE

#### Option #2: Use `kipoi init`

Alternatively, you can use `kipoi init` instead of copying the existing model:

```bash
cd ~/.kipoi/models && kipoi init
```

This will ask you a few questions and create a new model folder.

```bash
$ kipoi init
INFO [kipoi.cli.main] Initializing a new Kipoi model

Please answer the questions bellow. Defaults are shown in square brackets.

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

The created folder contains a model and a dataloader for predicting the Iris species. You can check if the tests pass for this newly created folder by running the comands bellow.

### How to test the model

#### Step 1: Run `kipoi test ~/.kipoi/models/my_new_model`

<!-- To make sure this work as you expect, test your model by running: -->

<!-- ```bash -->
<!-- kipoi test ~/.kipoi/models/my_new_model -->
<!-- ``` -->

This checks the yaml files and runs `kipoi predict` for the example files (specified in `dataloader.yaml > args > my_arg > example`). Once this command returns no errors or warnings proceed to the next step.

#### Step 2: Run `kipoi test-source kipoi --all -k my_new_model`

<!-- To also test that the conda environment for the model can be installed correctly, run: -->

<!-- ```bash -->
<!-- kipoi test-source kipoi --all -k my_new_model -->
<!-- ``` -->

This will run `kipoi test` in a new conda environment with dependencies specified in `model.yaml` and `dataloader.yaml`.
