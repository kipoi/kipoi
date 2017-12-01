# Testing contributed models

## 0. Accept the review

Assign yourself as the pull-request reviewer on github.

## 1. Review code

Check the contributed code. Does it look right?

## 2. Locally checkout the pull-request

Move to the local kipoi/models folder at: `~/.kipoi/models`.

```bash
cd ~/.kipoi/models
```

git pull the PR code: 

- Folow: https://help.github.com/articles/checking-out-pull-requests-locally/
  - Click to `view command line instructions` on the pull request page

## 3. Create a new conda environment

```bash
conda create --name kipoi-<model> python=3.5 numpy h5py pandas
source activate kipoi-<model>
```

## 4. Install kipoi

```bash
pip install <local kipoi path>
```

## 5. Run tests 

```bash
kipoi test -i <model> 
```

If everyhing runs through without any warnings or errors, merge the pull-request. If not, report the issues in the PR comments.
