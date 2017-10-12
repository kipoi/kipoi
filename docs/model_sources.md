## `~/.kipoi/config.yaml`

`kipoi` package has a config file located at `~/.kipoi/config.yaml`. By default, it will look like this (without comments):

```yaml
model_sources:
  kipoi: # source name 
    type: git-lfs  # git repository with large file storage (lfs)
    remote_url: git@github.com:kipoi/models.git  # git remote
    local_path: /home/avsec/.kipoi/models/ # local storage path
```

### `model_sources`

`model_sources` defines all the places where kipoi will search for models and pull them to a local directory.
By default, it contains the model-zoo from `github.com/kipoi/models`. It is not a normal git repository,
since bigger files are stored with [git large file storage (git-lfs)](https://git-lfs.github.com/).
This repository will be stored locally under `local_path`.

Advantage of using `git-lfs` is that only the files tracked by `git` will be downloaded first. Larger files 
stored in `git-lfs` will be downloaded individually for each model upon request (say when a user invokes a `predict` command).

#### All possible model sources

In addition to the default `kipoi` source, you can define your own model sources in `~/.kipoi/config.yaml`. Available types are:

- `git-lfs` - As for `kipoi` model source
- `git` - Normal git repository, all the files will be downloaded.
- `local` - Local directory. 

Example:

```yaml
model_sources:
  kipoi:
    type: git-lfs
    remote_url: git@github.com:kipoi/models.git
    local_path: /home/avsec/.kipoi/models/
  my_local_models:
    type: local
    local_path: /data/mymodels/
  my_git_models:
    type: git
    remote_url: git@github.com:asd/other_models.git
    local_path: ~/.kipoi/other_models/
```	


## Notes

A particular model is defined by its source (key under `model_sources`, say `kipoi`) and the relative path of the desired model directory from the model source root (say `rbp/`).

A directory is considered a model if it contains a `model.yaml` file.
