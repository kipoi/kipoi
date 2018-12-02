## Command-line interface

For the command line interface the help command should explain most functionality

```bash
kipoi -h
```

### ls

List all models

```bash
kipoi ls
```

### info

Get information on how the required dataloader keyword arguments

```bash
kipoi info Basset
```

### predict

Run model prediction

```bash
kipoi get-example Basset -o example
kipoi predict Basset \
  --dataloader_args='{"intervals_file": "example/intervals_file", "fasta_file": "example/fasta_file"}' \
  -o '/tmp/Basset.example_pred.tsv'
# check the results
head '/tmp/Basset.example_pred.tsv'
```

You can add `--singularity` to the command in order to execute the command in the virtual environment.


### test

Test whether a model is defined correctly and whether is execution using the example files is successful.

```bash
kipoi test ~/.kipoi/models/Basset/example_files
```

### env 
#### install
Install model dependencies

```bash
kipoi env install Basset
```

#### create
Create a new conda environment for the model

```bash
kipoi env create Basset
source activate kipoi-Basset
```

#### list
List all environments

```bash
kipoi env list
```

Use `source activate <env>` or `conda activate <env>` to activate the environment.

See also <https://github.com/kipoi/examples> for more information.
