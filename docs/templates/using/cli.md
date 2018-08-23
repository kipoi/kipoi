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
kipoi info -i --source kipoi rbp_eclip/UPF1
```

### predict
Run model prediction

```bash
cd ~/.kipoi/models/rbp_eclip/UPF1/example_files

kipoi predict rbp_eclip/UPF1 \
  --dataloader_args='{'intervals_file': 'intervals.bed', 'fasta_file': 'hg38_chr22.fa', 'gtf_file': 'gencode.v24.annotation_chr22.gtf'}' \
  -o '/tmp/rbp_eclip__UPF1.example_pred.tsv'

# check the results
head '/tmp/rbp_eclip__UPF1.example_pred.tsv'
```

### test

Test whether a model is defined correctly and whether is execution using the example files is successful.

```bash
kipoi test ~/.kipoi/models/rbp_eclip/UPF1/example_files
```

### env 
#### install
Install model dependencies

```bash
kipoi env install rbp_eclip/UPF1
```

#### create
Create a new conda environment for the model

```bash
kipoi env create rbp_eclip/UPF1
source activate kipoi-rbp_eclip__UPF
```

#### list
List all environments

```bash
kipoi env list
```

Use `source activate <env>` or `conda activate <env>` to activate the environment.
