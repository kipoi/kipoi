This docs shows how our API should ideally look like.

## CLI definition

Main arguments

- `<model_dir>`: can be a git repository, local file path, remote file path or a short model string.

### Pre-process

```
usage: kipoi preproc [-h] [--source {kipoi,dir}]
                        [--extractor_args EXTRACTOR_ARGS]
                        [--batch_size BATCH_SIZE] [-i] [-o OUTPUT]
                        model

Run the extractor and save the output to an hdf5 file.

positional arguments:
  model                 Model name.

optional arguments:
  -h, --help            show this help message and exit
  --source {kipoi,dir}  Model source to use. Specified in ~/.kipoi/config.yaml
                        under model_sources
  --extractor_args EXTRACTOR_ARGS
                        Extractor arguments either as a json string:'{"arg1":
                        1} or as a file path to a json file
  --batch_size BATCH_SIZE
                        Batch size to use in prediction
  -i, --install-req     Install required packages from requirements.txt
  -o OUTPUT, --output OUTPUT
                        Output hdf5 file
```


### Predict

```
usage: kipoi predict [-h] [--source {kipoi,dir}]
                        [--extractor_args EXTRACTOR_ARGS] [-f {tsv,bed,hdf5}]
                        [--batch_size BATCH_SIZE] [-n NUM_WORKERS] [-i] [-k]
                        [-o OUTPUT]
                        model

Run the model prediction.

positional arguments:
  model                 Model name.

optional arguments:
  -h, --help            show this help message and exit
  --source {kipoi,dir}  Model source to use. Specified in ~/.kipoi/config.yaml
                        under model_sources
  --extractor_args EXTRACTOR_ARGS
                        Extractor arguments either as a json string:'{"arg1":
                        1} or as a file path to a json file
  -f {tsv,bed,hdf5}, --file_format {tsv,bed,hdf5}
                        File format.
  --batch_size BATCH_SIZE
                        Batch size to use in prediction
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        Number of parallel workers for loading the dataset
  -i, --install-req     Install required packages from requirements.txt
  -k, --keep_inputs     Keep the inputs in the output file. Only compatible
                        with hdf5 file format
  -o OUTPUT, --output OUTPUT
                        Output hdf5 file
```

### Test the model

Runs a set of unit-tests for the model

```
usage: kipoi test [-h] [--source {kipoi,dir}] [--batch_size BATCH_SIZE]
                     [-i]
                     model

script to test model zoo submissions

positional arguments:
  model                 Model name.

optional arguments:
  -h, --help            show this help message and exit
  --source {kipoi,dir}  Model source to use. Specified in ~/.kipoi/config.yaml
                        under model_sources
  --batch_size BATCH_SIZE
                        Batch size to use in prediction
  -i, --install-req     Install required packages from requirements.txt
```


### Score variants

**Not implemented**

```
model_zoo score_variants <model> <preprocessor inputs...> <vcf file> -o <output>
```

### Pull the model

Downloads the directory associated with the model

```
$ kipoi pull -h
usage: kipoi pull [-h] [--source {kipoi}] model

Downloads the directory associated with the model.

positional arguments:
  model                 Model name.

optional arguments:
  -h, --help            show this help message and exit
  --source {kipoi}      Model source to use. Specified in
                        ~/.kipoi/config.yaml under model_sources
```
