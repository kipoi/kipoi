# dataloader.yaml

<aside class="warning">
Before writing a dataloader yourself please check whether the same functionality can be achieved using a ready-made 
dataloader in [kipoiseq](https://github.com/kipoi/kipoiseq).
</aside>

The dataloader.yaml file describes how a dataloader for a certain model can be created and how it has to be set up. 
A model without functional dataloader is as bad as a model that doesn't work, so the correct setup of the 
dataloader.yaml is essential for the use of a model in the zoo. Make sure you have read 
[Writing dataloader.py](./04_Writing_dataloader.py.md).

To help understand the syntax of YAML please take a look at: 
[YAML Syntax Basics](http://docs.ansible.com/ansible/latest/YAMLSyntax.html#yaml-basics)

Here is an example `dataloader.yaml`:

```yaml
defined_as: dataloader.MyDataset  # We need to implement MyDataset class inheriting from kipoi.data.Dataset in dataloader.py
args:
    features_file:
        # descr: > allows multi-line fields
        doc: >
          Csv file of the Iris Plants Database from
          http://archive.ics.uci.edu/ml/datasets/Iris features.
        type: str
        example: 
            url: https://zenodo.org/path/to/example_files/features.csv  # example file
            md5: 7a6s5d76as5d76a5sd7
    targets_file:
        doc: >
          Csv file of the Iris Plants Database targets.
          Not required for making the prediction.
        type: str
        example:
            url: https://zenodo.org/path/to/example_files/targets.csv  # example file
            md5: 76sd8f7687sd6fs68a67
        optional: True  # if not present, the `targets` field will not be present in the dataloader output
info:
    authors: 
        - name: Your Name
          github: your_github_account
          email: your_email@host.org
    doc: Model predicting the Iris species
dependencies:
    conda:
      - python
      - pandas
      - numpy
      - sklearn
output_schema:
    inputs:
        features:
            shape: (4,)
            doc: Features in cm: sepal length, sepal width, petal length, petal width.
    targets:
        shape: (3, )
        doc: One-hot encoded array of classes: setosa, versicolor, virginica.
    metadata:  # field providing additional information to the samples (not directly required by the model)
        example_row_number:
            type: int
            doc: Just an example metadata column
```


## type

The type of the dataloader indicates from which class the dataloader is inherits. It has to be one of the following values:

- `PreloadedDataset`
- `Dataset`
- `BatchDataset`
- `SampleIterator`
- `SampleGenerator`
- `BatchIterator`
- `BatchGenerator`

## defined_as

`defined_as` indicates where the dataloader class can be found. It is a string value of `file.ClassName` where file 
refers to file `file.py` in the same directory as `dataloader.yaml` which contains the data-loader class `ClassName`.
 E.g.: `dataloader.MyDataLoader`.

This class will then be instantiated by Kipoi with keyword arguments that have to be mentioned explicitly in 
`args` (see below).

## args

A dataloader will always require arguments, they might for example be a path to the reference genome fasta file, a 
bed file that defines which regions should be investigated, etc. Dataloader arguments are given defined as a yaml 
dictionary with argument names as keys, e.g.:

```yaml
args:
   reference_fasta:
       example:
           url: https://zenodo.org/path/to/example_files/chr22.fa
           md5: 765sadf876a
   argument_2:
       example:
           url: https://zenodo.org/path/to/example_files/example_input.txt
           md5: 786as8d7aasd
```

An argument has the following fields:

* `doc`: A free text field describing the argument
* `example`: A value that can be used to demonstrate the functionality of the dataloader and of the entire model. 
Those example files are very useful for users and for automatic testing procedures. For example the command line 
call `kipoi test` uses the exmaple values given for dataloader arguments to assess that a model can be used and 
is functional. It is therefore important to submit the URLs of all necessary example files with the model.
* `type`: Optional: datatype of the argument (`str`, `bool`, `int`, `float`)
* `default`: This field is used to define external zenodo or figshare links that are automatically downloaded and 
assigned. See example below.
* `optional`: Optional: Boolean flag (`true` / `false`) for an argument if it is optional.

If your dataloader requires an external data file at runtime which are not example/test files, you can specify these using the `default` attribute. `default` will override the default arguments of the dataloader init method (e.g. `dataloader.MyDataloader.__init__`). Example:

```yaml
defined_as: dataloader.MyDataset
args:
   ...
   override_me:
       default: 10
   essential_other_file:
       default:  # download and replace with the path on the local filesystem
           url: https://zenodo.org/path/to/my/essential/other/file.xyz
           md5: 765sadf876a
...
```

## info

The `info` field of a dataloader.yaml file contains general information about the model.

* `authors`: a list of authors with the field: `name`, and the optional fields: `github` and `email`. Where the 
`github` name is the github user id of the respective author
* `doc`: Free text documentation of the dataloader. A short description of what it does.
* `version`: Version of the dataloader
* `license`: String indicating the license, if not defined it defaults to `MIT`
* `tags`: A list of key words describing the dataloader and its use cases

A dummy example could look like this:

```yaml
info:
  authors:
    - name: My Name
      github: myGithubName
      email: my@email.com
  doc: Datalaoder for my fancy model description
  version: 1.0
  license: GNU
  tags:
    - TFBS
    - tag2
```

## output_schema

`output_schema` defines what the dataloader outputs are, what they consist in, what the dimensions are and some 
additional meta data.

`output_schema` contains three categories `inputs`, `targets` and `metadata`. `inputs` and `targets` each specify the 
shapes of data generated for the model input and model. Offering the `targets` option enables the opportunity to 
possibly train models with the same dataloader.

In general model inputs and outputs can either be a numpy array, a list of numpy arrays or a dictionary (or 
`OrderedDict`) of numpy arrays. Whatever format is defined in the schema is expected to be produced by the dataloader 
and is expected to be accepted as input by the model. The three different kinds are represented by the single entries, 
lists or dictionaries in the yaml definition:

* A single numpy array as input or target:

```yaml
output_schema:
    inputs:
       name: seq
       shape: (1000,4)
```

* A list of numpy arrays as inputs or targets:

```yaml
output_schema:
    targets:
       - name: seq
         shape: (1000,4)
       - name: inp2
         shape: (10)
```

* A list of numpy arrays as inputs or targets:

```yaml
output_schema:
    inputs:
       seq:
         shape: (1000,4)
       inp2:
         shape: (10)
```

### `inputs`
The `inputs` fields of `output_schema` may be lists, dictionaries or single occurences of the following entries:

* `shape`: Required: A tuple defining the shape of a single input sample. E.g. for a model that predicts a batch of `(1000, 4)` inputs `shape: (1000, 4)` should be set. If a dimension is of variable size then the numerical should be replaced by `None`.
* `doc`: A free text description of the model input
* `name`: Name of model input, not required if input is a dictionary.
* `special_type`: Possibility to flag that respective input is a 1-hot encoded DNA sequence (`special_type: DNASeq`) or a string DNA sequence (`special_type: DNAStringSeq`).
* `associated_metadata`: Link the respective model input to metadata, such as a genomic region. E.g: If model input is a DNA sequence, then metadata may contain the genomic region from where it was extracted. If the associated `metadata` field is called `ranges` then `associated_metadata: ranges` has to be set.


### `targets`

The `targets` fields of `schema` may be lists, dictionaries or single occurences of the following entries:

* `shape`: Required: Details see in `input`
* `doc`: A free text description of the model target
* `name`: Name of model  target, not required if target is a dictionary.
* `column_labels`: Labels for the tasks of a multitask matrix output. Can be the file name of a text file containing the task labels (one label per line).


### `metadata`
Metadata fields capture additional information on the data generated by the dataloader. So for example a model input can be linked to a metadata field using its `associated_metadata` flag (see above). The metadata fields themselves are yaml dictionaries where the name of the metadata field is the key of dictionary and possible attributes are:

* `doc`: A free text description of the metadata element
* `type`: The datatype of the metadata field: `str`, `int`, `float`, `array`, `GenomicRanges`. Where the convenience class `GenomicRanges` is defined in `kipoi.metadata`, which is essentially an in-memory representation of a bed file.

Definition of `metadata` is essential for postprocessing algorihms such as variant effect prediction. Please refer to their detailed description for their requirements.

An example of the defintion of dataloader.yaml with `metadata` can be seen here:

```yaml
output_schema:
    inputs:
       - name: seq
         shape: (1000,4)
         associated_metadata: my_ranges
       - name: inp2
         shape: (10)
    ...
    metadata:
         my_ranges:
            type: GenomicRanges
            doc: Region from where inputs.seq was extracted
```

## dependencies
One of the core elements of ensuring functionality of a dataloader is to define software dependencies correctly and strictly. Dependencies can be defined for conda and for pip using the `conda` and `pip` sections respectively.

Both can either be defined as a list of packages or as a text file (ending in `.txt`) which lists the dependencies.

Conda as well as pip dependencies can and should be defined with exact versions of the required packages, as defining a package version using e.g.: `package>=1.0` is very likely to break at some point in future.

### conda
Conda dependencies can be defined as lists or if the dependencies are defined in a text file then the path of the text must be given (ending in `.txt`).

If conda packages need to be loaded from a channel then the nomenclature `channel_name::package_name` can be used.

### pip

Pip dependencies can be defined as lists or if the dependencies are defined in a text file then the path of the text must be given (ending in `.txt`).

