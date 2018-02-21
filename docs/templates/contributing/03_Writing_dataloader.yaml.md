#dataloader.yaml

The dataloader.yaml file describes how a dataloader for a certain model can be created and how it has to be set up. A model without functional dataloader is as bad as a model that doesn't work, so the correct setup of the dataloader.yaml is essential for the use of a model in the zoo.

The main aim of a dataloader is to generate data with which a model can be run. It therefore has to return a dictionary with three keys: `inputs`, `targets`, `metadata`. The `inputs` and `targets` have to be compatible with the model, more details see below (definition of `output_schema`).


A dataloader has to be a subclass of the following classes which are defined in `kipoi.data`:

- `PreloadedDataset` - Function that returns the whole dataset as a nested dictionary/list of numpy arrays
  - **useful when:** the dataset is expected to load quickly and fit into the memory

- `Dataset` - Class that inherits from `kipoi.data.Dataset` and implements `__len__` and `__getitem__` methods. `__getitem__` returns a single sample from the dataset.
  - **useful when:** dataset length is easy to infer, there are no significant performance gain when reading data of the disk in batches

- `BatchDataset` - Class that inherits from `kipoi.data.BatchDataset` and implements `__len__` and `__getitem__` methods. `__getitem__` returns a single batch of samples from the dataset.
  - **useful when:** dataset length is easy to infer, and there is a significant performance gain when reading data of the disk in batches

- `SampleIterator` - Class that inherits from `kipoi.data.SampleIterator` and implements `__iter__` and `__next__` (`next` in python 2). `__next__` returns a single sample from the dataset or raises `StopIteration` if all the samples were already returned.
  - **useful when:** the dataset length is not know in advance or is difficult to infer, and there are no significant performance gain when reading data of the disk in batches

- `BatchIterator` - Class that inherits from `kipoi.data.BatchIterator` and implements `__iter__` and `__next__` (`next` in python 2). `__next__` returns a single batch of samples sample from the dataset or raises `StopIteration` if all the samples were already returned.
  - **useful when:** the dataset length is not know in advance or is difficult to infer, and there is a significant performance gain when reading data of the disk in batches

- `SampleGenerator` - A generator function that yields a single sample from the dataset and returns when all the samples were yielded.
  - **useful when:** same as for `SampleIterator`, but can be typically implemented in fewer lines of code

- `BatchGenerator` - A generator function that yields a single batch of samples from the dataset and returns when all the samples were yielded.
  - **useful when:** same as for `BatchIterator`, but can be typically implemented in fewer lines of code


Here is a table showing the (recommended) requirements for each dataloader type:

| Dataloader type   	| Length known? 	| Significant benefit from loading data in batches? 	| Fits into memory and loads quickly? 	|
|-------------------	|---------------	|---------------------------------------------------	|-------------------------------------	|
| PreloadedDataset  	| yes           	| yes                                               	| yes                                 	|
| Dataset           	| yes           	| no                                                	| no                                  	|
| BatchDataset      	| yes           	| yes                                               	| no                                  	|
| SampleIterator    	| no            	| no                                                	| no                                  	|
| BatchIterator     	| no            	| yes                                               	| no                                  	|
| SampleGenerator   	| no            	| no                                                	| no                                  	|
| BatchGenerator    	| no            	| yes                                               	| no                                  	|

To help understand the synthax of YAML please take a look at: [YAML Synthax Basics](http://docs.ansible.com/ansible/latest/YAMLSyntax.html#yaml-basics)



##type
The type of the dataloader indicates from which class the dataloader is inherits. It has to be one of the following values: `PreloadedDataset`, `Dataset`, `BatchDataset`, `SampleIterator`, `SampleGenerator`, `BatchIterator`, `BatchGenerator`.


## defined_as
`defined_as` indicates where the dataloader class can be found. It is a string value of `path/to/file.py::class_name` with a the relative path from where the dataloader.yaml lies. E.g.: `model_files/dataloader.py::MyDataLoader`.

This class will then be instantiated by Kipoi with keyword arguments that have to be mentioned explicitely in `args` (see below).

##args
A dataloader will always require arguments, they might for example be a path to the reference genome fasta file, a bed file that defines which regions should be investigated, etc. Dataloader arguments are given defined as a yaml dictionary with argument names as keys, e.g.:

```yaml
args:
   refernce_fasta:
       example: example_files/chr22.fa
   argument_2:
       example: example_files/example_input.txt
```

An argument has the following fields:

* `doc`: A free text field describing the argument
* `example`: A value that can be used to demonstrate the functionality of the dataloader and of the entire model. Those example files are very useful for users and for automatic testing procedures. For example the command line call `kipoi test` uses the exmaple values given for dataloader arguments to assess that a model can be used and is functional. It is therefore important to submit all necessary example files with the model.
* `type`: Optional: datatype of the argument (`str`, `bool`, `int`, `float`)
* `optional`: Optional: Boolean flag (`true` / `false`) for an argument if it is optional.

## info
The `info` field of a dataloader.yaml file contains general information about the model.

* `authors`: a list of authors with the field: `name`, and the optional fields: `github` and `email`. Where the `github` name is the github user id of the respective author
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

`output_schema` defines what the dataloader outputs are, what they consist in, what the dimensions are and some additional meta data.

`output_schema` contains three categories `inputs`, `targets` and `metadata`. `inputs` and `targets` each specify the shapes of data generated for the model input and model. Offering the `targets` option enables the opportunity to possibly train models with the same dataloader.

In general model inputs and outputs can either be a numpy array, a list of numpy arrays or a dictionary (or `OrderedDict`) of numpy arrays. Whatever format is defined in the schema is expected to be produced by the dataloader and is expected to be accepted as input by the model. The three different kinds are represented by the single entries, lists or dictionaries in the yaml definition:

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
* `special_type`: Possibility to flag that respective input is a 1-hot encoded DNA sequence (`special_type: DNASeq`) or a string DNA sequence (`special_type: DNAStringSeq`), which is important for variant effect prediction.
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

Definition of `metadata` is essential for postprocessing algorihms as variant effect prediction. Please refer to their detailed description for their requirements.

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

###conda
Conda dependencies can be defined as lists or if the dependencies are defined in a text file then the path of the text must be given (ending in `.txt`).

If conda packages need to be loaded from a channel then the nomenclature `channel_name::package_name` can be used.

###pip
Pip dependencies can be defined as lists or if the dependencies are defined in a text file then the path of the text must be given (ending in `.txt`).

## postprocessing

The postprocessing section of a dataloader.yaml is necessary to indicate that a dataloader is compatible with a certain kind of postprocessing feature available in Kipoi. At the moment only variant effect prediction is available for postprocessing. To understand how to set your dataloader up for variant effect prediction, please take a look at the documentation of variant effect prediction.

