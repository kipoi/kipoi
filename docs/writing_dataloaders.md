## Writing dataloaders

### Possible classess

There are 7 different way how you can implement the dataloader:

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


## Examples

See [tests/test_12_dataloader_classes.py](../tests/test_12_dataloader_classes.py) for implementation examples of each
dataloader type.
