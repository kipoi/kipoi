from __future__ import absolute_import
from __future__ import print_function

import sys
import re
import os
import abc
import six
import textwrap
import inspect
from collections import OrderedDict

import related
import kipoi  # for .config module
from kipoi.specs import DataLoaderDescription, Info, example_kwargs, RemoteFile, download_default_args
from kipoi_utils import (load_module, cd, getargs, classproperty, inherits_from, rsetattr,
                    _get_arg_name_values, load_obj, infer_parent_class, override_default_kwargs)
from kipoi_utils.external.torch.data import DataLoader
from kipoi_utils.data_utils import (numpy_collate, numpy_collate_concat, get_dataset_item,
                              DataloaderIterable, batch_gen, get_dataset_lens, iterable_cycle)
from tqdm import tqdm
import types

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#
PREPROC_IFILE_TYPES = ['DNA_regions']
PREPROC_IFILE_FORMATS = ['bed3']


class BaseDataLoader(object):
    """Abstract Dataloader class
    """
    __metaclass__ = abc.ABCMeta

    # dataloader descriptors. Parsed from DataLoaderSchema in _add_description_factory()
    type = None
    defined_as = None
    args = None
    output_schema = None
    info = None
    dependencies = None
    path = None
    postprocessing = None
    # optionally set in get_dataloader_factory
    source = None
    source_dir = None

    # ---------------------------------
    # Core data-loading methods

    @abc.abstractmethod
    def batch_iter(self, **kwargs):
        raise NotImplementedError

    def batch_train_iter(self, cycle=True, **kwargs):
        """Returns samples directly useful for training the model:
        (x["inputs"],x["targets"])

        Args:
          cycle: when True, the returned iterator will run indefinitely go through the dataset
            Use True with `fit_generator` in Keras.
          **kwargs: Arguments passed to self.batch_iter(**kwargs)
        """
        if cycle:
            return ((x["inputs"], x["targets"])
                    for x in iterable_cycle(self._batch_iterable(**kwargs)))
        else:
            return ((x["inputs"], x["targets"]) for x in self.batch_iter(**kwargs))

    def batch_predict_iter(self, **kwargs):
        """Returns samples directly useful for prediction x["inputs"]

        Args:
          **kwargs: Arguments passed to self.batch_iter(**kwargs)
        """
        return (x["inputs"] for x in self.batch_iter(**kwargs))

    def load_all(self, **kwargs):
        """Loads and returns the whole dataset

        Arguments:
            **kwargs: passed to batch_iter()
        """
        return numpy_collate_concat([x for x in tqdm(self.batch_iter(**kwargs))])

    # ---------------------------------
    # Book-keeping methods dealing with
    # the annotation of the dataloader
    @classmethod
    def _add_description_factory(cls, descr):
        """Factory method which populates the un-set class variables

        Returns:
          new dataloader class
        """
        for field in ['type', 'defined_as', 'args', 'output_schema',
                      'info', 'dependencies', 'path', 'postprocessing']:
            setattr(cls, field, getattr(descr, field))
        return cls

    @classproperty
    def example_kwargs(cls):
        if cls.args is None:
            raise ValueError("Class description `args` is missing. "
                             "Use `_add_description_factory` to annotate the class")
        if cls.source_dir is None:
            logger.info("Using current directory for source_dir")
            cls.source_dir = os.getcwd()

        # Add init_example method.
        # example_kwargs also downloads files to {dataloader_dir}/dataloader_files
        return example_kwargs(cls.args, os.path.join(cls.source_dir, "downloaded/example_files"))

    @classmethod
    def download_example(cls, output_dir, absolute_path=False, dry_run=False):
        """Download the example files to the desired directory

        # Arguments
          output_dir: output directory where to store the file
          absolute_path: if True, return absolute paths to the
            output directories
          dry_run: if True, return only the file paths without
            actually downloading the files

        # Returns
          dictionary of keyword arguments for the dataloader
        """
        return example_kwargs(cls.args, output_dir, absolute_path=absolute_path, dry_run=dry_run)

    @classmethod
    def init_example(cls):
        """Instantiate the class using example_kwargs
        """
        if cls.source_dir is not None:
            with cd(cls.source_dir):
                # always init the example in the original directory
                return cls(**cls.example_kwargs)
        else:
            return cls(**cls.example_kwargs)

    @classmethod
    def print_args(cls, format_examples_json=False):
        """Print dataloader kwargs

        # Arguments
          format_examples_json: format the results as json
        """
        from kipoi_utils.external.related.fields import UNSPECIFIED
        if not hasattr(cls, "args"):
            logger.warning("No keyword arguments defined for the given dataloader.")
            return None
            # print("No keyword arguments defined for the given dataloader.")
        args = cls.args
        for k in args:
            print("{0}:".format(k))
            for elm in ["doc", "type", "optional", "example"]:
                if hasattr(args[k], elm) and \
                        (not isinstance(getattr(args[k], elm), UNSPECIFIED)):
                    print("    {0}: {1}".format(elm, getattr(args[k], elm)))
        # example_kwargs = cls.example_kwargs
        # print("-" * 80)
        # if hasattr(cls, "example_kwargs"):
        #     if format_examples_json:
        #         import json
        #         example_kwargs = json.dumps(example_kwargs)
        #     print("Example keyword arguments are: {0}".format(str(example_kwargs)))

    @classmethod
    def get_output_schema(cls):
        return cls.output_schema


def kipoi_dataloader(override=dict()):
    """Decorator for converting a Dataloader class with dataloader.yaml description in the docstring
    into a proper Kipoi dataloader

    It parses the doc-string of the class as the DataloaderDescription
    and populates the class description attributes with it

    # Arguments
        override: dictionary containing values to override or specify in the decorated class.
          supports nesting the attributes. example: `{'info.authors': [Author(name='name')]}`

    # __call__
        dataloader containing the descripiton specified in the yaml doc-string
    """

    def wrap(cls):
        if inspect.isfunction(cls):
            raise ValueError("Function-based dataloader are currently not supported with kipoi_dataloader decorator")

        # figure out the right dataloader type
        dl_type_inferred = infer_parent_class(cls, AVAILABLE_DATALOADERS)
        if dl_type_inferred is None:
            raise ValueError("Dataloader needs to inherit from one of the available dataloaders {}".format(list(AVAILABLE_DATALOADERS)))

        # or not inherits_from(cls, Dataset)
        doc = cls.__doc__
        doc = textwrap.dedent(doc)  # de-indent

        if not re.match("^defined_as: ", doc):
            doc = "defined_as: {}\n".format(cls.__name__) + doc
        if not re.match("^type: ", doc):
            doc = "type: {}\n".format(dl_type_inferred) + doc

        # parse the yaml
        yaml_dict = related.from_yaml(doc)
        dl_descr = DataLoaderDescription.from_config(yaml_dict)

        # override parameters
        for k, v in six.iteritems(override):
            rsetattr(dl_descr, k, v)

        # setup optional parameters
        arg_names, default_values = _get_arg_name_values(cls)

        if set(dl_descr.args) != set(arg_names):
            raise ValueError("Described args don't exactly match the implemented arguments"
                             "docstring: {}, actual: {}".format(list(dl_descr.args), list(arg_names)))

        # properly set optional / non-optional argument values
        for i, arg in enumerate(dl_descr.args):
            optional = i >= len(arg_names) - len(default_values)
            if dl_descr.args[arg].optional and not optional:
                logger.warning("Parameter {} was specified as optional. However, there "
                            "are no defaults for it. Specifying it as not optinal".format(arg))
            dl_descr.args[arg].optional = optional

        dl_descr.info.name = cls.__name__

        # enrich the class with dataloader description
        return cls._add_description_factory(dl_descr)
    return wrap

# --------------------------------------------
# Different implementations

# Other options:
# - generator - sample, batch-based
#   - yield.
# - iterator, iterable - sample, batch-based
#   - __iter__()
#     - __next__()
# - full dataset
#   - everything numpy arrays with the same first axis length


class PreloadedDataset(BaseDataLoader):
    """Generated by supplying a function returning the full dataset.

    The full dataset is a nested (list/dict) python structure of numpy arrays
    with the same first axis dimension.
    """
    data_fn = None

    @classmethod
    def from_fn(cls, data_fn):
        """setup the class variable
        """
        cls.data_fn = staticmethod(data_fn)
        return cls

    @classmethod
    def from_data(cls, data):
        return cls.from_data_fn(lambda: data)()

    @classmethod
    def _get_data_fn(cls):
        assert cls.data_fn is not None
        return cls.data_fn

    def __init__(self, *args, **kwargs):
        self.data = self._get_data_fn()(*args, **kwargs)
        lens = get_dataset_lens(self.data, require_numpy=True)
        # check that all dimensions are the same
        assert len(set(lens)) == 1
        self.n = lens[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return get_dataset_item(self.data, index)

    def _batch_iterable(self, batch_size=32, shuffle=False, drop_last=False, **kwargs):
        """See batch_iter docs

        Returns:
          iterable
        """
        dl = DataLoader(self, batch_size=batch_size,
                        collate_fn=numpy_collate,
                        shuffle=shuffle,
                        num_workers=0,
                        drop_last=drop_last)
        return dl

    def batch_iter(self, batch_size=32, shuffle=False, drop_last=False, **kwargs):
        """Return a batch-iterator

        Arguments:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: False).
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If False and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: False)

        Returns:
            iterator
        """
        dl = self._batch_iterable(batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=drop_last,
                                  **kwargs)
        return iter(dl)

    def load_all(self, **kwargs):
        """Load the whole dataset into memory

        Arguments:
            **kwargs: ignored
        """
        return self.data


class Dataset(BaseDataLoader):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __getitem__(self, index):
        """Return one sample

        index: {0, ..., len(self)-1}
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        """Return the number of all samples
        """
        raise NotImplementedError

    def _batch_iterable(self, batch_size=32, shuffle=False, num_workers=0, drop_last=False, **kwargs):
        """Return a batch-iteratrable

        See batch_iter docs

        Returns:
            Iterable
        """
        dl = DataLoader(self,
                        batch_size=batch_size,
                        collate_fn=numpy_collate,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        drop_last=drop_last,
                        **kwargs)
        return dl

    def batch_iter(self, batch_size=32, shuffle=False, num_workers=0, drop_last=False, **kwargs):
        """Return a batch-iterator

        Arguments:
            dataset (Dataset): dataset from which to load the data.
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled
                at every epoch (default: False).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main process
                (default: 0)
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If False and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: False)

        Returns:
            iterator
        """
        dl = self._batch_iterable(batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=drop_last,
                                  **kwargs)
        return iter(dl)

    def load_all(self, batch_size=32, **kwargs):
        """Load the whole dataset into memory
        Arguments:
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
        """
        return numpy_collate_concat([x for x in tqdm(self.batch_iter(batch_size, **kwargs))])


class BatchDataset(BaseDataLoader):
    """An abstract class representing a BatchDataset.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __getitem__(self, index):
        """Return one batch
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        """Number of all batches
        """
        raise NotImplementedError

    def _batch_iterable(self, num_workers=0, **kwargs):
        """Return a batch-iteratorable

        See batch_iter for docs
        """
        dl = DataLoader(self, batch_size=1,
                        collate_fn=numpy_collate_concat,
                        shuffle=False,
                        num_workers=num_workers,
                        drop_last=False)
        return dl

    def batch_iter(self, num_workers=0, **kwargs):
        """Return a batch-iterator

        Arguments:
            dataset (Dataset): dataset from which to load the data.
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main process
                (default: 0)
        Returns:
            iterator
        """
        dl = self._batch_iterable(num_workers=num_workers, **kwargs)
        return iter(dl)


class SampleIterator(BaseDataLoader):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    # TODO - how to maintain compatibility with python2?
    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError

    next = __next__

    def batch_iter(self, batch_size=32, **kwargs):
        return batch_gen(iter(self), batch_size=batch_size)

    def _batch_iterable(self, batch_size=32, **kwargs):
        kwargs['batch_size'] = batch_size
        return DataloaderIterable(self, kwargs)


class BatchIterator(BaseDataLoader):

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    # TODO - how to maintain compatibility with python2?
    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError

    next = __next__

    def batch_iter(self, **kwargs):
        return iter(self)

    def _batch_iterable(self, **kwargs):
        return DataloaderIterable(self, kwargs)


class SampleGenerator(BaseDataLoader):
    """Transform a generator of samples into SampleIterator
    """
    generator_fn = None

    @classmethod
    def from_fn(cls, generator_fn):
        """setup the class variable
        """
        cls.generator_fn = staticmethod(generator_fn)
        return cls

    @classmethod
    def _get_generator_fn(cls):
        assert cls.generator_fn is not None
        return cls.generator_fn

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        """Return a new generator every time
        """
        return self._get_generator_fn()(*self.args, **self.kwargs)

    def batch_iter(self, batch_size=32, **kwargs):
        return batch_gen(iter(self), batch_size=batch_size)

    def _batch_iterable(self, batch_size=32, **kwargs):
        kwargs['batch_size'] = batch_size
        return DataloaderIterable(self, kwargs)


class BatchGenerator(BaseDataLoader):
    """Transform a generator of batches into BatchIterator
    """
    generator_fn = None

    @classmethod
    def from_fn(cls, generator_fn):
        cls.generator_fn = staticmethod(generator_fn)
        return cls

    @classmethod
    def _get_generator_fn(cls):
        assert cls.generator_fn is not None
        return cls.generator_fn

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return self._get_generator_fn()(*self.args, **self.kwargs)

    def batch_iter(self, **kwargs):
        return iter(self)

    def _batch_iterable(self, **kwargs):
        return DataloaderIterable(self, kwargs)
# --------------------------------------------


def get_dataloader(dataloader, source="kipoi"):
    """Loads the dataloader

    # Arguments
        dataloader (str): dataloader name
        source (str): source name

    # Returns
    - Instance of class inheriting from `kipoi.data.BaseDataLoader` (like `kipoi.data.Dataset`)
           decorated with additional attributes.

    # Methods
    - __batch_iter(batch_size, num_workers, **kwargs)__
         - Arguments
             - **batch_size**: batch size
             - **num_workers**: Number of workers to use in parallel.
             - ****kwargs**: Other kwargs specific to each dataloader
         - Yields
             - `dict` with `"inputs"`, `"targets"` and `"metadata"`
    - __batch_train_iter(cycle=True, **kwargs)__
         - Arguments
             - **cycle**: if True, cycle indefinitely
             - ****kwargs**: Kwargs passed to `batch_iter()` like `batch_size`
         - Yields
             - tuple of ("inputs", "targets") from the usual dict returned by `batch_iter()`
    - __batch_predict_iter(**kwargs)__
         - Arguments
             - ****kwargs**: Kwargs passed to `batch_iter()` like `batch_size`
         - Yields
             - "inputs" field from the usual dict returned by `batch_iter()`
    - __load_all(**kwargs)__ - load the whole dataset into memory
         - Arguments
             - ****kwargs**: Kwargs passed to `batch_iter()` like `batch_size`
         - Returns
             - `dict` with `"inputs"`, `"targets"` and `"metadata"`
    - **init_example()** - instantiate the dataloader with example kwargs
    - **print_args()** - print information about the required arguments

    # Appended attributes
    - **type** (str): dataloader type (class name)
    - **defined_as** (str): path and dataloader name
    - **args** (list of kipoi.specs.DataLoaderArgument): datalaoder argument description
    - **info** (kipoi.specs.Info): general information about the dataloader
    - **schema** (kipoi.specs.DataloaderSchema): information about the input/output
            data modalities
    - **dependencies** (kipoi.specs.Dependencies): class specifying the dependencies.
          (implements `install` method for running the installation)
    - **name** (str): model name
    - **source** (str): model source
    - **source_dir** (str): local path to model source storage
    - **postprocessing** (dict): dictionary of loaded plugin specifications
    - **example_kwargs** (dict): kwargs for running the provided example
    """
    # if source == 'py':
    #     # load it from the python object
    #     sys.path.append(os.path.getcwd())
    #     return DataLoaderImport(defined_as=dataloader).get()
    # TODO - allow source=py

    # pull the dataloader & get the dataloader directory
    if isinstance(source, str):
        source = kipoi.config.get_source(source)
    source.pull_dataloader(dataloader)
    dataloader_dir = source.get_dataloader_dir(dataloader)

    # --------------------------------------------
    # Setup dataloader description
    descr = source.get_dataloader_descr(dataloader)
    with cd(dataloader_dir):  # move to the dataloader directory temporarily
        if "::" in descr.defined_as:
            # old API
            file_path, obj_name = tuple(descr.defined_as.split("::"))
            CustomDataLoader = getattr(load_module(file_path), obj_name)
        else:
            # new API - directly specify the object
            CustomDataLoader = load_obj(descr.defined_as)

    # download util links if specified under default & override the default parameters
    override = download_default_args(descr.args, source.get_dataloader_download_dir(dataloader))
    if override:
        # override default arguments specified under default
        CustomDataLoader = override_default_kwargs(CustomDataLoader, override)

    # infer the type
    if descr.type is None:
        if inspect.isfunction(CustomDataLoader):
            raise ValueError("Datalodaers implemented as functions/generator need to specify the type flag in dataloader.yaml")
        else:
            # figure out the right dataloader type
            descr.type = infer_parent_class(CustomDataLoader, AVAILABLE_DATALOADERS)
            if descr.type is None:
                raise ValueError("Dataloader needs to inherit from one of the available dataloaders {}".format(list(AVAILABLE_DATALOADERS)))

        # check that descr.type is correct
    if descr.type not in AVAILABLE_DATALOADERS:
        raise ValueError("dataloader type: {0} is not in supported dataloaders:{1}".
                         format(descr.type, list(AVAILABLE_DATALOADERS.keys())))

    # check that the extractor arguments match yaml arguments
    if not getargs(CustomDataLoader) == set(descr.args.keys()):
        raise ValueError("DataLoader arguments: \n{0}\n don't match ".format(set(getargs(CustomDataLoader))) +
                         "the specification in the dataloader.yaml file:\n{0}".
                         format(set(descr.args.keys())))

    # check that CustomDataLoader indeed interits from the right DataLoader
    if descr.type in DATALOADERS_AS_FUNCTIONS:
        # transform the functions into objects
        assert isinstance(CustomDataLoader, types.FunctionType)
        CustomDataLoader = AVAILABLE_DATALOADERS[descr.type].from_fn(CustomDataLoader)
    else:
        if not issubclass(CustomDataLoader, AVAILABLE_DATALOADERS[descr.type]):
            raise ValueError("DataLoader does't inherit from the specified dataloader: {0}".
                             format(AVAILABLE_DATALOADERS[descr.type].__name__))
    logger.info('successfully loaded the dataloader {} from {}'.
                format(dataloader, os.path.normpath(os.path.join(dataloader_dir, descr.defined_as))))

    # enrich the original dataloader class with description
    Dl = CustomDataLoader._add_description_factory(descr)
    # add other fields
    Dl.source = source
    Dl.source_dir = dataloader_dir
    return Dl


# simplify - alias
get_dataloader_factory = get_dataloader

# NOTE: the dataloaders need to be ordered in a way they inherit from each other
# e.g. child can't appear before the parent in the list below
# TODO this could be automatically checked in the future
AVAILABLE_DATALOADERS = OrderedDict([
    ("PreloadedDataset", PreloadedDataset),
    ("Dataset", Dataset),
    ("BatchDataset", BatchDataset),
    ("SampleIterator", SampleIterator),
    ("SampleGenerator", SampleGenerator),
    ("BatchIterator", BatchIterator),
    ("BatchGenerator", BatchGenerator)])

DATALOADERS_AS_FUNCTIONS = ["PreloadedDataset", "SampleGenerator", "BatchGenerator"]
