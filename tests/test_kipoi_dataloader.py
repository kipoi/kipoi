import pytest
import numpy as np
from kipoi.data import Dataset
from kipoi.specs import DataLoaderArgument, DataLoaderSchema, DataLoaderDescription
from kipoi.specs import Author, Dependencies
from kipoi_utils.utils import inherits_from
from collections import OrderedDict
import related
from kipoi.data import kipoi_dataloader

deps = Dependencies(pip='kipoiseq')
package_authors = [Author(name='John')]


@kipoi_dataloader(override={"dependencies": deps, 'info.authors': package_authors})
class Dl(Dataset):
    """
    info:
        doc: short doc
    args:
        arg1:
            doc: this is arg1
            example: hey
        n:
            doc: length of the dataset
    output_schema:
        inputs:
            name: seq
            shape: (10,4)
            special_type: DNASeq
            doc: inputs doc
        targets:
            name: targets
            shape: (None,)
            doc: targets doc
    """

    def __init__(self, arg1, n=10):
        self.arg1 = arg1
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "inputs": np.arange((10, 4)),
            "targets": np.arange(self.arg1)
        }


class Dl_warn(Dataset):
    """
    info:
        doc: short doc
    args:
        arg1:
            doc: this is arg1
            example: hey
            optional: True
        n:
            doc: length of the dataset
            optional: False
    output_schema:
        inputs:
            name: seq
            shape: (10,)
            doc: inputs doc
        targets:
            name: targets
            shape: (None,)
            doc: targets doc
    """

    def __init__(self, arg1, n=10):
        self.arg1 = arg1
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "inputs": np.arange(10),
            "targets": np.arange(self.arg1)
        }


class Dl_error(Dataset):
    """
    info:
        doc: short doc
    args:
        arg1:
            doc: this is arg1
            example: hey
        n:
            doc: length of the dataset
        extra:
            doc: length of the dataset
    output_schema:
        inputs:
            name: seq
            shape: (10,)
            doc: inputs doc
        targets:
            name: targets
            shape: (None,)
            doc: targets doc
    """

    def __init__(self, arg1, n=10):
        self.arg1 = arg1
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "inputs": np.arange(10),
            "targets": np.arange(self.arg1)
        }


def test_type_defined_as():
    assert Dl.type == 'Dataset'
    assert Dl.defined_as == 'Dl'   # TODO - get full python path?


def test_info():
    assert len(Dl.info.authors) == 1
    assert Dl.info.authors[0].name == 'John'
    assert Dl.info.doc == 'short doc'
    assert Dl.info.name == 'Dl'
    assert Dl.info.license == 'MIT'


def test_args():
    assert list(Dl.args.keys()) == ['arg1', 'n']
    assert Dl.args['arg1'] == DataLoaderArgument(doc='this is arg1',
                                                 example='hey',
                                                 optional=False,
                                                 # type=''  # TODO - infer the type as well?
                                                 name='arg1'
                                                 )
    assert Dl.args['n'] == DataLoaderArgument(doc='length of the dataset',
                                              optional=True,
                                              name='n'
                                              )


def test_dependencies():
    assert Dl.dependencies == deps


def test_output_schema():
    # inputs
    assert Dl.output_schema.inputs.name == 'seq'
    assert Dl.output_schema.inputs.shape == (10, 4)
    assert Dl.output_schema.inputs.doc == 'inputs doc'

    # targets
    assert Dl.output_schema.targets.name == 'targets'
    assert Dl.output_schema.targets.shape == (None,)   # TODO - try to infer the shape when overriding the parameters (arg1)
    assert Dl.output_schema.targets.doc == 'targets doc'


def test_path():
    assert Dl.path is None


def test_error():
    dec = kipoi_dataloader(override={"dependencies": deps, 'info.authors': package_authors})
    with pytest.raises(ValueError):
        dec(Dl_error)

    def a():
        pass

    with pytest.raises(ValueError):
        dec(a)

    class B():
        pass

    with pytest.raises(ValueError):
        dec(B)

    dl = dec(Dl_warn)
    assert dl.args['arg1'].optional == False
    assert dl.args['n'].optional == True
