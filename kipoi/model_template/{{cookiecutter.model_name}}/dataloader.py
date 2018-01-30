"""Dataloader
"""
import pickle
{%- if cookiecutter.dataloader_type not in ["PreloadedDataset", "SampleGenerator", "BatchGenerator"] %}
from kipoi.data import {{ cookiecutter.dataloader_type }}
{%- endif %}
import pandas as pd
import numpy as np
import os

# access the absolute path to this script
# https://stackoverflow.com/questions/3718657/how-to-properly-determine-current-script-directory-in-python
import inspect
this_file_path = os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
this_dir = os.path.dirname(this_file_path)


def read_pickle(f):
    with open(f, "rb") as f:
        return pickle.load(f)
{% macro return_dict(w_targets=True, ret='return') -%}
{{ ret }} {
    {%- if cookiecutter.model_input_type == 'np.array' %}
    "inputs": x_features,
    {%- elif cookiecutter.model_input_type == 'list of np.arrays' %}
    "inputs": [x_features],
    {%- elif cookiecutter.model_input_type == 'dict of np.arrays' %}
    "inputs": {
        "features": x_features  
    },
    {%- endif %}
    {%- if w_targets %}
    {%- if cookiecutter.model_output_type == 'np.array' %}
    "targets": y_class,
    {%- elif cookiecutter.model_output_type == 'list of np.arrays' %}
    "targets": [y_class],
    {%- elif cookiecutter.model_output_type == 'dict of np.arrays' %}
    "targets": {
        "iris_class": y_class
    },
    {%- endif %}
    {%- endif %}
    "metadata": {
        "example_row_number": idx
    }
}    
{%- endmacro %}
{% if cookiecutter.dataloader_type == 'Dataset' %}
class My{{ cookiecutter.dataloader_type }}(Dataset):

    def __init__(self, features_file, targets_file=None):
        self.features_file = features_file
        self.targets_file = targets_file

        self.x_transformer = read_pickle(this_dir + "/dataloader_files/x_transformer.pkl")

        self.features = pd.read_csv(features_file)
        if targets_file is not None:
            self.y_transformer = read_pickle(this_dir + "/dataloader_files/y_transformer.pkl")
            self.targets = pd.read_csv(targets_file)
            assert len(self.targets) == len(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x_features = np.ravel(self.x_transformer.transform(self.features.iloc[idx].values[np.newaxis]))
        if self.targets_file is None:
            {{return_dict(False)|indent(12)}}
        else:
            y_class = np.ravel(self.y_transformer.transform(self.targets.iloc[idx].values[np.newaxis]))
            {{return_dict(True)|indent(12)}}

{% elif cookiecutter.dataloader_type == 'PreloadedDataset' %}
def my{{ cookiecutter.dataloader_type }}(features_file, targets_file=None):
    x_transformer = read_pickle(this_dir + "/dataloader_files/x_transformer.pkl")

    x_features = x_transformer.transform(pd.read_csv(features_file).values)
    idx = np.arange(x_features.shape[0])
    if targets_file is None:
        {{return_dict(False)|indent(8)}}
    else:
        y_transformer = read_pickle(this_dir + "/dataloader_files/y_transformer.pkl")
        y_class = y_transformer.transform(pd.read_csv(targets_file).values)
        {{return_dict(True)|indent(8)}}

{% elif cookiecutter.dataloader_type == 'BatchDataset' %}
class My{{ cookiecutter.dataloader_type }}(BatchDataset):

    def __init__(self, features_file, batch_size=5, targets_file=None):
        self.features_file = features_file
        self.targets_file = targets_file
        self.batch_size = batch_size

        self.x_transformer = read_pickle(this_dir + "/dataloader_files/x_transformer.pkl")

        self.features = pd.read_csv(features_file)
        if targets_file is not None:
            self.y_transformer = read_pickle(this_dir + "/dataloader_files/y_transformer.pkl")
            self.targets = pd.read_csv(targets_file)

    def __len__(self):
        return int(np.ceil(len(self.features) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(len(self.features), (idx + 1) * self.batch_size)
        x_features = self.x_transformer.transform(self.features.iloc[start:end].values)
        if self.targets_file is None:
            {{return_dict(False)|indent(12)}}
        else:
            y_class = self.y_transformer.transform(self.targets.iloc[start:end].values)
            {{return_dict(True)|indent(12)}}


{% elif cookiecutter.dataloader_type == 'SampleIterator' %}
def row2np(row):
    return np.array(row.strip().split(",")).astype(float)

class My{{ cookiecutter.dataloader_type }}(SampleIterator):

    def __init__(self, features_file, targets_file=None):
        self.features_file = features_file
        self.targets_file = targets_file

        self.x_transformer = read_pickle(this_dir + "/dataloader_files/x_transformer.pkl")

        self.features_it = open(self.features_file, "r")
        next(self.features_it)  # skip the header

        self.idx = 0
        if targets_file is not None:
            self.y_transformer = read_pickle(this_dir + "/dataloader_files/y_transformer.pkl")
            self.targets_it = open(self.targets_file, "r")
            next(self.targets_it)  # skip the header

    def __iter__(self):
        return self

    def __next__(self):
        x_features = np.ravel(self.x_transformer.transform(row2np(next(self.features_it))[np.newaxis]))
        idx = self.idx
        self.idx += 1
        if self.targets_file is None:
            {{return_dict(False)|indent(12)}}
        else:
            y_class = np.ravel(self.y_transformer.transform(row2np(next(self.targets_it))[np.newaxis]))
            {{return_dict(True)|indent(12)}}

    # python2
    next = __next__


{% elif cookiecutter.dataloader_type == 'BatchIterator' %}
def row2np(row):
    return np.array(row.strip().split(",")).astype(float)


class My{{ cookiecutter.dataloader_type }}(BatchIterator):

    def __init__(self, features_file, batch_size=5, targets_file=None):
        self.features_file = features_file
        self.targets_file = targets_file
        self.batch_size = batch_size

        self.x_transformer = read_pickle(this_dir + "/dataloader_files/x_transformer.pkl")

        self.features_it = open(self.features_file, "r")
        next(self.features_it)  # skip the header
        self.idx = 0

        if targets_file is not None:
            self.y_transformer = read_pickle(this_dir + "/dataloader_files/y_transformer.pkl")
            self.targets_it = open(self.targets_file, "r")
            next(self.targets_it)  # skip the header


    def __iter__(self):
        return self

    def __next__(self):
        x_feat = []
        y_targets = []
        idx = []
        start = self.batch_size * self.idx
        for i in range(self.batch_size):
            try:
                x_feat.append(row2np(next(self.features_it)))
                if self.features_file is not None:
                    y_targets.append(row2np(next(self.targets_it)))
                self.idx += 1
                idx.append(self.idx)
            except StopIteration:
                if len(x_feat) == 0:
                    raise StopIteration
                else:
                    break
        idx = np.array(idx)
        x_features = self.x_transformer.transform(np.stack(x_feat))
        if self.targets_file is None:
            {{return_dict(False)|indent(12)}}
        else:
            y_class = self.y_transformer.transform(np.stack(y_targets))
            {{return_dict(True)|indent(12)}}

    # python2
    next = __next__


{% elif cookiecutter.dataloader_type == 'SampleGenerator' %}
def row2np(row):
    return np.array(row.strip().split(",")).astype(float)


def my{{ cookiecutter.dataloader_type }}(features_file, targets_file=None):
    x_transformer = read_pickle(this_dir + "/dataloader_files/x_transformer.pkl")

    features_it = open(features_file, "r")
    next(features_it)  # skip the header
    if targets_file is not None:
        targets_it = open(targets_file, "r")
        y_transformer = read_pickle(this_dir + "/dataloader_files/y_transformer.pkl")
        next(targets_it)  # skip the header
    for idx, row in enumerate(features_it):
        x_features = np.ravel(x_transformer.transform(row2np(row)[np.newaxis]))
        if targets_file is None:
            {{return_dict(False, "yield")|indent(12)}}
        else:
            y_class = np.ravel(y_transformer.transform(row2np(next(targets_it))[np.newaxis]))
            {{return_dict(True, "yield")|indent(12)}}


{% elif cookiecutter.dataloader_type == 'BatchGenerator' %}
def row2np(row):
    return np.array(row.strip().split(",")).astype(float)

def my{{ cookiecutter.dataloader_type }}(features_file, targets_file=None, batch_size=5):
    x_transformer = read_pickle(this_dir + "/dataloader_files/x_transformer.pkl")

    features_it = open(features_file, "r")
    next(features_it)  # skip the header
    if targets_file is not None:
        targets_it = open(targets_file, "r")
        y_transformer = read_pickle(this_dir + "/dataloader_files/y_transformer.pkl")
        next(targets_it)  # skip the header
    gidx = 0
    try:
        while True:
            x_feat = []
            y_targets = []
            idx = []
            for i in range(batch_size):
                x_feat.append(row2np(next(features_it)))
                if targets_file is not None:
                    y_targets.append(row2np(next(targets_it)))
                idx.append(gidx)
                gidx += 1
            idx = np.array(idx)
            x_features = x_transformer.transform(np.stack(x_feat))
            if targets_file is None:
                {{return_dict(False, "yield")|indent(16)}}
            else:
                y_class = y_transformer.transform(np.stack(y_targets))
                {{return_dict(True, "yield"|indent(16))}}
    except StopIteration:
        if len(x_feat) == 0:
            raise StopIteration
        idx = np.array(idx)
        if targets_file is None:
            {{return_dict(False, "yield")|indent(12)}}
        else:
            y_class = y_transformer.transform(np.stack(y_targets))
            {{return_dict(True, "yield")|indent(12)}}
{% endif %}
