"""Dataloader
"""
import pickle
from kipoi.data import Dataset, kipoi_dataloader
import pandas as pd
import numpy as np
import os
import inspect
# https://stackoverflow.com/questions/3718657/how-to-properly-determine-current-script-directory-in-python
filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))


def read_pickle(f):
    with open(f, "rb") as f:
        return pickle.load(f)


@kipoi_dataloader()
class MyDataset(Dataset):
    """
    args:
        features_file:
            doc: >
              Csv file of the Iris Plants Database from
              http://archive.ics.uci.edu/ml/datasets/Iris features.
            type: str
            example:
                url: https://github.com/kipoi/kipoi/raw/57734d716b8dedaffe460855e7cfe8f37ec2d48d/example/models/sklearn_iris/example_files/features.csv
                md5: 64d4930eda29aa6f240ca0241be1f2ed
        targets_file:
            doc: >
              Csv file of the Iris Plants Database targets.
              Not required for making the prediction.
            type: str
            example:
              url: https://github.com/kipoi/kipoi/raw/57734d716b8dedaffe460855e7cfe8f37ec2d48d/example/models/sklearn_iris/example_files/targets.csv
              md5: 54e058d31d05897836302cd6961212a1
            optional: True
        x_transformer:
            doc: input_transformer
            default:
              url: https://github.com/kipoi/kipoi/raw/57734d716b8dedaffe460855e7cfe8f37ec2d48d/example/models/sklearn_iris/dataloader_files/x_transformer.pkl
              md5: bc1bf3c61c418b2d07506a7d0521a893
        y_transformer:
            doc: input_transformer
            default: dataloader_files/y_transformer.pkl
        dummy:
            doc: dummy argument
            example: 5
    info:
        authors:
            - name: Your Name
              github: your_github_username
        doc: Model predicting the Iris species
    dependencies:
        conda: # directly install via conda
          - python=2.7
          - scikit-learn
          # - conda-forge::spacy  # use a special channel via <channel>::<package>
        pip:
           - tqdm  # pip packages
    output_schema:
        inputs:
          shape: (4,)
          doc: "Scaled features: sepal length, sepal width, petal length, petal width."
        targets:
          shape: (3, )
          doc: "One-hot encoded array of classes: setosa, versicolor, virginica."
        metadata:
            example_row_number:
                type: int
                doc: Just an example metadata column
    """

    def __init__(self, features_file, targets_file=None, x_transformer=None, y_transformer=None, dummy=None):
        self.features_file = features_file
        self.targets_file = targets_file
        self.dummy = dummy

        assert x_transformer is not None
        self.x_transformer = read_pickle(x_transformer)
        assert y_transformer is not None
        self.y_transformer = read_pickle(y_transformer)

        self.features = pd.read_csv(features_file)
        if targets_file is not None:
            self.targets = pd.read_csv(targets_file)
            assert len(self.targets) == len(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x_features = np.ravel(self.x_transformer.transform(self.features.iloc[idx].values[np.newaxis]))
        if self.targets_file is None:
            y_class = {}
        else:
            y_class = np.ravel(self.y_transformer.transform(self.targets.iloc[idx].values[np.newaxis]))
        return {
            "inputs": x_features,
            "targets": y_class,
            "metadata": {
                "example_row_number": idx
            }
        }
