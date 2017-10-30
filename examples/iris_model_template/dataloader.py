"""Dataloader
"""
from kipoi.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer


class MyDataset(Dataset):

    def __init__(self, features_file, targets_file=None):
        """

        """
        self.features_file = features_file
        self.targets_file = targets_file

        self.features = pd.read_csv(features_file)
        if targets_file is not None:
            self.targets = pd.read_csv(targets_file)
            assert len(self.targets) == len(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        out = {}
        # inputs
        out["inputs"] = {
            "features": self.features.iloc[idx].values
        }

        # optional targets
        if self.targets_file is not None:
            out["targets"] = {
                "plant_class": self.targets.iloc[idx].values
            }

        # metadata
        out["metadata"] = {
            "example_row_number": np.array([idx])
        }
        return out
