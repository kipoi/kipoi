"""
Dataloader
"""
from kipoi.data import Dataset
import pandas as pd
import numpy as np


class DummyDataset(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        return 42

    def __getitem__(self, idx):
        return dict(inputs=np.ones(64) * idx)
        