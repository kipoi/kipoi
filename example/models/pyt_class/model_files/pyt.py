import torch
from torch import nn, sigmoid
import torch.nn.functional as F


class PyTNet(nn.Module):
    def __init__(self, a, b, c):
        super(PyTNet, self).__init__()
        assert int(a) == 1
        assert int(b) == 2
        assert int(c) == 3
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(240, 2)
        self.fc2 = nn.Linear(2, 1)
    def forward(self, x):
        x1 = self.pool(F.relu(self.conv1(x)))
        x2 = self.pool(sigmoid(self.conv1(x)))
        x1 = x1.view(-1, 240)
        x2 = x2.view(-1, 240)
        x = torch.cat((x1, x2), 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
