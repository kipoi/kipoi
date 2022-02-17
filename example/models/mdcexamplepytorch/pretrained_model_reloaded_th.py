
import torch
import torch.nn as nn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

def get_model(load_weights = True):
    # alphabet seems to be fine:
    """
    https://github.com/davek44/Basset/tree/master/src/dna_io.py#L145-L148
    seq = seq.replace('A','0')
    seq = seq.replace('C','1')
    seq = seq.replace('G','2')
    seq = seq.replace('T','3')
    """
    pretrained_model_reloaded_th = nn.Sequential( # Sequential,
        nn.Conv2d(4,300,(19, 1)),
        nn.BatchNorm2d(300),
        nn.ReLU(),
        nn.MaxPool2d((3, 1),(3, 1)),
        nn.Conv2d(300,200,(11, 1)),
        nn.BatchNorm2d(200),
        nn.ReLU(),
        nn.MaxPool2d((4, 1),(4, 1)),
        nn.Conv2d(200,200,(7, 1)),
        nn.BatchNorm2d(200),
        nn.ReLU(),
        nn.MaxPool2d((4, 1),(4, 1)),
        Lambda(lambda x: x.view(x.size(0),-1)), # Reshape,
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2000,1000)), # Linear,
        nn.BatchNorm1d(1000,1e-05,0.1,True),#BatchNorm1d,
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1000,1000)), # Linear,
        nn.BatchNorm1d(1000,1e-05,0.1,True),#BatchNorm1d,
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(1000,164)), # Linear,
        nn.Sigmoid(),
    )
    if load_weights:
        sd = torch.load('model_files/pretrained_model_reloaded_th.pth')
        pretrained_model_reloaded_th.load_state_dict(sd)
    return  pretrained_model_reloaded_th

model = get_model(load_weights = False)





