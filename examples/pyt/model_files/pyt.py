import torch
import numpy as np
from tqdm import tqdm


class Flatten(torch.nn.Module):
    # https://gist.github.com/VoVAllen/5531c78a2d3f1ff3df772038bca37a83

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def get_model():
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    D_in, H, D_out = 4000, 100, 1

    model = torch.nn.Sequential(
        Flatten(),
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid(),
    )
    return model


def generate_exmaple_model():
    # get model
    model = get_model()

    # define loss function
    loss_func = torch.nn.MSELoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    minibatch_size = 10
    np.random.seed(0)
    x = torch.Tensor(50, 1000, 4).uniform_(0, 1)
    y = torch.Tensor(50).uniform_(0, 1)

    for epoch in tqdm(range(10)):
        for mbi in tqdm(range(np.ceil(x.size()[0] / minibatch_size).astype(int))):
            minibatch = x[(mbi * minibatch_size):min(((mbi + 1) * minibatch_size), x.size()[0])]
            target = torch.autograd.Variable(y[(mbi * minibatch_size):min(((mbi + 1) * minibatch_size), x.size()[0])])
            model.zero_grad()

            # forward pass
            out = model(torch.autograd.Variable(minibatch))

            # backward pass
            L = loss_func(out, target)  # calculate loss
            L.backward()  # calculate gradients
            optimizer.step()  # make an update step

    torch.save(model, "model_files/full_model.pth")
    torch.save(model.state_dict(), "model_files/only_weights.pth")


def get_model_w_weights():
    model = get_model()
    model.load_state_dict(torch.load("model_files/only_weights.pth"))
    return model


def test_same_weights(dict1, dict2):
    for k in dict1:
        assert np.all(dict1[k].numpy() == dict2[k].numpy())

# test_same_weights(model.state_dict(), model_2.state_dict())
