import torch
import torch.nn as nn

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(1 * input)

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list, activation_function):
        super().__init__()
        # activation_function
        if activation_function == 'relu':
            activation_function = nn.ReLU()
        elif activation_function == 'siren':
            activation_function = Sine()
        else:
            NotImplementedError(activation_function + 'is not implemented')
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            # layers.append(nn.BatchNorm1d(hidden))
            layers.append(activation_function)
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)