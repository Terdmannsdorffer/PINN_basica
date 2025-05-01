import torch
import torch.nn as nn

class DeepPINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 64, 3]):
        super(DeepPINN, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []

        for i in range(len(layers) - 2):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
            layer_list.append(self.activation)

        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)
