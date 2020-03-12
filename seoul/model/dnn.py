import torch
import torch.nn as nn


class HeteroModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        """
        Heteroscedastic model
        output: mean and log-sigma
        """
        super(HeteroModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(input_dim, num_units)
        self.layer_mean = nn.Linear(num_units, output_dim)
        self.layer_sd = nn.Linear(num_units, output_dim)
        torch.nn.init.uniform_(self.layer_sd.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.layer_sd.bias, -1e-3, 1e-3)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x = x.view(-1, self.input_dim)
        x = self.layer1(x)
        x = self.activation(x)
        mean_x = self.layer_mean(x)
        sd_x = self.layer_sd(x)
        out = torch.cat((mean_x, sd_x), 1)
        return out


class HomoModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_units):
        """
        Homoscedastic model
        output: mean
        model.log_sigma: log-sigma
        """
        super(HomoModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # network with two hidden and one output layer
        # output layer: mean
        self.layer1 = nn.Linear(input_dim, num_units)
        self.layer2 = nn.Linear(num_units, output_dim)
        self.activation = nn.ReLU()
        self.log_sigma = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x):
        # x = x.view(-1, self.input_dim)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x
