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

    def pred(self, newx):
        """
        prediction at newx
        output: means and stds
        """
        self.eval()
        out = self(newx)
        means = out[:, 0]
        stds = torch.exp(out[:, 1])
        self.train()
        return (means, stds)


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

    def pred(self, newx):
        """
        prediction at newx
        output: means and stds
        """
        self.eval()
        out = self(newx)
        means = out[:, 0]
        stds = torch.exp(self.log_sigma).repeat(means.shape[0])
        self.train()
        return (means, stds)


# todo: RNN model
class HeteroRNNModel(nn.Module):
    """
    Heteroscedastic RNN model
    output: mean and log-sigma
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_units):
        super(HeteroRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_units = num_units
        self.i2h = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.i2o = nn.Lienar(input_dim + hidden_dim, num_units)
        self.o2o_mean = nn.Linear(num_units, output_dim)
        self.o2o_sd = nn.Linear(num_units, output_dim)
        torch.nn.init.uniform_(self.i2o_sd.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.i2o_sd.bias, -1e-3, 1e-3)
        self.activation = nn.ReLU()

    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = self.i2h(combined)
        origin_out = self.i2o(combined)
        origin_out = self.activation(origin_out)
        out_mean = self.o2o_mean(origin_out)
        out_sd = self.o2o_sd(origin_out)
        out = torch.cat((out_mean, out_sd), 1)
        return out, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_dim)

    def pred(self, newx):
        """
        prediction at newx
        output: means and stds
        """
        self.eval()
        out, _ = self(newx)
        means = out[:, 0]
        stds = torch.exp(out[:, 1])
        self.train()
        return (means, stds)


# todo: RNN model
class HomoRNNModel(nn.Module):
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

    def pred(self, newx):
        """
        prediction at newx
        output: means and stds
        """
        self.eval()
        out = self(newx)
        means = out[:, 0]
        stds = torch.exp(self.log_sigma).repeat(means.shape[0])
        self.train()
        return (means, stds)
