from tqdm import tqdm
from .utils import to_device
import torch


# train_homo
def train_homo(model, x_train, y_train, num_epochs,
               optimizer, loss_func,
               device=torch.device('cpu')):
    output_dim = model.output_dim
    if not (output_dim == y_train.shape[1]):
        raise ValueError

    for i in tqdm(range(num_epochs), position=0):
        x, y = to_device(x_train, y_train,
                         cuda=(device.type == 'cuda'))
        # reset gradient and total loss
        optimizer.zero_grad()
        output = model(x)
        loss_homo = loss_func(output, y, model.log_sigma.exp(),
                              no_dim=output_dim)

        loss_homo.backward()
        optimizer.step()

        if i % (num_epochs/5) == 0:
            print("\nEpoch: {:4d}, Train loss = {:7.3f}".format(
                i, loss_homo.cpu().data.numpy()))


# todo: `train_hetero`
def train_hetero(model, x_train, y_train, num_epochs,
                 optimizer, loss_func, device=torch.device('cpu')):
    output_dim = model.output_dim
    if not (output_dim == y_train.shape[1]):
        raise ValueError

    for i in tqdm(range(num_epochs), position=0):
        x, y = to_device(x_train, y_train,
                         cuda=(device.type == 'cuda'))
        # reset gradient and total loss
        optimizer.zero_grad()
        output = model(x)
        loss_hetero = loss_func(output[:, :1], y,
                                output[:, 1:].exp(),
                                no_dim=output_dim)

        loss_hetero.backward()
        optimizer.step()

        if i % (num_epochs/5) == 0:
            print("\nEpoch: {:4d}, Train loss = {:7.3f}".format(
                i, loss_hetero.cpu().data.numpy()))
