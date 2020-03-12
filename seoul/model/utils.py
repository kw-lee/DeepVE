import torch
import numpy as np


def to_device(*args, cuda=True):
    """
    change device of variables
    """
    var = []
    for x in args:
        if isinstance(x, np.ndarray):  # numpy to torch
            x = torch.from_numpy(x).type(torch.FloatTensor)
        if not x.is_cuda and cuda:
            x = x.cuda()
        var.append(x)
    return var


# todo: define plot function (homo vs hetero => isinstance)
def plot_model():
    return 0
