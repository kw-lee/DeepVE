import torch
import numpy as np
import scipy.stats as stats


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


def mse(output, target):
    return(((output - target)**2).mean())


def Gaussian_ratio(output, target, stds, level=0.05):
    quant = -stats.norm.ppf(level/2)
    dist = np.abs(output - target)
    return((dist < quant*stds).mean())


def Laplace_ratio(output, target, stds, level=0.05):
    quant = -np.log(level/2)
    dist = np.abs(output - target)
    return((dist < quant*stds).mean())


def Cauchy_ratio(output, target, stds, level=0.05,):
    quant = -stats.t.ppf(level/2, 1)
    dist = np.abs(output - target)
    return((dist < quant*stds).mean())
