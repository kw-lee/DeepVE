import numpy as np
import torch
import functools


def log_gaussian_loss(output, target, sigma, no_dim):
    """
    negative log-likelihood of N(output; target, sigma^2)
    """
    exponent = -0.5*(target - output)**2/sigma**2
    # exponent = -0.5*(target - output)**2
    log_coeff = -no_dim*torch.log(sigma) - 0.5*no_dim*np.log(2*np.pi)
    return (-exponent-log_coeff).mean()


def log_laplace_loss(output, target, sigma, no_dim):
    """
    negative log-likelihood of DE(output; target, sigma)
    """
    exponent = -torch.abs(target - output)/sigma
    log_coeff = -no_dim*torch.log(2*sigma)
    return (-exponent - log_coeff).mean()


def log_t_loss(output, target, sigma, no_dim, df=1.0):
    """
    negative log-likelihood of t_df(output; target, sigma)
    Note log_t_loss is
        Cauchy when df=1;
        Normal when df=infty
    """
    stand_out = (output - target)/sigma
    df_torch = torch.tensor(df)
    exponent = -(df+1)*0.5*(1 + stand_out**2/df_torch)
    nlog_coeff = no_dim*(0.5*torch.log(torch.tensor(df_torch*np.pi))
                         + torch.log(sigma)
                         + torch.lgamma(df_torch/2) 
                         - torch.lgamma((df_torch+1)/2))
    return (-exponent + nlog_coeff).mean()


def log_lgaussian_loss(output, target, sigma, no_dim):
    """
    negative log-likelihood of log_normal(output; target, sigma)
    suppose log(output) ~ N(log(target), sigma^2)
    """
    log_out = torch.log(output)
    log_target = torch.log(target)
    return (-log_out.mean()
            - log_gaussian_loss(log_out, log_target, sigma, no_dim))


# todo: define function which generate loss
def log_likelihood(likelihood, **kwargs):
    if (likelihood == "gaussian"):
        return functools.partial(log_gaussian_loss, **kwargs)
    elif (likelihood == "laplace"):
        return functools.partial(log_laplace_loss, **kwargs)
    elif (likelihood == "t"):
        return functools.partial(log_t_loss, **kwargs)
    else:
        return 0
