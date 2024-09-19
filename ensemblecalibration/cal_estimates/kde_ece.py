"""
Implementation of the Kernel Density Estimation method introduced by Popordanoska et al. (2022) to
estimate the Lp calibration error for canonical calibration. Code adapted from official github 
implementation: https://github.com/tpopordanoska/ece-kde
"""

import torch
import numpy as np
from torch import nn

from ensemblecalibration.utils.helpers import calculate_pbar

"""
own code
"""

def get_bandwidth(f):
    """
    Select a bandwidth for the kernel based on maximizing the leave-one-out likelihood (LOO MLE).

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :param device: The device type: 'cpu' or 'cuda'

    :return: The bandwidth of the kernel
    """
    bandwidths = torch.cat((torch.logspace(start=-5, end=-1, steps=15), torch.linspace(0.2, 1, steps=5)))
    max_b = -1
    max_l = 0
    n = len(f)
    for b in bandwidths:
        log_kern = get_kernel(f, b)
        log_fhat = torch.logsumexp(log_kern, 1) - torch.log(torch.tensor(n-1))
        l = torch.sum(log_fhat)
        if l > max_l:
            max_l = l
            max_b = b

    return max_b

def ece_kde_obj(p_bar: np.ndarray, y: np.ndarray, params: dict):

    # convert to torch tensors if necessary
    if not isinstance(p_bar, torch.Tensor):
        p_bar = torch.tensor(p_bar, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.int64)
    bw = params["bw"]
    p = params["p"]

    return get_ece_kde(p_bar, y, bw, p)  # TODO: check dimensions

def ece_kde_obj_lambda(weights_l, p_probs, y_labels, params, x_dep: bool = False):
    if x_dep:
        p_bar = calculate_pbar(weights_l, p_probs, reshape=True, n_dims=2)
    else:
        p_bar = calculate_pbar(weights_l, p_probs, reshape=False, n_dims=1)
    obj = ece_kde_obj(p_bar, y_labels, params)
    # convert to numpy array if needed
    if isinstance(obj, torch.Tensor):
        obj = obj.numpy()
    return obj

"""
Popordanoska et al. (2022) code
"""

def get_ece_kde(p_bar: torch.tensor, y: torch.tensor, bw: float, p: int = 2):
    """calculate estimate of the Lp calibration error.

    Parameters
    ----------
    p_bar :  torch.tensor of shape (n_samples, n_classes)
        vector containing probability scores for each sample
    y : torch.tensor
        vector containing labels for each sample
    bandwidth : float
        kernel bandwidth

    Returns
    -------
    torch.tensor
        estimate of the Lp calibration error
    """
    check_input(p_bar, bw)
    # check if input is binary
    if p_bar.shape[1] == 1:
        return 2 * get_ratio_binary(p_bar, y, bw, p)
    else:
        return get_ratio_canonical(f=p_bar, y=y, bandwidth=bw, p=p)


def get_ratio_binary(preds: torch.tensor, y: torch.tensor, p: int, bandwidth: float):
    assert preds.shape[1] == 1

    log_kern = get_kernel(preds, bandwidth)

    return get_kde_for_ece(preds, y, log_kern, p)


def get_kde_for_ece(f, y, log_kern, p):
    f = f.squeeze()
    N = len(f)
    # Select the entries where y = 1
    idx = torch.where(y == 1)[0]
    if not idx.numel():
        return torch.sum((torch.abs(-f)) ** p) / N

    if idx.numel() == 1:
        # because of -inf in the vector
        log_kern = torch.cat((log_kern[:idx], log_kern[idx + 1 :]))
        f_one = f[idx]
        f = torch.cat((f[:idx], f[idx + 1 :]))

    log_kern_y = torch.index_select(log_kern, 1, idx)

    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    ratio = torch.abs(ratio - f) ** p

    if idx.numel() == 1:
        return (ratio.sum() + f_one**p) / N

    return torch.mean(ratio)


def get_ratio_canonical(f, y, bandwidth, p):
    if f.shape[1] > 60:
        # Slower but more numerically stable implementation for larger number of classes
        return get_ratio_canonical_log(f, y, bandwidth, p)

    log_kern = get_kernel(f, bandwidth)
    if isnan(log_kern):
        print("log_kern is nan")
        print(f"nan values in log_kern: {torch.sum(torch.isnan(log_kern))}")
    kern = torch.exp(log_kern)
    kern = torch.clamp(kern, min=1e-20, max=1e35)
    if isnan(kern):
        print(f"nan values in kern: {torch.sum(torch.isnan(kern))}")
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    kern_y = torch.matmul(kern, y_onehot)
    # check if kern_y is nan
    if isnan(kern_y):
        #check how many nan values are in kern_y
        print(f"nan values in kern_y: {torch.sum(torch.isnan(kern_y))}")
    den = torch.sum(kern, dim=1)
    # to avoid division by 0
    #den = torch.clamp(den, min=1e-15, max=1e20)
    # check if den is nan
    if isnan(den):
        #check how many nan values are in den
        print(f"nan values in den: {torch.sum(torch.isnan(den))}")
    #kern_y = torch.clamp(kern_y, min=1e-10, max=1e10)

    ratio = kern_y / den.unsqueeze(-1)
    # check if ratio is nan
    if isnan(ratio):
        print("ratio is nan")
    ratio = torch.sum(torch.abs(ratio - f) ** p, dim=1)

    return torch.mean(ratio)


# Note for training: Make sure there are at least two examples for every class present in the batch, otherwise
# LogsumexpBackward returns nans.
def get_ratio_canonical_log(f, y, bandwidth, p):
    log_kern = get_kernel(f, bandwidth)
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    log_y = torch.log(y_onehot)
    log_den = torch.logsumexp(log_kern, dim=1)
    final_ratio = 0
    for k in range(f.shape[1]):
        log_kern_y = log_kern + (torch.ones([f.shape[0], 1]) * log_y[:, k].unsqueeze(0))
        log_inner_ratio = torch.logsumexp(log_kern_y, dim=1) - log_den
        inner_ratio = torch.exp(log_inner_ratio)
        inner_diff = torch.abs(inner_ratio - f[:, k]) ** p
        final_ratio += inner_diff

    return torch.mean(final_ratio)


def get_kernel(f, bandwidth):
    # if num_classes == 1
    if f.shape[1] == 1:
        log_kern = beta_kernel(f, f, bandwidth).squeeze()
    else:
        log_kern = dirichlet_kernel(f, bandwidth).squeeze()
    # Trick: -inf on the diagonal
    return log_kern + torch.diag(torch.finfo(torch.float).min * torch.ones(len(f)))


def beta_kernel(z, zi, bandwidth=0.1):
    p = zi / bandwidth + 1
    q = (1 - zi) / bandwidth + 1
    z = z.unsqueeze(-2)

    log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
    log_num = (p - 1) * torch.log(z) + (q - 1) * torch.log(1 - z)
    log_beta_pdf = log_num - log_beta

    return log_beta_pdf


def dirichlet_kernel(z, bandwidth=0.1):
    # add small value to avoid log of 0
    z = torch.clamp(z, min=1e-10, max=1.0 - 1e-10)
    alphas = z / bandwidth + 1
    log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(
        torch.sum(alphas, dim=1))
    )
    log_num = torch.matmul(torch.log(z), (alphas - 1).T)
    log_dir_pdf = log_num - log_beta

    return log_dir_pdf


def check_input(f, bandwidth,):
    assert not isnan(f)
    assert len(f.shape) == 2
    assert bandwidth > 0
    assert torch.min(f) >= 0
    assert torch.max(f) <= 1

def isnan(a):
    return torch.any(torch.isnan(a))