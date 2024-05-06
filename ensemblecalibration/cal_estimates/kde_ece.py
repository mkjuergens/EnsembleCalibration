"""
Implementation of the Kernel Density Estimation method introduced by Popordanoska et al. (2022) to
estimate the Lp calibration error for canonical calibration. Code adapted from official github 
implementation: https://github.com/tpopordanoska/ece-kde
"""

import torch
from torch import nn 

def get_ece_kde(p_bar: torch.tensor, y: torch.tensor, bandwidth: float, p: int = 2):
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

    # check if input is binary
    if p_bar.shape[1] == 1:
        return 2 * get_ratio_binary(p_bar, y, p,  bandwidth)
    else:
        return get_ratio_canonical(p_bar, y, p, bandwidth)
    
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
        return torch.sum((torch.abs(-f))**p) / N

    if idx.numel() == 1:
        # because of -inf in the vector
        log_kern = torch.cat((log_kern[:idx], log_kern[idx+1:]))
        f_one = f[idx]
        f = torch.cat((f[:idx], f[idx+1:]))

    log_kern_y = torch.index_select(log_kern, 1, idx)

    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    ratio = torch.abs(ratio - f)**p

    if idx.numel() == 1:
        return (ratio.sum() + f_one ** p)/N

    return torch.mean(ratio)

def get_ratio_canonical(f, y, bandwidth, p):
    if f.shape[1] > 60:
        # Slower but more numerically stable implementation for larger number of classes
        return get_ratio_canonical_log(f, y, bandwidth, p)

    log_kern = get_kernel(f, bandwidth)
    kern = torch.exp(log_kern)

    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    kern_y = torch.matmul(kern, y_onehot)
    den = torch.sum(kern, dim=1)
    # to avoid division by 0
    den = torch.clamp(den, min=1e-10)

    ratio = kern_y / den.unsqueeze(-1)
    ratio = torch.sum(torch.abs(ratio - f)**p, dim=1)

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
        inner_diff = torch.abs(inner_ratio - f[:, k])**p
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
    q = (1-zi) / bandwidth + 1
    z = z.unsqueeze(-2)

    log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
    log_num = (p-1) * torch.log(z) + (q-1) * torch.log(1-z)
    log_beta_pdf = log_num - log_beta

    return log_beta_pdf


def dirichlet_kernel(z, bandwidth=0.1):
    alphas = z / bandwidth + 1

    log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(torch.sum(alphas, dim=1)))
    log_num = torch.matmul(torch.log(z), (alphas-1).T)
    log_dir_pdf = log_num - log_beta

    return log_dir_pdf