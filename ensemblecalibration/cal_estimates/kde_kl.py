"""
Implementation of the Kernel Density based calibration error based on KL-divergence,
introduced by Popordanoska et al. (2024) to
estimate the KL calibration error for canonical calibration. Code adapted from official github 
implementation:
https://github.com/tpopordanoska/proper-calibration-error
"""

import torch
import numpy as np
from torch import nn

from .kde_ece import get_kernel, check_input
from ensemblecalibration.utils.helpers import calculate_pbar


"""
own code for KL calibration error
"""

def kl_kde_obj(p_bar: np.ndarray, y: np.ndarray, params: dict):

    if not isinstance(p_bar, torch.Tensor):
        p_bar = torch.tensor(p_bar, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.int64)
    bw = params["bw"]

    kl_kde = get_bregman_ce(negative_entropy, p_bar, y, bandwidth=bw)
    return kl_kde


def kl_kde_obj_lambda(weights_l, p_probs, y_labels, params, x_dep: bool = False):
    if x_dep:
        p_bar = calculate_pbar(weights_l, p_probs, reshape=True, n_dims=2)
    else:
        p_bar = calculate_pbar(weights_l, p_probs, reshape=False, n_dims=1)

    kl_kde = kl_kde_obj(p_bar, y_labels, params)
    if isinstance(kl_kde, torch.Tensor):
        kl_kde = kl_kde.detach().numpy()

    return kl_kde

def get_kl_kde(p_bar: torch.tensor, y: torch.tensor, bw: float, device: str = "cpu"):
    # print("Does p_bar require grad?", p_bar.requires_grad)
    # check if p_bar contains nans:
    print(p_bar)
    if _isnan(p_bar):
        raise ValueError("p_bar contains nans")
    kl_kde = get_bregman_ce(negative_entropy, p_bar, y, bandwidth=bw, device=device)
    return kl_kde


"""
Code adapted from official github implementation, Popordanoska et al. (2024)
"""


def get_bregman_ce(convex_fcn, gx, y, bandwidth, device="cpu"):
    """
    Calculate an estimate of Bregman calibration error.

    Args:
        convex_fcn: A strictly convex function F
        gx: The vector containing the probability scores, shape [num_samples, num_classes]
        y: The vector containing the labels, shape [num_samples]
        bandwidth: The bandwidth of the kernel
        device: The device type: 'cpu', 'cuda', 'mps'

    Returns: Bregman/proper calibration error characterized by the given convex function.

    """
    check_input(gx, bandwidth=bandwidth)
    gx = _convert_to_device_and_dtype(gx, device)
    #gx = gx.clone().detach().requires_grad_()
    if not gx.requires_grad:
        gx.requires_grad = True
    ratio = _get_ratio(gx, y, bandwidth, device)
    #check if ratio is nan
    if _isnan(ratio):
        raise ValueError("Ratio contains nans")
    ratio = torch.clamp(ratio, min=1e-45)
    f_ratio = convex_fcn(ratio)
    # check if f_ratio is nan
    if _isnan(f_ratio):
        raise ValueError("f_ratio contains nans")
    f_gx = convex_fcn(gx)
    # check if f_gx is nan
    if _isnan(f_gx):
        raise ValueError("f_gx contains nans")
    # check if f_gx, gx requires grad
    if not gx.requires_grad:
        raise ValueError("gx does not require grad")
    if not f_gx.requires_grad:
        raise ValueError("f_gx does not require grad")
    grad_f_gx = torch.autograd.grad(
        f_gx, gx, grad_outputs=torch.ones_like(f_gx), retain_graph=True
    )[0]
    # check if grad_f_gx is nan
    if _isnan(grad_f_gx):
        raise ValueError("grad_f_gx contains nans")
    diff_ratio_gx = ratio - gx
    dot_prod = torch.sum(grad_f_gx * diff_ratio_gx, dim=1)
    # check if dot_prod is nan
    if _isnan(dot_prod):
        raise ValueError("dot_prod contains nans")
    CE = torch.mean(f_ratio - f_gx - dot_prod)

    _check_output(CE, "CE")
    return CE


def get_bandwidth(gx, device):
    """
    Select a bandwidth for the kernel based on maximizing the leave-one-out likelihood (LOO MLE).

    Args:
        gx: The vector containing the probability scores, shape [num_samples, num_classes]
        device: The device type: 'cpu' or 'cuda'

    Returns: The bandwidth of the kernel/
    """
    bandwidths = torch.cat(
        (torch.logspace(start=-5, end=-1, steps=50), torch.linspace(0.2, 1, steps=5))
    )
    max_b = -1
    max_l = torch.finfo(torch.float).min
    n = len(gx)
    for b in bandwidths:
        log_kern = get_kernel(gx, b, device)
        log_fhat = torch.logsumexp(log_kern, 1) - torch.log(torch.tensor(n - 1))
        l = torch.sum(log_fhat)
        if l > max_l:
            max_l = l
            max_b = b

    return max_b


def _check_output(out, name=""):
    assert not _isnan(out), f"{name} contains nans"


def _isnan(a):
    return torch.any(torch.isnan(a))


def _get_ratio(f, y, bandwidth, device="cpu"):
    if f.shape[1] > 20:
        # Slower but more numerically stable implementation for larger number of classes
        return _get_ratio_iter(f, y, bandwidth, device)

    log_kern = get_kernel(f, bandwidth, device)

    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1])
    y_onehot = _convert_to_device_and_dtype(y_onehot, device)
    # matrix multiplication in log space using broadcasting
    log_kern_y = torch.logsumexp(
        log_kern.unsqueeze(2) + torch.log(y_onehot).unsqueeze(0), dim=1
    )
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_kern_y - log_den.unsqueeze(-1)
    ratio = torch.exp(log_ratio)

    _check_output(ratio, "ratio")
    return ratio


def _get_ratio_iter(f, y, bandwidth, device="cpu"):
    log_kern = get_kernel(f, bandwidth, device)
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1])
    y_onehot = _convert_to_device_and_dtype(y_onehot, device)
    log_y = torch.log(y_onehot)
    log_den = torch.logsumexp(log_kern, dim=1)
    final_ratio = []
    for k in range(f.shape[1]):
        log_kern_y = log_kern + (
            torch.ones([f.shape[0], 1]).to(device) * log_y[:, k].unsqueeze(0)
        )
        log_inner_ratio = torch.logsumexp(log_kern_y, dim=1) - log_den
        inner_ratio = torch.exp(log_inner_ratio)
        final_ratio.append(inner_ratio)

    return torch.transpose(torch.stack(final_ratio), 0, 1)


def _convert_to_device_and_dtype(x, device):
    # mps does not support double
    if device == "mps":
        return x.float().to(device)
    # for cpu and cuda, convert to double precision
    else:
        if x.dtype != torch.double:
            x = x.double().to(device)
    return x


def negative_entropy(x):
    # ensure that x is not zero
    x = torch.clamp(x, min=1e-10)
    neg_entropy = torch.sum(x * torch.log(x), dim=1)
    neg_entropy = torch.where(torch.isnan(neg_entropy),
                          torch.zeros_like(neg_entropy),
                          neg_entropy)


    return neg_entropy
