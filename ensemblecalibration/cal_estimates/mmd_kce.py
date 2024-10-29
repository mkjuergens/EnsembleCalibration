"""
Implementation of the kernel calibration error by distribution matching using maximum mean discrepancy. 
Code is adapted from 
https://github.com/kernel-calibration/kernel-calibration/blob/main/src/metrics/losses.py

    """

import torch
import numpy as np

import torch.nn.functional as F
from ensemblecalibration.utils.helpers import calculate_pbar


def rbf_kernel(u: torch.Tensor, v: torch.Tensor, bandwidth=1):
    diff_norm_mat = torch.norm(u.unsqueeze(1) - v, dim=2).square()
    return torch.exp(-diff_norm_mat / bandwidth)


def mmd_kce_obj(p_bar: np.ndarray, y: np.ndarray, params: dict, take_square: bool = False):
    if not isinstance(p_bar, torch.Tensor):
        p_bar = torch.tensor(p_bar, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.int64)
    bw = params["bw"]

    return mmd_kce(p_bar, y, kernel_fct=rbf_kernel, bw=bw, take_square=take_square)


def mmd_kce_obj_lambda(weights_l, p_probs, y_labels, params, x_dep: bool = False,
                        take_square: bool = False):
    if x_dep:
        p_bar = calculate_pbar(weights_l, p_probs, reshape=True, n_dims=2)
    else:
        p_bar = calculate_pbar(weights_l, p_probs, reshape=False, n_dims=1)

    obj = mmd_kce_obj(p_bar, y_labels, params, take_square=take_square)
    # convert to numpy array if needed
    if isinstance(obj, torch.Tensor):
        obj = obj.numpy()

    return obj


def mmd_kce(
    p_bar: torch.Tensor,
    y: torch.tensor,
    kernel_fct=rbf_kernel,
    bw: float = 0.1,
    take_square: bool = True,
):
    """calculates the kernel calibration error for the classification case using the maximum mean discrepancy.


    Parameters
    ----------
    p_bar : torch.Tensor
        tensor of predicted probabilities per class, shape (n_samples, n_classes)
    y : torch.tensor
        tensor of true labels, shape (n_samples,)
    kernel_fct : _type_
        kernel function to be used
    sigma : float
        bandwidth of the kernel
    take_square : bool
        whether to take the square of the kernel calibration error

    Returns
    -------
    torch.tensor
        kernel calibration error
    """

    kernel_out = None
    loss_mats = [None for i in range(3)]

    n_classes = p_bar.shape[1]
    y_all = torch.eye(n_classes).to(p_bar.device)
    k_yy = kernel_fct(y_all, y_all, bw)
    q_yy = torch.einsum("ic,jd->ijcd", p_bar, p_bar)
    total_yy = q_yy * k_yy.unsqueeze(0)

    k_yj = k_yy[:, y].T
    total_yj = torch.einsum("ic,jc->ijc", p_bar, k_yj)
    y_one_hot = F.one_hot(y, num_classes=n_classes).float()

    loss_mat = total_yy.sum(dim=(2, 3))
    loss_mat2 = total_yj.sum(-1)
    loss_mat3 = kernel_fct(y_one_hot, y_one_hot, bw)

    for i, value in enumerate([loss_mat, loss_mat2, loss_mat3]):
        if loss_mats[i] is None:
            loss_mats[i] = value
        else:
            loss_mats[i] = loss_mats[i] * value

    kernel_out = (
        mean_no_diag(loss_mats[0])
        - 2 * mean_no_diag(loss_mats[1])
        + mean_no_diag(loss_mats[2])
    )
    if take_square:
        kernel_out = torch.sqrt(kernel_out ** 2 + 1e-8)

    return kernel_out


def mean_no_diag(A):
    """helper function for the mmd_kce function.
    Calculates the mean of the off-diagonal elements of a matrix.

    Parameters
    ----------
    A : torch.Tensor
        tensor of shape (n, n)

    Returns
    -------
    float
        mean of the off-diagonal elements of the matrix
    """
    # make sure A is a square matrix
    assert A.dim() == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    # dont sum over diagonal elements (set them to zero first)
    A = A - torch.eye(n).to(A.device) * A.diag()
    return A.sum() / (n * (n - 1))


def tanh_kernel(u: torch.Tensor, v: torch.Tensor, bandwidth=1):
    out = torch.tanh(v) * torch.tanh(u).unsqueeze(1)  # N x N x 1 x num_samples
    return out.squeeze(2)
