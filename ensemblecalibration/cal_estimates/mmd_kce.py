"""
Implementation of the kernel calibration error by distribution matching using maximum mean discrepancy. 
Code is adapted from 
https://github.com/kernel-calibration/kernel-calibration/blob/main/src/metrics/losses.py

    """

import torch
import numpy as np

import torch.nn.functional as F


def rbf_kernel(u: torch.Tensor, v: torch.Tensor, bandwidth=1):
    diff_norm_mat = torch.norm(u.unsqueeze(1) - v, dim=2).square()
    return torch.exp(-diff_norm_mat / bandwidth)


def mmd_kce(
    p_bar: torch.Tensor, y: torch.tensor, kernel_fct=rbf_kernel, sigma: float = 0.1
):
    """calculates the kenerl calibration error for the classification case using the maximum mean discrepancy.


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

    Returns
    -------
    torch.tensor
        kernel calibration error
    """

    kernel_out = None
    loss_mats = [None for i in range(3)]

    n_classes = p_bar.shape[1]
    y_all = torch.eye(n_classes).to(p_bar.device)
    k_yy = kernel_fct(y_all, y_all, sigma)
    q_yy = torch.einsum("ic,jd->ijcd", p_bar, p_bar)
    total_yy = q_yy * k_yy.unsqueeze(0)

    k_yj = k_yy[:, y].T
    total_yj = torch.einsum("ic,jc->ijc", p_bar, k_yj)
    y_one_hot = F.one_hot(y, num_classes=n_classes).float()

    loss_mat = total_yy.sum(dim=(2, 3))
    loss_mat2 = total_yj.sum(-1)
    loss_mat3 = kernel_fct(y_one_hot, y_one_hot, sigma)

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
