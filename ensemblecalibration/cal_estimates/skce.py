import numpy as np
import torch
from scipy.stats import norm

from ensemblecalibration.utils.distances import tv_distance
from ensemblecalibration.utils.helpers import calculate_pbar


def skce_obj(p_bar: torch.Tensor, y: torch.Tensor, params: dict, take_square: bool = False):

    # convert to torch tensors if necessary
    if not isinstance(p_bar, torch.Tensor):
        p_bar = torch.tensor(p_bar, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.int64)
    bw = params["bw"]

    return get_skce_ul(p_bar, y, dist_fct=tv_distance, bw=bw, take_square=take_square)


def skce_obj_lambda(weights_l, p_probs, y_labels, params, x_dep: bool = False, take_square: bool = False):
    if x_dep:
        p_bar = calculate_pbar(weights_l, p_probs, reshape=True, n_dims=2)
    else:
        p_bar = calculate_pbar(weights_l, p_probs, reshape=False, n_dims=1)
    obj = skce_obj(p_bar, y_labels, params, take_square=take_square)
    # convert to numpy array if needed
    if isinstance(obj, torch.Tensor):
        obj = obj.numpy()
    return obj


def get_skce_ul(
    p_bar: torch.Tensor,
    y: torch.Tensor,
    dist_fct=tv_distance,
    bw: float = 2.0,
    take_square: bool = True,
):
    skce_ul_stats = skce_ul_tensor(p_bar, y, dist_fct=dist_fct, bw=bw)
    # replace each entry with its L2 norm
    #skce_ul_stats = torch.norm(skce_ul_stats, p=2, dim=0)

    skce_ul = torch.mean(skce_ul_stats)  # sum instead of mean ?
    if take_square:
        skce_ul = torch.sqrt(skce_ul**2)  # take square root of the sum
    return skce_ul


def skce_ul_tensor(p_bar, y, dist_fct=tv_distance, bw=2.0):
    n = int(p_bar.shape[0] / 2)
    # One-hot encode labels
    yoh = torch.nn.functional.one_hot(y, num_classes=p_bar.shape[1]).float()
    # put it on the same device as p_bar
    yoh = yoh.to(p_bar.device)

    # Extract even and odd indices for p_i and p_j
    p_i = p_bar[0::2]
    p_j = p_bar[1::2]
    y_i = yoh[0::2]
    y_j = yoh[1::2]

    # Compute distances between p_i and p_j
    dist = dist_fct(p_i, p_j)  # Shape: (n,)

    # Compute gamma
    gamma = torch.exp(-(dist**2) / bw)  # Shape: (n,)

    # Compute differences y_i - p_i and y_j - p_j
    y_diff_i = y_i - p_i  # Shape: (n, n_classes)
    y_diff_j = y_j - p_j  # Shape: (n, n_classes)

    # Compute dot products between corresponding rows
    dot_products = torch.sum(y_diff_i * y_diff_j, dim=1)  # Shape: (n,)

    # Compute h_ij
    h_ij = gamma * dot_products  # Shape: (n,)

    return h_ij


# OLD CODE


def get_skce_uq(
    p_bar: torch.Tensor, y: torch.Tensor, dist_fct=tv_distance, sigma: float = 2.0
):
    skce_uq_stats = skce_uq_tensor(p_bar, y, dist_fct=dist_fct, bw=sigma)
    skce_uq = torch.mean(skce_uq_stats)
    return skce_uq


def skce_ul_tensor_old(
    p_bar: torch.Tensor, y: torch.Tensor, dist_fct=tv_distance, bw: float = 2.0
):
    """calculates the skce_ul calibration error used as a test statistic in Mortier et  al, 2022.

    Parameters
    ----------
    P_bar :  torch.Tensor of shape (n_predictors, n_classes)
        matrix containing probabilistic predictions for each instance
    y : torch.Tensor
        vector with class labels of shape
    dist_fct : [tv_distance, l2_distance]
        distance function to be used
    sigma : float
        bandwidth used in the matrix valued kernel

    Returns
    -------
    torch.Tensor
        _description_
    """
    # bugfix: round down instead of up
    n = int(p_bar.shape[0] / 2)
    # transform y to one-hot encoded labels
    yoh = torch.eye(p_bar.shape[1])[y, :]
    stats = torch.zeros(n)
    for i in range(0, n):
        stats[i] = tensor_h_ij(
            p_bar[(2 * i), :],
            p_bar[(2 * i) + 1, :],
            yoh[(2 * i), :],
            yoh[(2 * i) + 1, :],
            dist_fct=dist_fct,
            sigma=bw,
        )

    return stats


def skce_uq_tensor(
    p_bar: torch.Tensor, y: torch.Tensor, dist_fct=tv_distance, bw: float = 2.0
):
    """calculates the SKCEuq miscalibration measure introduced in Widman et al, 2019.

    Parameters
    ----------
    p_bar : torch.Tensor
        tensor containing all
    y : torch.Tensor
        _description_
    dist_fct : _type_, optional
        _description_, by default tv_distance_tensor
    sigma : float, optional
        _description_, by default 2.0

    Returns
    -------
    _type_
        _description_
    """

    N, M = (
        p_bar.shape[0],
        p_bar.shape[1],
    )  # p is of shape (n_samples, m_predictors, n_classes)
    # one-hot encoding
    y_one_hot = torch.eye(M)[y, :]

    # binomial coefficient n over 2
    stats = torch.zeros(int((N * (N - 1)) / 2))
    count = 0
    for j in range(1, N):
        for i in range(j):
            stats[count] = tensor_h_ij(
                p_bar[i, :],
                p_bar[j, :],
                y_one_hot[i, :],
                y_one_hot[j, :],
                dist_fct=dist_fct,
                sigma=bw,
            )
            count += 1

    return stats


def tensor_h_ij(
    p_i: torch.Tensor,
    p_j: torch.Tensor,
    y_i: torch.Tensor,
    y_j: torch.Tensor,
    dist_fct,
    sigma: float = 2.0,
):
    """calculates the entries h_ij which are summed over in the expression of the calibration

    Parameters
    ----------
    p_i : torch.Tensor

        first point prediction
    p_j : torch.Tensor
        second point prediction
    y_i : torch.Tensor
        one hot encoding of labels for sample j
    y_j : torch.Tensor
        one hot encoding of labels for sample j
    dist_fct :
        function used as a distance measure in the matrix valued kernel
    sigma : float, optional
        bandwidth, by default 2.0

    Returns
    -------
    torch.Tensor

    """
    gamma_ij = tensor_kernel(p_i, p_j, dist_fct=dist_fct, sigma=sigma).float()
    y_ii = (y_i - p_i).float()
    y_jj = (y_j - p_j).float()
    h_ij = torch.matmul(y_ii, torch.matmul(gamma_ij, y_jj))
    return h_ij


def tensor_kernel(
    p: torch.Tensor, q: torch.Tensor, dist_fct=tv_distance, sigma: float = 2.0
):
    """returns the matrix-valued kernel evaluated at two point predictions

    Parameters
    ----------
    p : torch.Tensor
        first point prediction
    q : torch.Tensor
        second point prediction
    sigma : float
        bandwidth
    dist_fct : _type_
        distance measure. Options: {tv_distance, l2_distance}

    Returns
    -------
    torch.Tensor
        matirx valued kernnel evaluated at
    """
    p = p.squeeze()
    q = q.squeeze()

    assert len(p) == len(q), "vectors need to be of the same length"
    # put it to device
    id_k = torch.eye(len(p)).to(p.device)
    return torch.exp((-1 / sigma) * (dist_fct(p, q) ** 2)) * id_k
