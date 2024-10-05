import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from numpy import random
import torch

from ensemblecalibration.utils.helpers import (
    calculate_pbar,
    multinomial_label_sampling,
    sample_function,
    ab_scale,
)


def exp_gp(
    n_samples: int,
    x_bound: list = [0.0, 5.0],
    kernel=rbf_kernel,
    bounds_p: list = [[0.5, 0.7], [0.6, 0.8]],
    h0: bool = True,
    x_dep: bool = True,
    deg: int = 2,
    setting: int = 1,
    **kwargs,
):
    """Experiment for generating probabilistic predictions of binary classifiers."""

    x_inst = np.random.uniform(*x_bound, n_samples)
    p_preds = sample_binary_preds_gp(x_inst, kernel, bounds_p, **kwargs)

    p_bar, weights_l = (
        sample_pbar_h0(x_inst, p_preds, x_dep, deg)
        if h0
        else (sample_pbar_h1(x_inst, p_preds, kernel, setting, **kwargs), None)
    )

    y_labels = multinomial_label_sampling(p_bar, tensor=True).view(-1)

    x_inst = torch.from_numpy(x_inst).float().view(-1, 1)

    return (
        (x_inst, p_preds, p_bar, y_labels, weights_l)
        if h0
        else (x_inst, p_preds, p_bar, y_labels)
    )


def exp_gp_old(
    n_samples: int,
    x_bound: list = [0.0, 5.0],
    kernel=rbf_kernel,
    bounds_p: list = [[0.4, 0.5], [0.6, 0.8]],
    h0: bool = True,
    x_dep: bool = True,
    deg: int = 2,
    setting: int = 1,
    **kwargs,
):
    """experiment for generating probabilistic predictions of binary classifiers,
    as well as either a convex combination of these predictions or a sample from
    a GP outside of the convex hull.

    Parameters
    ----------
    n_samples : int
        number of instances to sample
    x_bound : list, optional
        range of the instance values, by default [0.0, 5.0]
    kernel : sklearn.metrics.pairwise, optional
        kernel used in the Gaussian processes, by default rbf_kernel
    bounds_p : list, optional
        list of lower and upper values for the probabilistic predictions of the individual
        (ensemble) members, by default [[0, 0.2], [0.5, 0.7]]
    h0 : bool, optional
        whether the null hypothesis is true, i.e. the convex combination lies in the convex hull,
        by default True
    x_dep : bool, optional
        whether the convex combination is a function dependent on the instance value, by default
          True
    deg : int, optional
        degree of the function in the case of h0==True and x_dep==True, by default 2
    s_1 : bool, optional
        whether the GP sample is close to the boundary of the probability simplex, by default False
    **kwargs : dict
        additional arguments for the kernel

    Returns
    -------
    if h0:
        x_inst, p_preds, p_bar, y_labels, weights_l
            instance values, probabilistic predictions, convex combination, labels, weights of cc
              and labels
    else:
        x_inst, p_preds, p_bar, y_labels
            instance values, probabilistic predictions, cc, labels
    """
    # sample instances uniformly
    x_inst = np.random.uniform(x_bound[0], x_bound[1], n_samples)
    # sample predictions
    p_preds = sample_binary_preds_gp(x_inst, kernel, bounds_p, **kwargs)

    if h0:
        p_bar, weights_l = sample_pbar_h0(x_inst, p_preds, x_dep, deg)
    else:
        p_bar = sample_pbar_h1(x_inst, p_preds, kernel, setting=setting, **kwargs)

    # now sample labels from categorical distributiuon induced by p_bar_h0
    y_labels = torch.stack(
        [multinomial_label_sampling(p, tensor=True) for p in torch.unbind(p_bar, dim=0)]
    )
    x_inst = torch.from_numpy(x_inst).float().view(-1, 1)
    y_labels = y_labels.view(-1)

    return (
        (x_inst, p_preds, p_bar, y_labels, weights_l)
        if h0
        else (x_inst, p_preds, p_bar, y_labels)
    )


def sample_pbar_h0(x, p_preds, x_dep=False, deg=2):
    """Sample convex combination within the convex hull."""

    n_samples, n_preds = len(x), p_preds.shape[1]
    l_weights = (
        sample_function(x, deg=deg) if x_dep else np.random.rand(1).repeat(n_samples)
    )
    weights_l = np.column_stack((l_weights, 1 - l_weights))
    weights_l = torch.from_numpy(weights_l).float()

    p_bar = calculate_pbar(weights_l, p_preds, reshape=False)
    return p_bar, weights_l


def sample_pbar_h1(x, p_preds, kernel, setting=1, eps=1e-4, **kwargs):
    """Sample outside the convex hull."""

    p_preds = (
        torch.from_numpy(p_preds).float()
        if not isinstance(p_preds, torch.Tensor)
        else p_preds
    )
    p_preds_min, p_preds_max = (
        torch.min(p_preds[:, :, 0]).item(),
        torch.max(p_preds[:, :, 0]).item(),
    )

    dist_0, dist_1 = abs(p_preds_min - 0), abs(p_preds_max - 1)

    if setting == 1:
        p_bar_values = (
            p_preds[:, 0, 0] - 0.03 if dist_0 > dist_1 else p_preds[:, 1, 0] + 0.03
        )
    elif setting == 2:
        ivl_pbar = (
            [0 + dist_0 / 2, p_preds_min - eps]
            if dist_0 < dist_1
            else [p_preds_max + eps, 1 - dist_1 / 2]
        )
        p_bar_values = gp_sample_prediction(x, kernel, bounds_p=ivl_pbar, **kwargs)
    else:
        ivl_pbar = [0, p_preds_min] if dist_0 > dist_1 else [p_preds_max, 1]
        p_bar_values = gp_sample_prediction(x, kernel, bounds_p=ivl_pbar, **kwargs)

    p_bar_values = torch.clamp(p_bar_values, 0, 1)
    p_bar = torch.column_stack((p_bar_values, 1 - p_bar_values))

    return p_bar


# def sample_pbar_h1(
#     x: np.ndarray,
#     p_preds: np.ndarray,
#     kernel,
#     setting: int = 1,
#     eps: float = 1e-4,
#     **kwargs,
# ):

#     # initialize p_bar as a tensor
#     p_bar = torch.zeros(len(x), p_preds.shape[2])
#     # convert p_preds to tensor if it is not
#     if not isinstance(p_preds, torch.Tensor):
#         p_preds = torch.from_numpy(p_preds).float()
#     # look at max and minimum of p_preds
#     p_preds_max = torch.max(p_preds[:, :, 0]).item()
#     p_preds_min = torch.min(p_preds[:, :, 0]).item()
#     # look which one is closeer to the borders of (0,1)
#     dist_1 = np.abs(p_preds_max - 1)
#     dist_0 = np.abs(p_preds_min - 0)

#     if setting == 1:
#         # add small random noise to upper OR lower bound (based on which is further to
#         # the boundary), set pbar to this values
#         p_bar[:, 0] = (
#             p_preds[:, 0, 0] - np.ones(len(x))*.03
#             if dist_0 > dist_1
#             else p_preds[:, 1, 0] + np.ones(len(x))*.03
#         )
#         # make sure it lies in interval [0,1]
#         p_bar[:, 0] = torch.clamp(p_bar[:, 0], 0, 1)

#     elif setting == 2:
#         # sample pbar from smaller interval with values clost to the boundaries
#         ivl_pbar = (
#             [0 + dist_0 / 2, p_preds_min - eps]
#             if dist_0 < dist_1
#             else [p_preds_max + eps, 1 - dist_1 / 2]
#         )
#         p_bar[:, 0] = gp_sample_prediction(
#             x, kernel=kernel, bounds_p=ivl_pbar, **kwargs
#         )
#     else:
#         # sample pbar from bigger interval which does not contain (p_preds_min, p_preds_max)
#         ivl_pbar = [0, p_preds_min] if dist_0 > dist_1 else [p_preds_max, 1]
#         p_bar[:, 0] = gp_sample_prediction(
#             x, kernel=kernel, bounds_p=ivl_pbar, **kwargs
#         )

#     p_bar[:, 1] = 1 - p_bar[:, 0]

#     return p_bar


def sample_binary_preds_gp(x, kernel, list_bounds_p=[[0, 1], [0, 1]], **kwargs):
    """Generate probabilistic predictions for binary classifiers."""

    n_samples = len(x)
    p_preds = torch.zeros((n_samples, len(list_bounds_p), 2), dtype=torch.float32)

    for i, bounds_p in enumerate(list_bounds_p):
        preds = gp_sample_prediction(x, kernel, bounds_p, tensor=True, **kwargs)
        p_preds[:, i, 0], p_preds[:, i, 1] = preds, 1 - preds

    return p_preds


def gp_sample_prediction(x, kernel, bounds_p=[0, 1], tensor=True, **kwargs):
    """Sample prediction from a Gaussian process."""

    cov = kernel(x.reshape(-1, 1), x.reshape(-1, 1), **kwargs)
    mean = np.zeros_like(x)
    p_pred = np.random.multivariate_normal(mean, cov, 1).flatten()
    p_pred_sc = ab_scale(p_pred, bounds_p[0], bounds_p[1])

    return torch.from_numpy(p_pred_sc).float() if tensor else p_pred_sc


# Function to enforce the range constraints
def enforce_range(samples, ranges):
    constrained_samples = np.copy(samples)
    for i, (min_val, max_val) in enumerate(ranges):
        if samples[i] < min_val:
            constrained_samples[i] = min_val
        elif samples[i] > max_val:
            constrained_samples[i] = max_val
    return constrained_samples
