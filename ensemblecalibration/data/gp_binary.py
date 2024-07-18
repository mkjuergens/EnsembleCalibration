import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel
from numpy import random
import torch

from ensemblecalibration.utils.helpers import (
    calculate_pbar,
    multinomial_label_sampling,
    sample_function,
    ab_scale
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


def sample_pbar_h0(
    x: np.ndarray, p_preds: torch.Tensor, x_dep: bool = False, deg: int = 2
):

    weights_l = np.zeros((len(x), p_preds.shape[1]))
    # convert p_preds to tensor if it is not
    # sample convex combination within convex hull
    if x_dep:
        # sample lambda as a polynomial function of x
        l_weights = sample_function(x, deg=deg)
        weights_l[:, 0] = l_weights
        weights_l[:, 1] = 1 - l_weights
    else:
        # sample a single lambda from the unit interval
        l_weights = np.random.rand(1)
        # set all entries to the same value
        weights_l[:, 0] = l_weights
        weights_l[:, 1] = 1 - l_weights

    weights_l = torch.from_numpy(weights_l).float()
    p_bar = calculate_pbar(weights_l, p_preds, reshape=False)
    return p_bar, weights_l


def sample_pbar_h1(
    x: np.ndarray,
    p_preds: np.ndarray,
    kernel,
    setting: int = 1,
    eps: float = 1e-4,
    **kwargs,
):

    # initialize p_bar as a tensor
    p_bar = torch.zeros(len(x), p_preds.shape[2])
    # convert p_preds to tensor if it is not
    if not isinstance(p_preds, torch.Tensor):
        p_preds = torch.from_numpy(p_preds).float()
    # look at max and minimum of p_preds
    p_preds_max = torch.max(p_preds[:, :, 0]).item()
    p_preds_min = torch.min(p_preds[:, :, 0]).item()
    # look which one is closeer to the borders of (0,1)
    dist_1 = np.abs(p_preds_max - 1)
    dist_0 = np.abs(p_preds_min - 0)

    if setting == 1:
        # add small random noise to upper OR lower bound (based on which is further to
        # the boundary), set pbar to this values
        p_bar[:, 0] = (
            p_preds[:, 0, 0] - np.ones(len(x))*.03
            if dist_0 > dist_1
            else p_preds[:, 1, 0] + np.ones(len(x))*.03
        )
        # make sure it lies in interval [0,1]
        p_bar[:, 0] = torch.clamp(p_bar[:, 0], 0, 1)

    elif setting == 2:
        # sample pbar from smaller interval with values clost to the boundaries
        ivl_pbar = (
            [0 + dist_0 / 2, p_preds_min - eps]
            if dist_0 < dist_1
            else [p_preds_max + eps, 1 - dist_1 / 2]
        )
        p_bar[:, 0] = gp_sample_prediction(
            x, kernel=kernel, bounds_p=ivl_pbar, **kwargs
        )
    else:
        # sample pbar from bigger interval which does not contain (p_preds_min, p_preds_max)
        ivl_pbar = [0, p_preds_min] if dist_0 > dist_1 else [p_preds_max, 1]
        p_bar[:, 0] = gp_sample_prediction(
            x, kernel=kernel, bounds_p=ivl_pbar, **kwargs
        )

    p_bar[:, 1] = 1 - p_bar[:, 0]

    return p_bar


def sample_binary_preds_gp(
    x: np.ndarray, kernel, list_bounds_p: list = [[0, 1], [0, 1]], **kwargs
):
    """returns a matrix of shape (n_samples, n_preds, 2) containing probabilistic predictions
    of a number of binary classifiers. The probabilty for the first class is sampled from a
    Gaussian process with kernel specified by the user. The second class probability is
    1 - p1.

    Parameters
    ----------
    x : np.ndarray
        array of shape (n_samples,) containing the inputs.
    kernel : sklearn.metrics.pairwise
        Kernel function which is used to sample from the GP.
    list_bounds_p : list, optional
        upper and lower bound for heprobability of the first class per predictor
        , by default [[0, 1], [0,1]]

    Returns
    -------
    np.ndarray
        matrix of shape (n_samples, n_preds, 2) containing probabilistic predictions
    """

    # intialize matrix of shape (n_samples, n_preds, 2)
    p_preds = torch.zeros((len(x), len(list_bounds_p), 2), dtype=torch.float32)
    for i, bounds_p in enumerate(list_bounds_p):
        preds = gp_sample_prediction(x, kernel, bounds_p, tensor=True, **kwargs)
        p_preds[:, i, 0] = preds
        p_preds[:, i, 1] = 1 - preds

    return p_preds


def gp_sample_prediction(
    x: np.ndarray, kernel, bounds_p: list = [0, 1], tensor: bool = True, **kwargs
):
    """

    Arguments:
      x : ndarray (n_samples,)
        Inputs.
      kernel : sklearn.metrics.pairwise
        Kernel function which defines the hypothesis class of ensemble members
      bounds_p : list (default=[0,1])
        Lower and upper bound for probabilities of predictor.
      x_dependent : boolean (default=False)
        Whether convex combination depends on inputs or not.
      **kwargs : dict
        Additional arguments for kernel.

    Output:
      p1 : ndarray (n_samples,)
        Probs first ensemble member.
      p2 : ndarray (n_samples,)
        Probs second ensemble member.
      l : ndarray (n_samples,)
        Weight for convex combination.
    """
    # calculate sample covariance matrix given kernel
    cov = kernel(x.reshape(-1, 1), x.reshape(-1, 1), **kwargs)
    # specify mean function
    mean = np.zeros_like(x)
    # sample from GP(mean,cov)
    p_pred = random.multivariate_normal(mean, cov, 1).flatten()
    # removed sampling of weights for now
    #  if x_dependent:
    #     l = random.multivariate_normal(mean, cov, 1).flatten()
    # else:
    #    l = np.repeat(np.random.rand(1), len(x))
    # min-max scale
    p_pred_sc = ab_scale(p_pred, bounds_p[0], bounds_p[1])

    if tensor:
        return torch.from_numpy(p_pred_sc).float()
    else:
        return p_pred_sc


# Function to enforce the range constraints
def enforce_range(samples, ranges):
    constrained_samples = np.copy(samples)
    for i, (min_val, max_val) in enumerate(ranges):
        if samples[i] < min_val:
            constrained_samples[i] = min_val
        elif samples[i] > max_val:
            constrained_samples[i] = max_val
    return constrained_samples
