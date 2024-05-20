from typing import Union

import random
import numpy as np
import torch

from ensemblecalibration.utils.helpers import multinomial_label_sampling, calculate_pbar
from ensemblecalibration.utils.minimization import calculate_min


def npbe_test_ensemble(
    alpha: list,
    x_inst: np.ndarray,
    p_preds: np.ndarray,
    y_labels: np.ndarray,
    params: dict,
):
   
   
    """new version of the bootstrapping test using uniform sampling of the polytope for testing
    whether there exists a calibrated version in the convex hull

    Parameters
    ----------
    alpha : list
        significance level(s) of the test
    x_inst : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing predictions for each instance and classifier
    p_preds : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing probabilistic predictions for each instance and classifier
    y_labels : np.ndarray of shape (n_samples,)
        array containing labels
    params : dict
        dictionary of test parameters
    Returns
    -------
    decision, (p_vals, stats)
        decision: integer defining whether tso reject (1) or accept (0) the null hypothesis
        ( p_vals: array of p values for each predictor )
        ( stats: array of test statistics for each predictor )

    """

    # calculate optimal weights
    minstat, l_weights = calculate_min(x_inst, p_preds, y_labels, params)
    # calculate p_bar # TODO: calculate pbar ehre or in loop!!!??
    p_bar = calculate_pbar(l_weights, p_preds)
    # run bootstrap test
    decision, p_val, stat = npbe_test_vaicenavicius(alpha, p_bar, y_labels, params)
    print("Decision: ", decision)

    return decision, p_val, stat


def npbe_test_vaicenavicius(
    alpha: Union[list, float], p_probs: np.ndarray, y_labels: np.ndarray, params: dict
):
    """
    Non-parametric bootstrpping etst for a single classifier setting: see also Vaicenavicius et al. (2019).

    Parameters
    ----------
    alpha: list | float
        significance level(s) of the test
    p_probs : np.ndarray
        tensor of probabilistic predictions of shape (n_instances, n_classes)
    y_labels : np.ndarray
        labels
    params : dict
        test parameters
    Returns
    -------
    decision, p_val, stat
        decision of the test, p-value and value of the test statistic for the real data
    """

    # save values of bootstrap statistics here
    stats_h0 = np.zeros(params["n_resamples"])
    # iterate over bootstrap samples
    for b in range(params["n_resamples"]):
        # extract bootstrap sample
        p_probs_b = random.sample(p_probs.tolist(), p_probs.shape[0])
        p_probs_b = np.stack(p_probs_b)
        # sample labels according to categorical distribution
        y_b = np.apply_along_axis(multinomial_label_sampling, 1, p_probs_b)
        # calculate test statistic (miscalibration estimate) under null hypothesis
        stats_h0[b] = params["obj"](p_probs_b, y_b, params)

    # calculate statistic on real data
    stat = params["obj"](p_probs, y_labels, params)
    # to numpy if torch tensor
    if isinstance(stat, torch.Tensor):
        stat = stat.detach().numpy()
    # calculate alpha-quantile of the empirical distribution of the test statistic under the null hypothesis
    q_alpha = np.quantile(stats_h0, 1 - np.array(alpha))
    # decision: reject test if stat > q_alpha
    decision = list(map(int, np.abs(stat) > q_alpha))
    # p-value: fraction of bootstrap samples that are larger than the test statistic on the real data
    p_val = np.sum(stats_h0 > stat) / params["n_resamples"]

    return decision, p_val, stat
