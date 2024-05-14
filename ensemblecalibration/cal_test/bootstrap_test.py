import random
import numpy as np
import torch

from ensemblecalibration.utils.helpers import multinomial_label_sampling

def npbe_test_vaicenavicius(p_probs: np.ndarray, y_labels: np.ndarray,
                            params: dict):
    """
    Non-parametric bootstrpping etst for a single classifier setting: see also Vaicenavicius et al. (2019).

    Parameters
    ----------
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
    # calculate alpha-quantile of the empirical distribution of the test statistic under the null hypothesis
    q_alpha = np.quantile(stats_h0, 1 - np.array(params["alpha"]))
    # decision: reject test if stat > q_alpha
    decision = list(map(int, np.abs(stat) > q_alpha))
    # p-value: fraction of bootstrap samples that are larger than the test statistic on the real data
    p_val = np.sum(stats_h0 > stat) / params["n_resamples"]
    
    return decision, p_val, stat