from .brier import brier_obj
from .kde_ece import ece_kde_obj
from .mmd_kce import mmd_kce_obj
from .skce import skce_obj
from .kde_kl import kl_kde_obj

from src.utils.helpers import multinomial_label_sampling
import numpy as np
import torch


def miscalibration_estimate_sim(
    p_probs, p_true=None, n_samples=2000, cal_estimate="L2"
):
    """
    Compute a miscalibration estimate for a given probability vector p_probs
    vs. some 'true' distribution p_true.

    Parameters
    ----------
    p_probs : np.ndarray
        A probability vector of shape (K,) or (K) distribution over classes.
    p_true : np.ndarray
        'True' distribution, shape (K,). Default None => can't do anything
    n_samples : int
        Number of random samples used for the estimate.
    cal_estimate : str
        One of {"L2", "SKCE", "Brier", "MMD", "KL"}.

    Returns
    -------
    float
        The estimated miscalibration value.
    """
    if p_true is None:
        raise ValueError("p_true must be provided as the ground-truth distribution.")

    if cal_estimate == "L2":
        params = {"p": 2, "bw": 0.001}
        miscal_obj = ece_kde_obj
    elif cal_estimate == "SKCE":
        params = {"bw": 0.001}
        miscal_obj = skce_obj
    elif cal_estimate == "Brier":
        params = {}
        miscal_obj = brier_obj
    elif cal_estimate == "MMD":
        params = {"bw": 0.01}
        miscal_obj = mmd_kce_obj
    elif cal_estimate == "KL":
        params = {"bw": 0.001}
        miscal_obj = kl_kde_obj
    else:
        raise ValueError(f"Invalid calibration estimate: {cal_estimate}")

    # Repeat the single distribution p_true for n_samples
    p_true_full = np.tile(p_true, (n_samples, 1))
    # Sample from p_true to get 'labels'
    y_labels = multinomial_label_sampling(p_true_full)

    # Also tile the predicted p_probs
    p_probs_full = np.tile(p_probs, (n_samples, 1))

    # Compute the miscalibration measure
    miscalibration = miscal_obj(p_probs_full, y_labels, params).detach()
    return miscalibration


def create_heatmap_data(
    scale, n_samples, p_true, measures=("Brier", "L2", "SKCE", "MMD", "KL")
):
    """
    Generate miscalibration heatmap data for given measures
    across the 2-simplex with specified resolution 'scale'.

    Returns a dict { measure_name : dict[(i,j) -> float] }

    scale=40 => (i+j+k=40) grid.
    p_true is the ground truth distribution, e.g. [0.1,0.1,0.8].
    """
    # We'll store a separate dictionary for each measure
    data_dicts = {m: {} for m in measures}

    for i in range(scale + 1):
        for j in range(scale + 1 - i):
            k = scale - i - j

            # Probability vector
            p = np.array([i, j, k], dtype=float) / scale

            # Evaluate each measure
            for measure in measures:
                val = miscalibration_estimate_sim(
                    p_probs=p, p_true=p_true, n_samples=n_samples, cal_estimate=measure
                )
                data_dicts[measure][(i, j)] = val

            # (Optional) debug print
            # print(f"Computed {i},{j},{k} -> p={p} done.")

    return data_dicts
