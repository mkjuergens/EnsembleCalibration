import numpy as np
import torch

from ensemblecalibration.calibration.calibration_estimates.ece_kde import get_ece_kde, get_bandwidth
from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar

def ece_kde_obj(weights_l: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray, params: dict):
    """
    objective function for the bootstrapping test which calcuolates the Lp calibration error
    using kernel density estimation, see also Popordonoska et al. (2022).
    Parameters
    ----------
    weights_l : np.ndarray
        vector of weights of shape (n_samples, n_classes)
    p_probs : np.ndarray
        vector of probabilistic predictions of shape (n_samples, n_ensemble, n_classes)
    y_labels : np.ndarray
        vector of labels of shape (n_samples,) containing labels for each instance
    params : dict
        dictionary of the test parameters
    """
    # calculate convex combination
    p_bar = calculate_pbar(weights_l, p_probs, reshape=True, n_dims=2)
    # lp error: first transform arrays to torch tensors
    p_bar_tensor = torch.from_numpy(p_bar).float()
    y_labels_tensor = torch.from_numpy(y_labels).long()
    # get device
    if "device" in params.keys():
        device = params["device"]
    else:
        device = "cpu"
    # calculate error
    if params["sigma"] is None:
        bw = get_bandwidth(p_bar_tensor, device=device)
    else:
        bw = params["sigma"]
    lp_error = get_ece_kde(p_bar_tensor, y_labels_tensor, bandwidth=bw, p=params["p"],
                           mc_type="canonical", device=device)
    
    # transform to numpy array
    lp_error = lp_error.cpu().detach().numpy()
    return lp_error


def ece_kde_test(p_bar: np.ndarray, y_labels: np.ndarray, params: dict):
    """
    test function for the bootstrapping test which calcuolates the Lp calibration error
    using kernel density estimation, see also Popordonoska et al. (2022).
    Parameters
    ----------
    p_bar : np.ndarray
        vector of convex combinations of shape (n_samples, n_classes)
    y_labels : np.ndarray
        vector of labels of shape (n_samples,) containing labels for each instance
    params : dict
        dictionary of the test parameters
    """
    # lp error: first transform arrays to torch tensors
    p_bar_tensor = torch.from_numpy(p_bar).float()
    y_labels_tensor = torch.from_numpy(y_labels).long()
    # get device
    if "device" in params.keys():
        device = params["device"]
    else:
        device = "cpu"
    # calculate error
    if params["sigma"] is None:
        bw = get_bandwidth(p_bar_tensor, device=device)
    else:
        bw = params["sigma"]
    lp_error = get_ece_kde(p_bar_tensor, y_labels_tensor, bandwidth=bw, p=params["p"],
                           mc_type="canonical", device=device)
    
    # transform to numpy array
    lp_error = lp_error.cpu().detach().numpy()
    return lp_error






    