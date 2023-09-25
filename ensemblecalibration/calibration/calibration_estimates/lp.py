import numpy as np
import torch

from ensemblecalibration.calibration.calibration_estimates.ece_kde import get_ece_kde, get_bandwidth
from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar

def ece_kde_obj(p_bar: np.ndarray, y_labels: np.ndarray, params: dict):
    """
    objective function for the bootstrapping test which calcuolates the Lp calibration error
    using kernel density estimation, see also Popordonoska et al. (2022).
    Parameters
    ----------
    p_bar : np.ndarray
        vector of probabilistic predictions of shape (n_samples, n_classes)
    y_labels : np.ndarray
        vector of labels of shape (n_samples,) containing labels for each instance
    params : dict
        dictionary of the test parameters
    """
    
    # lp error: first transform arrays to torch tensors
    p_bar_tensor = torch.from_numpy(p_bar)
    y_labels_tensor = torch.from_numpy(y_labels).int()
    # get device
    if "device" in params.keys():
        device = params["device"]
    else:
        device = "cpu"
    # calculate error
    if params["sigma"] is None:
        bw = get_bandwidth(p_bar_tensor, device=device)
    lp_error = get_ece_kde(p_bar_tensor, y_labels_tensor, bandwidth=bw, p=params["p"],
                           mc_type="canonical", device=device)
    
    # transform to numpy array
    lp_error = lp_error.cpu().detach().numpy()
    return lp_error






    