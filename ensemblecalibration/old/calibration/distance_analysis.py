import numpy as np
from typing import Optional

from tqdm import tqdm
from scipy.spatial.distance import jensenshannon, euclidean
from scipy.special import rel_entr

from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar
from ensemblecalibration.sampling import multinomial_label_sampling

def analyse_stats_distances(l_weights, y_labels, p_probs: np.ndarray, params: dict,
                             distance=euclidean, n_iter: int = 1000, 
                             l_prior: Optional[np.ndarray] = None):
    """function for analysing the corerlation between distances of weight vectors and the 
    respective values of the miscalibration estimates.

    Parameters
    ----------
    l_weights : np.ndarray
        array of shape (n_ensembles,) containing weight coefficients
    p_probs : np.ndarray
        array of shape (n_instances, n_ensembles, n_classes) containing probabilistic predictions
    y_labels : np.ndarray
        array of shape (n_instances,) containing labels
    params : dict
        dictionary containing test parameters
    distance : function, optional
        measure of distance used, by default euclidean
    n_iter : int, optional
        number of iterations, by default 1000
    l_prior : Optional[np.ndarray], optional
        parameter of the dirichlet distribution, by default None, in which case it is set to
          [1]*n_ensembles

    Returns
    -------
    dists, stats, l_all
        list of distances and list of respective values of the miscalibration estimates,
        as well as a matrix of all newly sampled weight vectors
    """

    n_instances, n_ensembles, _ = p_probs.shape 
    assert n_instances == y_labels.shape[0], "p_probs and y_labels must have the same number of instances"

    dists = []
    stats = []
    l_all = np.zeros((n_iter, n_ensembles))
    # sample new weight vector
    if l_prior is None:
        l_prior = np.ones(n_ensembles)
    for i in tqdm(range(n_iter)):
        # sample new weights
        l_new = np.random.dirichlet(l_prior)
        l_all[i] = l_new
        # calculate new predictions
        p_bar_new = calculate_pbar(l_new, p_probs, reshape=False, n_dims=1)
        # calculate distance between lambda vectors
        dist = distance(l_weights, l_new)
        stat = params["obj"](p_bar_new, y_labels, params)
        dists.append(dist)
        stats.append(stat)

    return dists, stats, l_all


def run_distance_analysis(params: dict , y_labels: np.ndarray, p_probs: np.ndarray, l_weight: np.ndarray,
                           n_iter: int = 1000, distance=euclidean, l_prior: Optional[np.ndarray] = None):
    """function for running the distance analysis for a given set of parameters"""
    
    res = {}

    for test in params:
        print(f"Running analysis for test {test}")
        dists, stats, l_all = analyse_stats_distances(l_weight, y_labels, p_probs, params[test]["params"],
                                                         distance=distance, n_iter=n_iter,
                                                         l_prior=l_prior)
        res[test] = {"dists": dists, "stats": stats, "l_all": l_all}

    return res

def run_heatmap_analysis(params: dict, l_weight: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray,
                         n_iter: int = 1000):
    
    res = {}

    for i, test in tqdm(enumerate(params)):
        print(f"Running analysis for test {test}")
        dists, stats, l_all = analyse_stats_distances(l_weight, y_labels, p_probs, params[test]["params"],
                                                         distance=euclidean, n_iter=n_iter)
        res[test] = {"dists": dists, "stats": stats, "l_all": l_all}




    