import random
import numpy as np

from scipy.optimize import minimize

from ensemblecalibration.calibration.test_objectives import skce_ul_obj_new, skce_uq_obj_new, classece_obj_new, confece_obj_new, hl_obj_new, calculate_pbar
from ensemblecalibration.calibration.helpers import constraint1_new, constraint2_new
from ensemblecalibration.calibration.minimization import solve_cobyla
from ensemblecalibration.sampling import multinomial_label_sampling


def calculate_min_new(P: np.ndarray, y: np.ndarray, params: dict):
    """Calculation of the convex combination resulting in a minimal calibration value
     using a predefined method

    Parameters
    ----------
    P : np.ndarray
        matrix of shape (N, M, K) containing predictions of each predictor for each instance
    y : np.ndarray
        array of shape (N,) containing labels
    params : dict
        dictionary containing test parameters

    Returns
    -------
    minstat, l
        minimum value of the test statistic and the respective matrix of weight vectors for each instance
    """

    # calculate minimum of objective
    if params["optim"] == "cobyla":

        l = solve_cobyla(P, y, params)
    else:
        raise NotImplementedError

   # l = l.reshape(P.shape[0], P.shape[1]) # reshape to get a matrix of weight vectors

    minstat = params["obj"](l, P, y, params) # minimum value of calibration objective/measure
    return minstat, l

def npbe_test_new(P: np.ndarray, y: np.ndarray, params: dict):
    """perform (updated) calibration test: It consists of first calculing the optimal convex combination
    of predictors, and then using the weight matrix for the bootstrapping to calculate the empirical distribution
    of the calibration measure under the null hypothesis

    Parameters
    ----------
    P : np.ndarray
        tensor containing probabilitic predictions for each instance for each predictor
    y : np.ndarray
        (N,) shaped array containing labels for each instance
    params : dict
        dictionary containing test parameters

    Returns
    -------
    decision: integer defining whether to reject (1) or accept (0) the null hypothesis
    l: matrix of weight vectors for each instance
    """

    # calculate optimal convex combination of predictions
    minstat, l = calculate_min_new(P, y, params) 

    stats = np.zeros(params["n_resamples"]) # save test statistics here
    for b in range(params["n_resamples"]):
        # bootstrap new matrix of predictions
        P_b = random.sample(P.tolist(), P.shape[0])
        P_b = np.stack(P_b)
        # calculate predicted probabilities of optimal convex combination
        P_bar_b = calculate_pbar(l, P, reshape=True)
        # sample the labels from the respective categorical dsitribution
        y_b = np.apply_along_axis(multinomial_label_sampling, 1, P_bar_b)
        # calculate calibration test statistic
        stats[b] = params["test"](P_bar_b, y_b, params)

    # calculate 1 - alpha quantile from the empirical distribution of the test statistic under the null ypothesis
    q_alpha = np.quantile(stats, 1 - params["alpha"])
    # decision: reject test if minstat > q_alpha
    decision = int(np.abs(minstat) > q_alpha)

    return decision, l





