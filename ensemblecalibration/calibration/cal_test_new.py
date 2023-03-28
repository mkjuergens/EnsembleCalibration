import random
import numpy as np

from scipy.optimize import minimize

from ensemblecalibration.calibration.test_objectives import calculate_pbar
from ensemblecalibration.calibration.helpers import sample_m
from ensemblecalibration.calibration.minimization import (
    solve_cobyla2D,
    solve_neldermead2D,
    solve_cobyla1D,
    solve_neldermead1D,
)
from ensemblecalibration.sampling import multinomial_label_sampling
from ensemblecalibration.nn_training.train import get_optim_lambda_mlp
from ensemblecalibration.nn_training.model import MLPDataset


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
        if params["x_dependency"]:
            l = solve_cobyla2D(P, y, params)
        else:
            l = solve_cobyla1D(P, y, params)

    elif params["optim"] == "neldermead":
        if params["x_dependency"]:
            l = solve_neldermead2D(P, y, params)
        else:
            l = solve_neldermead1D(P, y, params)

    elif params["optim"] == "mlp":
        dataset = MLPDataset(P, y)
        l = get_optim_lambda_mlp(dataset_train=dataset, loss=params["loss"], 
                                   n_epochs=params["n_epochs"], lr=params["lr"])
    else:
        raise NotImplementedError

    minstat = params["obj"](
        l, P, y, params
    )
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

    stats = np.zeros(params["n_resamples"])  # save test statistics here
    for b in range(params["n_resamples"]):
        # bootstrap new matrix of predictions
        P_b = random.sample(P.tolist(), P.shape[0])
        P_b = np.stack(P_b)
        # calculate predicted probabilities of optimal convex combination
        if params["x_dependency"]:
            P_bar_b = calculate_pbar(l, P_b, reshape=True, n_dims=2)
        else:
            P_bar_b = calculate_pbar(l, P_b, n_dims=1)
        assert np.all(
            (P_bar_b >= 0.0) | (P_bar_b <= 1.0)
        ), "all the values of P_bar need to be between 0 and 1"
        # sample the labels from the respective categorical dsitribution
        y_b = np.apply_along_axis(multinomial_label_sampling, 1, P_bar_b)
        # calculate calibration test statistic
        stats[b] = params["test"](P_bar_b, y_b, params)

    # calculate 1 - alpha quantile from the empirical distribution of the test statistic under the null hypothesis
    q_alpha = np.quantile(stats, 1 - params["alpha"])
    # decision: reject test if minstat > q_alpha
    decision = int(np.abs(minstat) > q_alpha)

    return decision, l


def npbe_test_new_alpha(P: np.ndarray, y: np.ndarray, params: dict):
    """perform (updated) calibration test in dependence of the significance level alpha.

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

    stats = np.zeros(params["n_resamples"])  # save test statistics here
    for b in range(params["n_resamples"]):
        # bootstrap new matrix of predictions
        P_b = random.sample(P.tolist(), P.shape[0])
        P_b = np.stack(P_b)
        # calculate predicted probabilities of optimal convex combination
        if params["x_dependency"]:
            P_bar_b = calculate_pbar(l, P_b, reshape=True, n_dims=2)
        else:
            P_bar_b = calculate_pbar(l, P_b, n_dims=1)
        assert np.all(
            (P_bar_b >= 0.0) | (P_bar_b <= 1.0)
        ), "all the values of P_bar need to be between 0 and 1"
        P_bar_b = np.clip(P_bar_b, 0, 1)
        P_bar_b = np.trunc(P_bar_b * 10**3) / (10**3)
        # sample the labels from the respective categorical dsitribution
        y_b = np.apply_along_axis(sample_m, 1, P_bar_b)
        # calculate calibration test statistic
        stats[b] = params["test"](P_bar_b, y_b, params)

    # calculate 1 - alpha quantile from the empirical distribution of the test statistic under the null hypothesis
    q_alpha = np.quantile(stats, 1 - (np.array(params["alpha"])))
    # decision: reject test if minstat > q_alpha
    decision = list(map(int, np.abs(minstat) > q_alpha))

    return decision, l


def _npbe_test_new_alpha(P: np.ndarray, y: np.ndarray, alpha: float, params: dict):
    params["alpha"] = alpha
    dec, l = npbe_test_new_alpha(P, y, params)

    return dec
