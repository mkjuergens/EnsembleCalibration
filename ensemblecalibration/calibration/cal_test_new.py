import random
import numpy as np

from scipy.optimize import minimize

from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar
from ensemblecalibration.calibration.helpers import sample_m
from ensemblecalibration.calibration.minimization import (
    solve_cobyla2D,
    solve_neldermead2D,
    solve_cobyla1D,
    solve_neldermead1D,
)
from ensemblecalibration.sampling import multinomial_label_sampling, sample_p_bar
from ensemblecalibration.nn_training.train import get_optim_lambda_mlp
from ensemblecalibration.nn_training.dataset import MLPDataset
from ensemblecalibration.calibration.cal_test_vaicenavicius import npbe_test_vaicenavicius
from ensemblecalibration.calibration.helpers import sort_and_reject_alpha


def calculate_min_inst_dependent(x_inst: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray,
                                 params: dict):
    if params["optim"] == "perceptron":
        dataset = MLPDataset(x_train=x_inst, P=p_probs, y=y_labels)
        l_weights = get_optim_lambda_mlp(dataset, loss=params["loss"], n_epochs=params["n_epochs"],
                                         lr=params["lr"], batch_size = len(x_inst))
    else:
        raise NotImplementedError
    # calculate p_bar
    p_bar = calculate_pbar(l_weights, p_probs, reshape=True, n_dims=2)
    # calculate test statistic
    minstat = params["obj"](p_bar, y_labels, params)
    
    return minstat, l_weights


def npbe_test_mlp_new(x_inst: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray, params: dict):
    """new version of the bootstrapping test using uniform sampling of the polytope for testing
    whether there exists a calibrated version in the convex hull

    Parameters
    ----------
    x_inst : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing predictions for each instance and classifier
    p_probs : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing probabilistic predictions for each instance and classifier
    y_labels : np.ndarray of shape (n_samples,)
        array containing labels
    params : dict
        dictionary of test parameters
    correction : bool, optional
        whether to use a correction for the test statistic, by default False
    Returns
    -------
    decision, (p_vals, stats)
        decision: integer defining whether tso reject (1) or accept (0) the null hypothesis
       ( p_vals: array of p values for each predictor )
       ( stats: array of test statistics for each predictor )

    """

    # calculate optimal convex combination of predictions
    minstat, l = calculate_min_inst_dependent(x_inst, p_probs=p_probs, y_labels=y_labels,
                                               params=params)
    print(f'Minstat at : {minstat}')

    stats = np.zeros(params["n_resamples"])  # save test statistics here
    for b in range(params["n_resamples"]):
        # bootstrap new matrix of predictions
        P_b = random.sample(p_probs.tolist(), p_probs.shape[0])
        P_b = np.stack(P_b)
        # calculate predicted probabilities of optimal convex combination
        P_bar_b = calculate_pbar(l, P_b, reshape=True, n_dims=2)
        assert np.all(
            (P_bar_b >= 0.0) | (P_bar_b <= 1.0)
        ), "all the values of P_bar need to be between 0 and 1"
        # sample the labels from the respective categorical dsitribution
        y_b = np.apply_along_axis(multinomial_label_sampling, 1, P_bar_b)
        # calculate calibration test statistic
        stats[b] = params["test"](P_bar_b, y_b, params)

    # calculate 1 - alpha quantile from the empirical distribution of the test statistic under the null hypothesis
    q_alpha = np.quantile(stats, 1 - (np.array(params["alpha"])))
    # decision: reject test if minstat > q_alpha
    decision = list(map(int, np.abs(minstat) > q_alpha))

    return decision, l

def _npbe_test_mlp_new_alpha(x_inst: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray, alpha,
                              params: dict):

    params["alpha"] = alpha
    decision, l = npbe_test_mlp_new(x_inst, p_probs, y_labels, params)

    return decision, l


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
        l = get_optim_lambda_mlp(
            dataset_train=dataset,
            loss=params["loss"],
            n_epochs=params["n_epochs"],
            lr=params["lr"],
        )
    else:
        raise NotImplementedError
    
    minstat = params["obj"](P, y, params)


    return minstat, l


def npbe_test_v3_alpha(p_probs: np.ndarray, y_labels: np.ndarray, params: dict,
                       correction: bool = False):
    """new version of the bootstrapping test using uniform sampling of the polytope for testing
    whether there exists a calibrated version in the convex hull

    Parameters
    ----------
    p_probs : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing probabilistic predictions for each instance and classifier
    y_labels : np.ndarray of shape (n_samples,)
        array containing labels
    params : dict
        dictionary of test parameters
    correction : bool, optional
        whether to use a correction for the test statistic, by default False

    Returns
    -------
    decision, (p_vals, stats)
        decision: integer defining whether to reject (1) or accept (0) the null hypothesis
       ( p_vals: array of p values for each predictor )
       ( stats: array of test statistics for each predictor )

    """

    # array for saving decisions for each iteration
    decisions = np.zeros((params["n_predictors"], len(params["alpha"])))
    # save p values for each predictor here
    p_vals = np.zeros(params["n_predictors"])
    stats = np.zeros(params["n_predictors"])

    for n in range(params["n_predictors"]):
        # sample predictor within convex hull
        p_bar = sample_p_bar(p_probs=p_probs, params=params)
        # cut down values outside [0, 1] for numerical stability
        p_bar = np.trunc(p_bar*10**3)/(10**3)
        p_bar = np.clip(p_bar, 0, 1)
        # save test statistics for bootstrap iterations
        _, p_val, stat = npbe_test_vaicenavicius(p_bar, y_labels, params)
        p_vals[n] = p_val
        stats[n] = stat
    if correction:
        decs = sort_and_reject_alpha(p_vals, params["alpha"], method="hochberg")
    else:
        decs = sort_and_reject_alpha(p_vals, params["alpha"], method=None)
    # global hypothesis: accept if any test accepts
    # take minimum over second axis, for every level of alpha
    decision = np.min(decs, axis=1)
    return decision

def _npbe_test_v3_alpha(p_probs: np.ndarray, y_labels: np.ndarray, alpha, params: dict,
                        make_cor: bool = False):
    """
    version of the test where alpha is given in as a function parameter

    Parameters
    ----------
    p_probs : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing probabilistic predictions for each instance and classifier
    y_labels : np.ndarray of shape (n_samples,)
        array containing labels
    alpha : float
        significance level
    params : dict
        dictionary of test parameters
    make_cor : bool, optional
        whether to use a correction for the test statistic, by default False
    """
    params["alpha"] = alpha

    result = npbe_test_v3_alpha(p_probs=p_probs, y_labels=y_labels, params=params, correction=True)
    return result



def npbe_test_new_alpha(P: np.ndarray, y: np.ndarray, params: dict):
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
    q_alpha = np.quantile(stats, 1 - (np.array(params["alpha"])))
    # decision: reject test if minstat > q_alpha
    decision = list(map(int, np.abs(minstat) > q_alpha))
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



if __name__ == "__main__":
    p_bar = np.random.dirichlet([1] * 10, size=100)
    p_bar_b = random.sample(p_bar.tolist(), p_bar.shape[0])
    p_bar_b = np.stack(p_bar_b)
    print(p_bar_b.sum(1))
    y_b = np.apply_along_axis(multinomial_label_sampling, 1, p_bar_b)
    print(y_b)
