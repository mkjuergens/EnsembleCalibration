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


def npbe_test_v3_alpha(p_probs: np.ndarray, y_labels: np.ndarray, params: dict):
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

    Returns
    -------
    int, in {0, 1}
    test result
    """

    # array for saving decisions for each iteration
    decisions = np.zeros((params["n_predictors"], len(params["alpha"])))
    for n in range(params["n_predictors"]):
        # sample predictor within convex hull
        p_bar = sample_p_bar(p_probs=p_probs, params=params)
        # cut down values outside [0, 1] for numerical stability
        p_bar = np.trunc(p_bar*10**3)/(10**3)
        p_bar = np.clip(p_bar, 0, 1)
        # save test statistics for bootstrap iterations
        stats = np.zeros(params["n_resamples"])
        # bootstrap iterations
        for b in range(params["n_resamples"]):
            # randomly sample from p_bar
            p_bar_b = np.stack(random.sample(p_bar.tolist(), p_bar.shape[0]))
            # sample labels uniformly from the induced caftegorical distribution
            y_b = np.apply_along_axis(multinomial_label_sampling, 1, p_bar_b)
            stats[b] = params["test"](p_bar_b, y_b, params)
        # calculate quantile of empirical distribution
        q_alpha = np.quantile(stats, 1 - np.array(params["alpha"]))
        # calculate value of test statistic for original labels and predictor
        minstat = params["obj"](p_bar, y_labels, params)
        decision = list(map(int, np.abs(minstat) > q_alpha)) # zero if false, 1 (reject) if true 
        decisions[n, :] = decision

    # check if any test accepts, if yes, accept, else, no
    final_r = np.min(decisions, axis=0)
    return final_r


def _npbe_test_v3_alpha(p_probs: np.ndarray, y_labels: np.ndarray, alpha, params: dict):
    """
    version of the test where alpha is given in as a function parameter
    """
    params["alpha"] = alpha
    result = npbe_test_v3_alpha(p_probs=p_probs, y_labels=y_labels, params=params)
    return result



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



if __name__ == "__main__":
    p_bar = np.random.dirichlet([1] * 10, size=100)
    p_bar_b = random.sample(p_bar.tolist(), p_bar.shape[0])
    p_bar_b = np.stack(p_bar_b)
    print(p_bar_b.sum(1))
    y_b = np.apply_along_axis(multinomial_label_sampling, 1, p_bar_b)
    print(y_b)
