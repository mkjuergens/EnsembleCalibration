from typing import Optional
import random
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm

from ensemblecalibration.calibration.experiments import experiment_h0
from ensemblecalibration.calibration.cal_test_new import calculate_min_new
from ensemblecalibration.calibration.calibration_estimates.distances import w1_distance
from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar
from ensemblecalibration.sampling import multinomial_label_sampling, sample_p_bar


def npbe_test_distances_two_lambdas(
    p_probs: np.ndarray,
    dist_fct,
    weights_1: np.ndarray,
    weights_2: np.ndarray,
    params: dict,
):
    """function which, given a "true a set of probabilistic predictors, sets for a given weight
    vector the resulting convex combination as the truly calibraetd one, and computes the p value
    for another convex combination. I treturns the distance between the two convex combinations of
    probabilistic predicitons and the p value for the calibration test of the
    second convex combination.

    Parameters
    ----------
    p_probs : np.ndarray
        tensor of shape (n_instances, n_classes, n_predictors) containing the probabilistic predictions
    dist_fct : _type_
        distance measure between two convex combinations of probabilistic predictions
    weights_1 : np.ndarray
        vector of weights for the first convex combination of probabilistic predictions by which
        the categorical distribution from which the labels are then sampled is induced
    weights_2 : np.ndarray
        weigfht vector of second convex combination of probabilistic predictions
        which is then used for the calibration test
    params : dict
        dictionary containing parameters for the calibration test

    Returns
    -------
    p_val : float
    dist: float
        p value of the test and distance between the two probability distributions
    """

    # calculate pbar for first lambda
    if params["x_dependency"]:
        p_bar = calculate_pbar(weights_1, p_probs, reshape=True, n_dims=2)
        p_bar_2 = calculate_pbar(weights_2, p_probs, reshape=True, n_dims=2)
    else:
        p_bar = calculate_pbar(weights_1, p_probs, reshape=False, n_dims=1)
        p_bar_2 = calculate_pbar(weights_2, p_probs, reshape=False, n_dims=1)
    # sample labels from pbar_1
    y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_bar)
    # calculate pbar for second lambda
    # calculate distance between pbar and pbar_2
    dist = dist_fct(p_bar, p_bar_2)
    # calculate p value
    _, p_val, stat = npbe_test_vaicenavicius(p_probs=p_bar_2, y_labels=y_labels, params=params)

    return p_val, dist

def distance_analysis_npbe(p_probs: np.ndarray, params: dict, dist_fct=w1_distance,
                            n_iters: int = 1000):
    """
    function which analysis the p values of the NPBE test of Vaicenavicius et al with regard 
    to the distance between the probabilistic predictor of the underlying calibrated model and another 
    probabilistic predictor.

    Parameters
    ----------
    p_probs : np.ndarray
        tensor of shape (n_instances, n_classes, n_predictors) containing the probabilistic predictions
    params : dict
        dictionary containing parameters for the calibration test
    dist_fct : _type_
        distance measure between two convex combinations of probabilistic predictions
    n_iters : int
        number of iterations (default: 1000)

    Returns
    -------
    p_vals, distances
        p values and distances between the two convex combinations of probabilistic predictions
    """

    # sample first "true" weight vector
    weights_1 = np.random.dirichlet([1] * p_probs.shape[1], size=1)[0, :]
    # save p_values and distances
    p_vals = np.zeros(n_iters)
    distances = np.zeros(n_iters)
    for n in tqdm(range(n_iters)):
        # sample second weight vector
        weights_2 = np.random.dirichlet([1] * p_probs.shape[1], size=1)[0, :]
        # calculate p value and distance between the two convex combinations
        p_val, dist = npbe_test_distances_two_lambdas(p_probs=p_probs, dist_fct=dist_fct,
                                                        weights_1=weights_1, weights_2=weights_2,
                                                        params=params)
        p_vals[n] = p_val
        distances[n] = dist
    
    return p_vals, distances
                                                      


def npbe_test_null_hypothesis(
    params: dict,
    alpha: float = 0.05,
    n_iters: int = 1000,
    n_classes: int = 2,
    n_instances: int = 1000,
):
    """
    Function for testing the null hypothesis of the NPBE test for a single classifier setting.
    Parameters
    ----------
    params : dict
        test parameters
    n_iters : int
        number of iterations (default: 1000)
    n_classes : int
        number of classes (default: 2)
    n_instances : int
        number of instances (default: 1000)

    Returns
    -------
    p_vals, stats, decisions
        p-values, test statistics and decisions for the null hypothesis
    """
    # set alpha value
    params["alpha"] = alpha
    p_vals = np.zeros(n_iters)
    stats = np.zeros(n_iters)

    decisions = np.zeros(n_iters)
    for n in tqdm(range(n_iters)):
        p_probs = np.random.dirichlet([1] * n_classes, size=n_instances)
        y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_probs)
        decision, p_val, stat = npbe_test_vaicenavicius(p_probs, y_labels, params)
        p_vals[n] = p_val
        stats[n] = stat
        decisions[n] = decision

    return p_vals, stats, decisions


def npbe_test_vaicenavicius(p_probs: np.ndarray, y_labels: np.ndarray, params: dict):
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
        # calculate test statistic under null hypothesis
        stats_h0[b] = params["obj"](p_probs_b, y_b, params)
    # calculate statistic on real data
    stat = params["obj"](p_probs, y_labels, params)
    # calculate alpha-quantile of the empirical distribution of the test statistic under the null hypothesis
    q_alpha = np.quantile(stats_h0, 1 - params["alpha"])
    # decision: reject test if stat > q_alpha
    decision = int(np.abs(stat) > q_alpha)
    # p-value: fraction of bootstrap samples that are larger than the test statistic on the real data
    p_val = np.sum(stats_h0 > stat) / params["n_resamples"]
    return decision, p_val, stat


def npbe_test_v3_p_values(
    p_probs: np.ndarray,
    y_labels: np.ndarray,
    params: dict,
    weights_l: Optional[np.ndarray] = None,
):
    """version of the non-parametric bootstrappping test for analysing the p-values in relation
    to the empirical distribution under the null hypothesis. Given as input the probabilistic predictions
    of the ensemble members, the labels, and the test parameters, it returns the value of the test statistic


    Parameters
    ----------
    p_probs : np.ndarray
        tensor of probabilistic predictions of shape (n_instances, n_predictors, n_classes)
    y_labels : np.ndarray
        labels
    params : dict
        test parameters
    weights_l : Optional[np.ndarray],
        weight matrix for the convex combination of predictors from which the labels are (initially)
        sampled.

    Returns
    -------
    minstat, stats, p_val
        value of the statistic on the real data for the randomly sampled convex combination (c.c.),
        values of the bootstrapping samples (of the c.c.) under the null hypothesis, p-value
    """
    # save p-values here
    p_vals = np.zeros(params["n_predictors"])
    # save test statistics (evaluated for the bootstrap under the null hypothesis) here
    stats_h0 = np.zeros((params["n_predictors"], params["n_resamples"]))
    # save value of statistic on real data here
    stats = np.zeros(params["n_predictors"])
    if weights_l is not None:
        # calulate true value of the miscalibration measure for the "truely calibrated" convex combination
        true_stat = params["obj_lambda"](
            weights_l=weights_l, p_probs=p_probs, y_labels=y_labels, params=params
        )
    # predictor iterations (for each predictor)
    for n in tqdm(range(params["n_predictors"])):
        # sample a new p_bar using the p_probs
        p_bar = sample_p_bar(p_probs=p_probs, params=params)
        # truncate p_bar to avoid numerical issues
        p_bar = np.trunc(p_bar * 10**3) / (10**3)
        # clip p_bar to [0, 1]
        p_bar = np.clip(p_bar, 0, 1)
        # bootstrap iterations
        for b in range(params["n_resamples"]):
            # randomly sample from p_bar
            p_bar_b = np.stack(random.sample(p_bar.tolist(), p_bar.shape[0]))
            # sample labels uniformly from the induced caftegorical distribution
            y_b = np.apply_along_axis(multinomial_label_sampling, 1, p_bar_b)
            stats_h0[n, b] = params["test"](p_bar_b, y_b, params)
        # value of statistic on real data
        stat = params["obj"](p_bar, y_labels, params)
        stats[n] = stat
        # calculate empirical distribution
        ecdf = ECDF(stats_h0[n, :])
        # p value: 1 - F(minstat)
        p_val = 1 - ecdf(stat)
        p_vals[n] = p_val

    if weights_l is None:
        return stats, stats_h0, p_vals
    else:
        return stats, stats_h0, p_vals, true_stat


def npbe_test_p_values(p_probs: np.ndarray, y_labels: np.ndarray, params: dict):
    """test for comparing p values and values of statistics for the non-parametric bootstrapping
    test

    Parameters
    ----------
    p_probs : np.ndarray
        tensor of shape (n_samples, n_predictors, n_classes)
    y_labels : np.ndarray
        array of shape (n_samples,)
    params : dict
        dictionary of test parameters

    Returns
    -------
    minstat, p_val, stats
        value of the statistic on the real data,
        p-value,
        values of the statistic for the bootstrapped samples
        under the null hypothesis
    """
    # array fodr saving statistics in the bootstrapping iterations
    stats = np.zeros(params["n_resamples"])
    # calculate minimum value of evaluations of the statistics
    # TODO: adjust objectives here
    minstat, weights_l = calculate_min_new(p_probs, y_labels, params=params)

    # do boootstrapping
    for b in range(params["n_resamples"]):
        p_b = random.sample(p_probs.tolist(), p_probs.shape[0])
        p_b = np.stack(p_b)
        # calculate convex combination of predictions
        if params["x_dependency"]:
            p_bar_b = calculate_pbar(weights_l=weights_l, P=p_b, reshape=True, n_dims=2)
        else:
            p_bar_b = calculate_pbar(
                weights_l=weights_l, P=p_b, reshape=False, n_dims=1
            )

        assert not np.isnan(
            p_bar_b
        ).any(), "matrix with prob predictions contains nan values"
        p_bar_b = np.trunc(p_bar_b * 10**3) / (10**3)
        p_bar_b = np.clip(p_bar_b, 0, 1)
        # sample labels from categorical distributiuon induced by p_bar_b
        y_b = np.apply_along_axis(multinomial_label_sampling, 1, p_bar_b)
        stats[b] = params["test"](p_bar_b, y_b, params)

    # calculate empirical distribution
    ecdf = ECDF(stats)
    # p value: 1 - F(minstat)
    p_val = 1 - ecdf(minstat)

    return minstat, p_val, stats


def _simulation_pvals(
    tests, N: int, M: int, K: int, R: int, u: float, experiment=experiment_h0
):
    """simulation for testing the correrlation between p-values and minimum values of test statistics

    Parameters
    ----------
    tests : dictionary of test parameters
        _description_
    N : int
        number of instances per dataset
    M : int
        number of ensemble members
    K : int
        number of classes
    R : int
        number of dataset/resampling iterations
    u : float
        parameter describing the "uncertainty"/spread for the generated datasets
    experiment : _type_, optional
        _description_, by default experiment_h0

    Returns
    -------
    results
        dictionary containing matrices of p-values and values of statistics for each test
    """
    results = {}
    for test in tests:
        # save p_vals and statistics in one array
        results[test] = np.zeros((2, R))
    for r in tqdm(range(R)):
        # sample predictions and labels from experiments
        p_probs, y_labels = experiment(N, M, K, u)
        # run test
        for test in tests:
            minstat, p_val, stats = tests["test"](
                p_probs=p_probs, y_labels=y_labels, params=tests[test]["params"]
            )
            results[test][:, r] = np.array([minstat, p_val])

    return results
