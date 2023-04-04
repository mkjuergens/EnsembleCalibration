import random
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm

from ensemblecalibration.calibration.experiments import experiment_h0, experiment_h1
from ensemblecalibration.calibration.cal_test_new import calculate_min_new
from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar
from ensemblecalibration.sampling import multinomial_label_sampling
from ensemblecalibration.calibration.config import config_p_value_analysis


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
    p_val, minstat
        p-value and value of the "optimized" statistic
    """

    # array fodr saving statistics in the bootstrapping iterations
    stats = np.zeros(params["n_resamples"])
    #calculate minimum value of evaluations of the statistics
    minstat, weights_l = calculate_min_new(p_probs, y_labels, params=params)
    # do boootstrapping
    for b in range(params["n_resamples"]):
        p_b = random.sample(p_probs.tolist(), p_probs.shape[0])
        p_b = np.stack(p_b)
        # calculate convex combination of predictions
        if params["x_dependency"]:
            p_bar_b = calculate_pbar(weights_l=weights_l, P=p_b, reshape=True,
                                n_dims=2)
        else:
            p_bar_b = calculate_pbar(weights_l=weights_l, P=p_b, reshape=False, n_dims=1)
        p_bar_b = np.trunc(p_bar_b*10**3)/(10**3)
        p_bar_b = np.clip(p_bar_b, 0, 1)
        # sample labels from categorical distributiuon induced by p_bar_b
        y_b = np.apply_along_axis(multinomial_label_sampling, 1, p_bar_b)
        stats[b] = params["test"](p_bar_b, y_labels, params)

    # calculate empirical distribution
    ecdf = ECDF(stats)
   # print(stats)
    # p value: 1 - F(minstat)
    p_val = 1 - ecdf(minstat)

    return minstat, p_val, stats


def _simulation_pvals(tests, N: int, M: int, K: int, R: int, u: float,
                      experiment=experiment_h0):
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
        number of calasses
    R : int
        number of dataset/resampling iterations
    u : float
        parameter describing the "uncertainty"/spread for the generated datasets
    experiment : _type_, optional
        _description_, by default experiment_h0

    Returns
    -------
    results
        dictionary containing matrices of p-values and svalues of statistics for each test
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
            minstat, p_val, stats = npbe_test_p_values(p_probs=p_probs, y_labels=y_labels, 
                                                params=tests[test]["params"])
            results[test][:, r] = np.array([minstat, p_val])
    
    return results, 

if __name__ == "__main__":
    results = _simulation_pvals(tests=config_p_value_analysis, N=100, M=5, K=3, R=100, u=0.01,
                               experiment=experiment_h0)
    print(results)
    
    
        
