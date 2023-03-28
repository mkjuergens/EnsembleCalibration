import sys, os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.stats import multinomial

sys.path.append("../..")
from ensemblecalibration.calibration.iscalibrated import is_calibrated
from ensemblecalibration.calibration.config import (
    config_tests_new_cobyla_2d,
    config_tests_new_cobyla_1d,
    config_tests_new_neldermead_1d,
    config_tests_new_neldermead_2d,
)
from ensemblecalibration.calibration.config import (
    config_tests_cobyla_1d,
    config_tests_cobyla_2d,
    config_tests_neldermead_1d,
    config_tests_neldermead_2d,
    config_new_mlp
)
from ensemblecalibration.calibration.cal_test_new import _npbe_test_new_alpha
from ensemblecalibration.calibration.experiments import (
    experiment_h0,
    experiment_h1,
    experiment_h0_feature_dependency,
    experiment_h1_feature_dependecy,
)
def _simulation_h0(
    tests,
    N: int,
    M: int,
    K: int,
    R: int,
    u: float,
    alpha: float,
    experiment=experiment_h0,
):
    """Simulation of the test if the Null Hypothesis is true.

    Parameters
    ----------
    tests : dictionary
        dictionary with tests and parameters
    N : int
        nuber of features
    M : int
        number of point predictors
    K : int
        number of different classes to predcit the probability for
    R : int
        number of resamplings/different datasets
    u : float
        parameter which controls the uncertainty/spread in the datasets
    alpha : float
        confidence level of the test

    Returns
    -------
    dictionary
        dictionary containing results
    """

    results = {}
    for test in tests:
        results[test] = np.zeros(len(alpha))
    for _ in tqdm(range(R)):
        P, y = experiment(N, M, K, u)
        for test in tests:
            results[test] += np.array(
                tests[test]["test"](P, y, alpha, tests[test]["params"])
            )
    for test in tests:
        # calculate mean
        results[test] = results[test] / R
    return results


def _simulation_ha(
    tests,
    N: int,
    M: int,
    K: int,
    R: int,
    u: float,
    alpha: float,
    random: bool = False,
    experiment=experiment_h1,
):
    """Simulation of the test in a setting where the alternative hypothesis is true.

    Parameters
    ----------
    tests : dictionary
        _description_
    N : int
        number of samples for each dataset
    M : int
        number of point predictors
    K : int
        number of (different) classes
    R : int
        number of resamplings/different datasets
    u : float
        parameter which controls the uncertainty/ spread in the sampled datasets
    alpha : float
        confidence level of the tests
    random: bool
        whether to randomly chose the corner the outside distribution is sampled from

    Returns
    -------
    dictionary
        results of the test
    """
    results = {}
    for test in tests:
        results[test] = np.zeros(len(alpha))
    for r in tqdm(range(R)):
        P, y = experiment(N, M, K, u, random=random)
        for test in tests:
            results[test] += 1 - np.array(
                tests[test]["test"](P, y, alpha, tests[test]["params"])
            )
    for test in tests:
        results[test] = results[test] / R

    return results


def main_t1_t2(args, test_h1: bool = True, results_dir: str = "results"):
    sys.stdout = None
    results = []
    alpha = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
    N = args.N
    M = args.M
    K = args.K
    u = args.u
    R = args.R
    experiments = args.experiments
    sampling_method = args.sampling
    tests = args.config

    # change sampling method in configuration
    # for i in range(len(list(tests.keys()))):
    #   tests[list(tests.keys())[i]]["params"]["sampling"] = sampling_method

    os.makedirs(results_dir, exist_ok=True)

    # file name defining parameters of test
    # dim defines whether x dependency or not
    dim = "2d" if tests[list(tests.keys())[0]]["params"]["x_dependency"] else "1d"
    # optim defines optimization method used
    optim = tests[list(tests.keys())[0]]["params"]["optim"]
    # pattern defines whether new or old structure of test is used
    pattern = (
        "new" if tests[list(tests.keys())[0]]["test"] == _npbe_test_new_alpha else "old"
    )
    file_name = "results_experiments_t1t2_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        pattern, N, M, K, R, u, dim, optim
    )
    print(f"File name under which results are saved: {file_name}")

    # check which experiments shall be run
    if experiments == "new":
        print("Use new experiments..")
        exp_h0 = experiment_h0_feature_dependency
        exp_ha = experiment_h1_feature_dependecy
    elif experiments == "old":
        exp_h0 = experiment_h0
        exp_ha = experiment_h1
    else:
        raise NotImplementedError

    save_dir = os.path.join(results_dir, file_name)

    print("Start H0 simulation")
    res_h0 = _simulation_h0(tests, N, M, K, R, u, alpha,experiment=exp_h0)
    res = []
    for r in res_h0:
        res.append(list(res_h0[r]))
    results.append(res)

    # tests for when h1 hypothesis is true
    if test_h1:
        print("Start Ha simulation")
        res_h11 = _simulation_ha(tests, N, M, K, R, u, alpha)
        res = []
        for r in res_h0:
            res.append(list(res_h11[r]))
        results.append(res)
        print("Start second Ha simulation")
        res_h12 = _simulation_ha(tests, N, M, K, R, u, alpha, random=True, experiment=exp_ha)
        res = []
        for r in res_h0:
            res.append(list(res_h12[r]))
        results.append(res)

    results_df = pd.DataFrame(results)
    colnames = [t for t in tests]
    results_df.columns = colnames
    results_df.to_csv(save_dir, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiments for type I and type II error in function of alpha"
    )
    # data args
    parser.add_argument("-N", dest="N", type=int, default=100)
    parser.add_argument("-M", dest="M", type=int, default=10)
    parser.add_argument("-K", dest="K", type=int, default=10)
    parser.add_argument("-u", dest="u", type=float, default=0.01)
    parser.add_argument("-R", dest="R", type=int, default=1000)
    parser.add_argument("-experiments",dest="experiments", default="new", type=str)
    parser.add_argument("-sampling", dest="sampling", type=str, default="lambda")
    parser.add_argument(
        "-config", dest="config", type=dict, default=config_new_mlp
    )
    args = parser.parse_args()
    main_t1_t2(args, test_h1=True)
