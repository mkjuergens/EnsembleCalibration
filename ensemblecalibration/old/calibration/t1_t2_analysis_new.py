import sys, os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("../..")

from ensemblecalibration.calibration.config import config_new_mlp_binary
from ensemblecalibration.calibration.cal_test_new import _npbe_test_mlp_new_alpha
from ensemblecalibration.nn_training.sampling import (
    experiment_binary_nn,
    binary_experiment_cone_h0,
    binary_experiment_cone_h1,
)


def _simulation_h0_new_binary(
    tests,
    N: int,
    M: int,
    R: int,
    alpha: float,
    experiment=experiment_binary_nn,
):
    """Simulation of the test if the Null Hypothesis is true, for a binary classification setting.

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
    Returns
    -------
    dictionary
        dictionary containing results
    """

    results = {}
    for test in tests:
        print(f"Running test {test}")
        results[test] = np.zeros(len(alpha))
        for _ in tqdm(range(R)):
            x, p_probs, l_weights, p_bar, y_labels = experiment(
                N,
                M,
            )
            print(f'number of instances: {x.shape[0]}') 
            results[test] += np.array(
                tests[test]["test"](x, p_probs, y_labels, alpha, tests[test]["params"])
            )
        # calculate mean
        results[test] = results[test] / R

    return results


def main_t1_t2(
    args,
    test_h1: bool = True,
    results_dir: str = "results",
    experiment_h0=binary_experiment_cone_h0,
    experiment_h1=binary_experiment_cone_h1,
):
    # sys.stdout = None
    results = []
    alpha = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
    N = args.N
    M = args.M
    R = args.R
    tests = args.config

    os.makedirs(results_dir, exist_ok=True)

    # file name defining parameters of test
    # dim defines whether x dependency or not
    # optim defines optimization method used
    optim = tests[list(tests.keys())[0]]["params"]["optim"]
    # pattern defines whether new or old structure of test is used
    file_name = "results_experiments_t1t2_mlp_binary_{}_{}_{}_{}.csv".format(
        N, M, R, optim
    )
    print(f"File name under which results are saved: {file_name}")

    save_dir = os.path.join(results_dir, file_name)

    print("Start H0 simulation")
    res_h0 = _simulation_h0_new_binary(
        tests, N, M, R, alpha=alpha, experiment=experiment_h0
    )
    res = []
    for r in res_h0:
        res.append(list(res_h0[r]))
    results.append(res)

    results_df = pd.DataFrame(results)
    colnames = [t for t in tests]
    results_df.columns = colnames
    results_df.to_csv(save_dir, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiments for type I and type II error in function of alpha"
    )

    parser.add_argument("-N", dest="N", type=int, default=10000)
    parser.add_argument("-M", dest="M", type=int, default=3)
    parser.add_argument("-R", dest="R", type=int, default=10)
    parser.add_argument(
        "-config", dest="config", type=dict, default=config_new_mlp_binary
    )
    args = parser.parse_args()
    main_t1_t2(args, test_h1=True)
