import sys, os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append("../..")

from ensemblecalibration.calibration.config import config_new_mlp_binary
from ensemblecalibration.nn_training.binary_experiments import cone_experiment_h0, cone_experiment_h1


def _simulation_h0_cone(
        tests,
        n_samples: int,
        n_resamples: int,
        alpha: list,
        experiment=cone_experiment_h0
):
    results = {}
    for test in tests:
        print(f"Running test {test}")
        results[test] = np.zeros(len(alpha))
        for _ in tqdm(range(n_resamples)):
            x_inst, p_probs, y_labels, p_bar = experiment(n_samples, deg_fct=1)
            results[test] += np.array(
                tests[test]["test"](x_inst, p_probs, y_labels, alpha, tests[test]["params"])
            )
        results[test] = results[test] / n_resamples

    return results

def _simulation_h1_cone(
        tests,
        n_samples: int,
        n_resamples: int,
        alpha: list,
        experiment=cone_experiment_h1,
        frac_in: float = 0.0
):
    
    results = {}
    for test in tests:
        results[test] = np.zeros(len(alpha))

    for _ in tqdm(range(n_resamples)):
        # generate data for the test run
        x_inst, p_probs, y_labels, p_bar = experiment(n_samples, frac_in=frac_in)
        # save results (1 - p-value) for each test
        for test in tests:
            results[test] += 1 - np.array(
                tests[test]["test"](x_inst, p_probs, y_labels, alpha, tests[test]["params"])
            )
    for test in tests:
        results[test] = results[test] / n_resamples

    return results

def main_t1_t2(
        args,
        results_dir: str = "results",
        experiment_h0 = cone_experiment_h0,
        experiment_h1 = cone_experiment_h1,
):
    results = []
    alpha = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
    n_samples = args.N
    n_resamples = args.R
    tests = args.config
    prefix = args.prefix

    # create directory for results
    os.makedirs(results_dir, exist_ok=True)

    # specify name under which file is saved
    file_name = prefix + "{}_{}.csv".format(
        n_samples, n_resamples
    )
    save_dir = os.path.join(results_dir, file_name)


    print("Start H0 simulation...")
    res_h0 = _simulation_h0_cone(tests, n_samples=n_samples, n_resamples=n_resamples,
                                 alpha=alpha, experiment=experiment_h0)
    res = []
    for r in res_h0:
        res.append(list(res_h0[r]))
    results.append(res)

    print("Start H1 simulation...")
    res_h1 = _simulation_h1_cone(tests, n_samples=n_samples, n_resamples=n_resamples,
                                    alpha=alpha, experiment=experiment_h1)
    res = []
    for r in res_h1:
        res.append(list(res_h1[r]))
    results.append(res)

    # save in csv file
    results_df = pd.DataFrame(results)
    colnames = [t for t in tests]
    results_df.columns = colnames
    results_df.to_csv(save_dir, index=False)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiments for type I and type II error in function of alpha"
    )
    # data args
    parser.add_argument("-N", dest="N", type=int, default=15000)
    parser.add_argument("-R", dest="R", type=int, default=100)
    parser.add_argument(
        "-config", dest="config", type=dict, default=config_new_mlp_binary
    )
    parser.add_argument("-prefix", type=str, default="results_cone_experiment_t1t2_")
    args = parser.parse_args()
    main_t1_t2(args)
