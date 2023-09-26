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


if __name__ == "__main__":
    results = _simulation_h0_cone(config_new_mlp_binary,
                                  20000,
                                  10,
                                  [0, 0.5, 1]
                                  )