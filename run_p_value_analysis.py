import os
import sys
import argparse

sys.path.append("../")
import numpy as np
import pandas as pd

from ensemblecalibration.calibration.calibration_estimates.distances import (
    w1_distance,
    tv_distance,
    l2_distance,
    mmd,
    euclidean_distance
)
from ensemblecalibration.calibration.config import config_p_value_analysis
from ensemblecalibration.calibration.p_value_analysis import (
    distance_analysis_npbe,
    npbe_test_null_hypothesis,
)
from ensemblecalibration.calibration.experiments import (
    experiment_h0,
    experiment_h1,
    experiment_h0_feature_dependency,
    experiment_h1_feature_dependecy,
)


def run_analysis_h0(
    tests: dict,
    n_iters: int,
    n_features: int,
    n_predictors: int,
    n_classes: int,
    alpha: float,
    results_path: str = "results/",
):
    file_name = f"results_p_value_analysis_h0_{n_iters}_{n_features}_{n_predictors}_{n_classes}.csv"
    os.makedirs(results_path, exist_ok=True)
    file_path = os.path.join(results_path, file_name)

    res = []
    for test in tests:
        print(f"Running test {test}")
        p_vals_h0, stats, decisions = npbe_test_null_hypothesis(
            params=tests[test]["params"],
            alpha=alpha,
            n_iters=n_iters,
            n_classes=n_classes,
            n_instances=n_features,
        )
        res.append(list(p_vals_h0))

    df = pd.DataFrame(res)
    df = df.T
    df.columns = [t for t in tests]
    df.to_csv(file_path, index=False)
    return df


def run_analysis_distances(
    tests: dict,
    n_iters: int,
    n_features: int,
    n_predictors: int,
    n_classes: int,
    dist_fct=w1_distance,
    experiment=experiment_h0,
    results_path: str = "results/",
):
    file_name = f"results_p_value_analysis_distances_{n_iters}_{n_features}_{n_predictors}_{n_classes}_{dist_fct.__name__}.csv"
    os.makedirs(results_path, exist_ok=True)
    file_path = os.path.join(results_path, file_name)

    res = []
    p_probs, _ = experiment(N=n_features, M=n_predictors, K=n_classes, u=0.01)
    for test in tests:
        print(f"Running test {test}")
        p_vals, dists = distance_analysis_npbe(
            p_probs=p_probs,
            params=tests[test]["params"],
            dist_fct=dist_fct,
            n_iters=n_iters,
        )

        res.append(list(p_vals))
        res.append(list(dists))

    df = pd.DataFrame(res)
    df = df.T
    df.columns = [t for t in tests] * 2
    # save results to csv
    df.to_csv(file_path, index=False)

    return res, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iters", type=int, default=1000)
    parser.add_argument("--n_features", type=int, default=100)
    parser.add_argument("--n_predictors", type=int, default=10)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("save_dir", type=str, default="results/")

    args = parser.parse_args()

    dist_fct = euclidean_distance
    experiment = experiment_h0
    SAVE_DIR = "results/"
    conf = config_p_value_analysis

    res_dists, df = run_analysis_distances(
        tests=config_p_value_analysis,
        n_iters=parser.n_iters,
        n_features=parser.n_features,
        n_predictors=parser.n_predictors,
        n_classes=parser.n_classes,
        dist_fct=dist_fct,
        experiment=experiment,
        results_path=SAVE_DIR,
    )

    df_h0 = run_analysis_h0(
        tests=conf,
        n_iters=parser.n_iters,
        n_features=parser.n_features,
        n_predictors=parser.n_predictors,
        n_classes=parser.n_classes,
        alpha=parser.alpha,
        results_path=SAVE_DIR,
    )