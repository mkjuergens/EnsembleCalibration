import os
import sys
import argparse

sys.path.append("../")
import numpy as np
import pandas as pd

from ensemblecalibration.calibration.calibration_estimates.distances import (
    w1_distance,
    tv_distance,
    mmd,
    avg_euclidean_distance,
    avg_kl_divergence,
)
from ensemblecalibration.calibration.config import config_p_value_analysis
from ensemblecalibration.calibration.p_value_analysis import (
    distance_analysis_npbe,
    npbe_test_null_hypothesis,
    npbe_test_distances_two_lambdas,
    distance_analysis_const_preds
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

def run_dummy_analysis_distances(
        tests: dict, 
        n_iters: int,
        n_instances: int,
        n_classes: int,
        dist_fct=w1_distance,
        save_path: str = "results/"
):
    file_name = f"results_p_value_distance_analysis_dummy_{n_iters}_{n_instances}_{n_classes}_{dist_fct.__name__}.csv"
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)

    res = {}

    for test in tests:
        print(f"Running test {test}")
        p_vals, dists, stats = distance_analysis_const_preds(n_instances=n_instances,
                                                             n_classes=n_classes,
                                                             params=tests[test]["params"],
                                                             dist_fct=dist_fct,
                                                                n_iters=n_iters)
        res[test] = (p_vals, dists, stats)
    
    df = pd.DataFrame(res)
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
    file_name = f"results_p_value_distance_analysis_{n_iters}_{n_features}_{n_predictors}_{n_classes}_{dist_fct.__name__}.csv"
    os.makedirs(results_path, exist_ok=True)
    file_path = os.path.join(results_path, file_name)

    res = {}
    p_probs, _ = experiment(N=n_features, M=n_predictors, K=n_classes, u=0.01)
    for test in tests:
        print(f"Running test {test}")
        p_vals, dists, stats = distance_analysis_npbe(
            p_probs=p_probs,
            params=tests[test]["params"],
            dist_fct=dist_fct,
            n_iters=n_iters,
        )

        res[test] = (p_vals, dists, stats)

    df = pd.DataFrame(res)
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
    parser.add_argument("--save_dir", type=str, default="results/")
    parser.add_argument(
        "--dist_fct",
        type=str,
        default="w1_distance",
        choices=["w1_distance", "tv_distance", "euclidean_distance", "mmd", "kl"],
    )

    args = parser.parse_args()

    # set measure of distance
    if args.dist_fct == "w1_distance":
        dist_fct = w1_distance
    elif args.dist_fct == "tv_distance":
        dist_fct = tv_distance
    elif args.dist_fct == "euclidean_distance":
        dist_fct = avg_euclidean_distance
    elif args.dist_fct == "mmd":
        dist_fct = mmd
    elif args.dist_fct == "kl":
        dist_fct = avg_kl_divergence
    else:
        raise NotImplementedError

    experiment = experiment_h0
    SAVE_DIR = "results/"
    conf = config_p_value_analysis

    df_dummy = run_dummy_analysis_distances(
        tests=conf,
        n_iters=args.n_iters,
        n_instances=args.n_features,
        n_classes=args.n_classes,
        dist_fct=dist_fct,
        save_path=args.save_dir,
    )

    res_dists, df = run_analysis_distances(
        tests=conf,
        n_iters=args.n_iters,
        n_features=args.n_features,
        n_predictors=args.n_predictors,
        n_classes=args.n_classes,
        dist_fct=dist_fct,
        experiment=experiment,
        results_path=args.save_dir,
    )

    df_h0 = run_analysis_h0(
        tests=conf,
        n_iters=args.n_iters,
        n_features=args.n_features,
        n_predictors=args.n_predictors,
        n_classes=args.n_classes,
        alpha=args.alpha,
        results_path=args.save_dir,
    )
