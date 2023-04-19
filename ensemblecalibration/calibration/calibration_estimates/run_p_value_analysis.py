import numpy as np
import pandas as pd

from ensemblecalibration.calibration.calibration_estimates.distances import (
    w1_distance,
    tv_distance,
    l2_distanceƒ,
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

def run_analysis(n_iters: int, n_features: int, n_predictors: int, n_classes: int,
                 alpha: float, dist_fct=w1_distance, experiment=experiment_h0):
    # save results in a dictionary
    results_h0 = {}
    results_dists = {}

    for test in config_p_value_analysis:
        print(f"Running test {test}")
        p_vals, stats, decisions = npbe_test_null_hypothesis(
            params=config_p_value_analysis[test]["params"],
            alpha=alpha,
            n_iters=n_iters,
            n_classes=n_classes,
            n_instances=n_features,
        )
        results_h0["test"] = {"p_vals_h0": p_vals}

        # distance analysis
        p_probs, _ = experiment(N=n_features, M=n_predictors, K=n_classes, u=0.01)
        p_vals, dists = distance_analysis_npbe(p_probs=p_probs, 
                                               params=config_p_value_analysis[test]["params"],
                                               dist_fct=dist_fct, n_iters=n_iters)
        results_dists["test"] = {"p_vals_dists": p_vals, "dists": dists}


if __name__ == "__main__":
    N_ITERS = 1000
    N_FEATURES = 100
    N_PREDICTORS = 10
    N_CLASSES = 10
    ALPHA = 0.05
    dist_fct = w1_distance
    experiment = experiment_h0

    run_analysis(n_iters=N_ITERS, n_features=N_FEATURES, n_predictors=N_PREDICTORS, 
                 n_classes=N_CLASSES, alpha=ALPHA, dist_fct=dist_fct, experiment=experiment)


