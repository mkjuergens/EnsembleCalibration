import os
from tqdm import tqdm
import json
import pickle
import numpy as np
import pandas as pd
import argparse
import datetime

from ensemblecalibration.config.config_cal_test import (
    config_binary_classification_mlp,
    config_binary_const_weights,
    config_dirichlet_mlp,
    config_dirichlet_const_weights,
)
from ensemblecalibration.data.experiments import get_experiment


def _get_config(optim: str, exp_name: str):
    if optim == "mlp":
        if exp_name == "dirichlet":
            return config_dirichlet_mlp
        elif exp_name == "gp":
            return config_binary_classification_mlp
    elif optim == "cobyla":
        if exp_name == "dirichlet":
            return config_dirichlet_const_weights
        elif exp_name == "gp":
            return config_binary_const_weights
    else:
        raise ValueError("Unknown optimization method.")


def _simulation_h0_binary(
    dict_tests, n_resamples: int, alpha: list, x_dep: bool = True
):
    results = {}
    for test in dict_tests:
        results[test] = np.zeros(len(alpha))
    for _ in tqdm(range(n_resamples)):
        data, _, _ = get_experiment(config=dict_tests[test], h0=True, x_dep=x_dep)
        for test in dict_tests:
            results[test] += np.array(
                dict_tests[test]["test"](
                    alpha=alpha,
                    x_inst=data[0],
                    p_preds=data[1],
                    y_labels=data[3],
                    params=dict_tests[test]["params"],
                )[0],
                dtype=np.float64,
            )
    for test in dict_tests:
        results[test] = results[test] / n_resamples
        print(f" Results for test {test}: {results[test]}")

    return results


def _simulation_h1_binary(
    dict_tests,
    n_resamples: int,
    alpha: list,
    setting: int,
):

    results = {}
    for test in dict_tests:
        results[test] = np.zeros(len(alpha))

    for _ in tqdm(range(n_resamples)):
        # generate data for the test run
        data, _, _ = get_experiment(config=dict_tests[test], h0=False, setting=setting)
        # save results (1 - p-value) for each test
        for test in dict_tests:
            results[test] += 1 - np.array(
                dict_tests[test]["test"](
                    alpha,
                    x_inst=data[0],
                    p_preds=data[1],
                    y_labels=data[3],
                    params=dict_tests[test]["params"],
                )[0]
            )
    for test in dict_tests:
        results[test] = results[test] / n_resamples
        print(f" Results for test {test}: {results[test]}")

    return results


def main_t1_t2(args):
    results = []
    alpha = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]

    n_resamples = args.R
    # get config
    config = _get_config(args.optim, args.exp)
    prefix = args.prefix
    results_dir = args.results_dir
    # change x_dep parameter in config to the value from args
    # for test in config:
    #    config[test]["params"]["x_dep"] = args.x_dep

    # create directory for results
    exp_name = config[list(config.keys())[0]]["experiment"]
    experiment = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = f"{results_dir}/{exp_name}/" + f"{experiment}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # specify name under which file is saved
    file_name = prefix + "_{}.csv".format(n_resamples)
    save_dir_file = os.path.join(save_dir, file_name)

    # save config
    save_dir_config = os.path.join(save_dir, "config.pkl")
    with open(save_dir_config, "wb") as f:
        pickle.dump(config, f)

    print("Start H0 simulation...")
    # first no x-dependecy
    print("no x-dependency..")
    res_h0_no_x_dep = _simulation_h0_binary(
        dict_tests=config, n_resamples=n_resamples, x_dep=False, alpha=alpha
    )
    res = []
    for r in res_h0_no_x_dep:
        res.append(list(res_h0_no_x_dep[r]))
    results.append(res)
    # then with x-dependency
    print("x-dependency..")
    res_h0_x_dep = _simulation_h0_binary(
        dict_tests=config, n_resamples=n_resamples, x_dep=True, alpha=alpha
    )
    res = []
    for r in res_h0_x_dep:
        res.append(list(res_h0_x_dep[r]))
    results.append(res)

    print("Start H1 simulation...")
    res_h1_s1 = _simulation_h1_binary(
        dict_tests=config,
        n_resamples=n_resamples,
        alpha=alpha,
        setting=1,
    )
    res = []
    for r in res_h1_s1:
        res.append(list(res_h1_s1[r]))
    results.append(res)

    res_h1_s2 = _simulation_h1_binary(
        dict_tests=config,
        n_resamples=n_resamples,
        alpha=alpha,
        setting=2,
    )
    res = []
    for r in res_h1_s2:
        res.append(list(res_h1_s2[r]))
    results.append(res)

    res_h1_s3 = _simulation_h1_binary(
        dict_tests=config,
        n_resamples=n_resamples,
        alpha=alpha,
        setting=3,
    )
    res = []
    for r in res_h1_s3:
        res.append(list(res_h1_s3[r]))
    results.append(res)

    # save in csv file
    results_df = pd.DataFrame(results)
    colnames = [t for t in config]
    results_df.columns = colnames
    results_df.to_csv(save_dir_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiments for type I and type II error in function of alpha"
    )
    # data args
    parser.add_argument("-R", dest="R", type=int, default=100)
    parser.add_argument("-optim", type=str, default="mlp")
    parser.add_argument("-exp", type=str, default="dirichlet")
    parser.add_argument("-prefix", type=str, default="results_binary_t1t2")
    parser.add_argument("-results_dir", type=str, default="results")
    # parser.add_argument("-x_dep", type=bool, default=True)
    args = parser.parse_args()
    main_t1_t2(args)
