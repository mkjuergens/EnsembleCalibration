import os
from tqdm import tqdm
import json
import pickle
import numpy as np
import pandas as pd
import argparse
import datetime

from ensemblecalibration.cal_test import npbe_test_ensemble
from ensemblecalibration.config.config_cal_test import config_binary_clasification
from ensemblecalibration.data.experiments import get_experiment


def _simulation_h0_binary(
    dict_tests,
    n_resamples: int,
    alpha: list,
):
    results = {}
    for test in dict_tests:
        results[test] = np.zeros(len(alpha))
    for _ in tqdm(range(n_resamples)):
        data, _, _ = get_experiment(config=dict_tests[test], h0=True)
        for test in dict_tests:
            results[test] += np.array(
                dict_tests[test]["test"](
                    alpha=alpha,
                    x_inst=data[0],
                    p_preds=data[1],
                    y_labels=data[3],
                    params=dict_tests[test]["params"],
                )[0], dtype=np.float64
            )
    for test in dict_tests:
        results[test] = results[test] / n_resamples

    return results


def _simulation_h1_binary(
    dict_tests,
    n_resamples: int,
    alpha: list,
):

    results = {}
    for test in dict_tests:
        results[test] = np.zeros(len(alpha))

    for _ in tqdm(range(n_resamples)):
        # generate data for the test run
        data, _, _ = get_experiment(config=dict_tests[test], h0=False)
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
        print(f"Results: {results}") # TODO: only for debugging
    for test in dict_tests:
        results[test] = results[test] / n_resamples

    return results


def main_t1_t2(
    args
):
    results = []
    alpha = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]

    n_resamples = args.R
    config = args.config.copy()
    prefix = args.prefix
    results_dir = args.results_dir

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
    res_h0 = _simulation_h0_binary(dict_tests=config, n_resamples=n_resamples,alpha=alpha)
    res = []
    for r in res_h0:
        res.append(list(res_h0[r]))
    results.append(res)

    print("Start H1 simulation...")
    res_h1 = _simulation_h1_binary(dict_tests=config, n_resamples=n_resamples,alpha=alpha)
    res = []
    for r in res_h1:
        res.append(list(res_h1[r]))
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
    parser.add_argument(
        "-config", dest="config", type=dict, default=config_binary_clasification
    )
    parser.add_argument("-prefix", type=str, default="results_binary_t1t2")
    parser.add_argument("-results_dir", type=str, default="results")
    args = parser.parse_args()
    main_t1_t2(args)