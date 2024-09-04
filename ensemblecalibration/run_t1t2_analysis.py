import os
from tqdm import tqdm
import json
import pickle
import numpy as np
import pandas as pd
import argparse
import datetime

from ensemblecalibration.config.config_cal_test import (
    create_config,
    config_binary_classification_mlp,
    config_binary_classification_cobyla,
    config_dirichlet_mlp,
    config_dirichlet_cobyla,
)
from ensemblecalibration.data.experiments import get_experiment
from ensemblecalibration.cal_test import npbe_test_ensemble


def _get_config_from_parser(args: dict):
    config = create_config(
        exp_name=args.exp_name,
        cal_test=args.cal_test,
        optim=args.optim,
        n_samples=args.n_samples,
        n_resamples=args.n_resamples,
        n_classes=args.n_classes,
        n_members=args.n_members,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
        x_dep=args.x_dep,
        deg=args.deg,
        x_bound=args.x_bound,
    )
    return config


def _get_config(optim: str, exp_name: str):

    if optim == "mlp":
        if exp_name == "dirichlet":
            return config_dirichlet_mlp
        elif exp_name == "gp":
            return config_binary_classification_mlp
    elif optim == "cobyla":
        if exp_name == "dirichlet":
            return config_dirichlet_cobyla
        elif exp_name == "gp":
            return config_binary_classification_cobyla
    else:
        raise ValueError("Unknown optimization method.")


def _simulation_h0(dict_tests, n_resamples: int, alpha: list, x_dep: bool = True):
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


def _simulation_h1(
    dict_tests,
    n_resamples: int,
    alpha: list,
    setting: int | None,
    deg_h1: float | None = None,
):

    results = {}
    for test in dict_tests:
        results[test] = np.zeros(len(alpha))

    for _ in tqdm(range(n_resamples)):
        # generate data for the test run
        data, _, _ = get_experiment(
            config=dict_tests[test], h0=False, setting=setting, deg_h1=deg_h1
        )
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

    n_resamples = args.n_resamples
    config = _get_config_from_parser(args)
    prefix = args.prefix
    results_dir = args.results_dir

    # create directory for results
    exp_name = config[list(config.keys())[0]]["experiment"]
    experiment = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = f"{results_dir}/{exp_name}/" + f"{experiment}"

    print("Start H0 simulation...")
    # first no x-dependecy
    print("no x-dependency..")
    res_h0_no_x_dep = _simulation_h0(
        dict_tests=config, n_resamples=n_resamples, x_dep=False, alpha=alpha
    )
    res = []
    for r in res_h0_no_x_dep:
        res.append(list(res_h0_no_x_dep[r]))
    results.append(res)
    # then with x-dependency
    print("x-dependency..")
    res_h0_x_dep = _simulation_h0(
        dict_tests=config, n_resamples=n_resamples, x_dep=True, alpha=alpha
    )
    res = []
    for r in res_h0_x_dep:
        res.append(list(res_h0_x_dep[r]))
    results.append(res)

    print("Start H1 simulation...")
    # for setting in range(1, args.n_settings + 1):
    degs_h1 = [0.05, 0.25, 0.9]
    for deg_h1 in degs_h1:
        res_h1_s1 = _simulation_h1(
            dict_tests=config,
            n_resamples=n_resamples,
            alpha=alpha,
            setting=None,
            deg_h1=deg_h1,
        )
        res = []
        for r in res_h1_s1:
            res.append(list(res_h1_s1[r]))
        results.append(res)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # specify name under which file is saved
    file_name = prefix + "_{}.csv".format(n_resamples)
    save_dir_file = os.path.join(save_dir, file_name)

    # save config
    save_dir_config = os.path.join(save_dir, "config.pkl")
    with open(save_dir_config, "wb") as f:
        pickle.dump(config, f)

    # save in csv file
    results_df = pd.DataFrame(results)
    colnames = [t for t in config]
    results_df.columns = colnames
    results_df.to_csv(save_dir_file, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create configuration for calibration test"
    )
    parser.add_argument(
        "--exp_name", type=str, default="gp", help="name of the experiment"
    )
    parser.add_argument(
        "--cal_test", type=callable, default=npbe_test_ensemble, help="calibration test"
    )
    parser.add_argument("--optim", type=str, default="mlp", help="optimization method")
    parser.add_argument("--n_samples", type=int, default=1000, help="number of samples")
    parser.add_argument(
        "--n_resamples", type=int, default=100, help="number of resamples"
    )
    parser.add_argument("--n_classes", type=int, default=5, help="number of classes")
    parser.add_argument("--n_members", type=int, default=5, help="number of members")
    parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--patience", type=int, default=100, help="patience")
    parser.add_argument(
        "--hidden_layers", type=int, default=3, help="number of hidden layers"
    )
    parser.add_argument("--hidden_dim", type=int, default=32, help="hidden dimension")
    parser.add_argument("--x_dep", type=bool, default=True, help="x_dep")
    parser.add_argument("--deg", type=int, default=2, help="degree")
    parser.add_argument(
        "--x_bound", type=list, default=[0.0, 5.0], help="range of the instance values"
    )
    parser.add_argument("--prefix", type=str, default="results_dircihlet_mlp_t1t2_new")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    # parser = argparse.ArgumentParser(
    #     description="Experiments for type I and type II error in function of alpha"
    # )
    # # data args
    # parser.add_argument("-R", dest="R", type=int, default=100)
    # parser.add_argument("-optim", type=str, default="mlp")
    # parser.add_argument("-exp", type=str, default="dirichlet")
    # parser.add_argument("-prefix", type=str, default="results_binary_t1t2")
    # parser.add_argument("-results_dir", type=str, default="results")
    # parser.add_argument("-n_settings", type=int, default=2)
    # # parser.add_argument("-x_dep", type=bool, default=True)
    # args = parser.parse_args()
    main_t1_t2(args)
