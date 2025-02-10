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
    create_config_proper_losses,
)
from ensemblecalibration.data.experiments_cal_test import get_experiment
from ensemblecalibration.cal_test import (
    npbe_test_ensemble,
    npbe_test_ensemble_v2,
    npbe_test_ensemble_v0,
)
from ensemblecalibration.utils.helpers import save_results, make_serializable
from ensemblecalibration.utils.plot_functions import plot_error_analysis


def _get_config_from_parser(args: dict, proper_losses: bool = True):

    if proper_losses:
        config = create_config_proper_losses(
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
            device=args.device,
        )
    else:
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
            device=args.device,
        )
    return config


def _simulation_h0(
    dict_tests, n_resamples: int, alpha: list, x_dep: bool = True, verbose: bool = False
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
                    verbose=verbose,
                    use_val = True,
                    use_test = True
                )[0],
                dtype=np.float64,
            )
    for test in dict_tests:
        results[test] = results[test] / n_resamples
        print(f" Results for test {test}: {results[test]}")

    res = []
    for r in results:
        res.append(list(results[r]))

    return res


def _simulation_h1(
    dict_tests,
    n_resamples: int,
    alpha: list,
    setting: int | None,
    deg_h1: float | None = None,
    verbose: bool = False,
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
                    verbose=verbose,
                    use_val = True,
                    use_test = True
                )[0]
            )
    for test in dict_tests:
        results[test] = results[test] / n_resamples
        print(f" Results for test {test}: {results[test]}")

    res = []
    for r in results:
        res.append(list(results[r]))

    return res


def is_data_encapsulated(data):
    """
    function which checks if the data is encapsulated (list of lists of lists)
    """
    # Check if data is a list
    if isinstance(data, list) and len(data) > 0:
        # Check if the first element is a list
        if isinstance(data[0], list):
            # Check if the elements inside the first element are also lists
            if len(data[0]) > 0 and isinstance(data[0][0], list):
                # Data is encapsulated (list of lists of lists)
                return True
            else:
                # Data is not encapsulated (list of lists)
                return False
    return False


def save_results(results_list, save_dir, file_name: str, col_names: list):
    """saves a list of results to a csv file. The list has to be in an encapsulated format

    Parameters
    ----------
    results_list : list
        list of results
    save_dir : str
        directory where the results will
    file_name : str
        name of the file
    col_names : list
        list of column names
    """

    # create directory for results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # filename
    # file_name = prefix + "_{}.csv".format(n_resamples)
    save_dir_file = os.path.join(save_dir, file_name)
    # encapulated data if needed
    if not is_data_encapsulated(results_list):
        results_list = [results_list]
    # create dataframe
    results_df = pd.DataFrame(results_list)
    # colnames = [t for t in config]
    # print(colnames)
    # print(results_df.columns)
    results_df.columns = col_names
    # save results
    results_df.to_csv(save_dir_file, index=False)

    return results_df


def main_t1_t2(args):
    cal_test_functions = {
        "npbe_test_ensemble": npbe_test_ensemble,
        "npbe_test_ensemble_v2": npbe_test_ensemble_v2,
        "npbe_test_ensemble_v0": npbe_test_ensemble_v0,
    }
    args.cal_test = cal_test_functions[args.cal_test]

    # get parameters from parser
    n_resamples = args.n_resamples
    config = _get_config_from_parser(args)
    # print keys of config
    print(f"Config keys: {config.keys()}")
    # select keys args.miscal_stats if list is not None
    if args.miscal_stats:
        # select only the keys that are in the list (as strings)
        config = {k: config[k] for k in args.miscal_stats}
    prefix = args.prefix
    results_dir = args.results_dir
    device = args.device
    print(f"Device: {device}")
    verbose = args.verbose
    if args.types_test:
        test_types = args.types_test
    else:
        test_types = ["t1", "t2"]
    names_tests = [t for t in config]

    # create directory for results
    exp_name = config[list(config.keys())[0]]["experiment"]
    experiment = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = f"{results_dir}/{exp_name}/" + f"{experiment}"
    # check if "t1" is in the list of test types
    if "t1" in test_types:
        """
        H0 simulation, first without x-dependency and then with x-dependency
        """
        results_h0 = []
        alpha = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]

        print("Start H0 simulation...")
        # first no x-dependecy
        print("no x-dependency..")
        print(f"verbose: {verbose}")
        res = _simulation_h0(
            dict_tests=config,
            n_resamples=n_resamples,
            x_dep=False,
            alpha=alpha,
            verbose=verbose,
        )
        # save in csv file
        _ = save_results(
            res,
            save_dir,
            "_h0_no_x_dep_" + prefix + "_{}.csv".format(n_resamples),
            names_tests,
        )

        # append to list with all results
        results_h0.append(res)
        # then with x-dependency
        print("x-dependency..")
        res = _simulation_h0(
            dict_tests=config,
            n_resamples=n_resamples,
            x_dep=True,
            alpha=alpha,
            verbose=verbose,
        )

        # save results separately in experiment folder
        _ = save_results(
            res,
            save_dir,
            file_name="_h0_x_dep_" + prefix + "_{}.csv".format(n_resamples),
            col_names=names_tests,
        )

        results_h0.append(res)
        results_df = save_results(
            results_h0,
            save_dir,
            file_name="h0_" + prefix + "_{}.csv".format(n_resamples),
            col_names=names_tests,
        )

        # plot results
        print(f"list of errors: {results_df.columns}")
        fig = plot_error_analysis(
            results_df,
            list_errors=results_df.columns,
            figsize=(8, 11),
            list_col_titles=[r"$H_{0,1}$", r"$H_{0,2}$"],
            type_1=True,
            alpha=np.array(alpha),
        )
        fig.savefig(save_dir + "/error_analysis_t1.png", bbox_inches="tight", dpi=300)

    if "t2" in test_types:
        """
        H1 simulation
        """

        print("Start H1 simulation...")
        # for setting in range(1, args.n_settings + 1):
        degs_h1 = [0.02, 0.08, 0.1]
        # set alpha differently for h1
        alpha_h1 = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        results_h1 = []
        for deg_h1 in degs_h1:
            res = _simulation_h1(
                dict_tests=config,
                n_resamples=n_resamples,
                alpha=alpha_h1,
                setting=None,
                deg_h1=deg_h1,
                verbose=verbose,
            )
            # save results separately in experiment folder
            _ = save_results(
                res,
                save_dir,
                file_name=f"_h1_{deg_h1}_" + prefix + "_{}.csv".format(n_resamples),
                col_names=names_tests,
            )
            results_h1.append(res)
        # save results for h1
        df_results = save_results(
            results_h1,
            save_dir,
            file_name="h1_" + prefix + "_{}.csv".format(n_resamples),
            col_names=names_tests,
        )

        # plot results
        fig = plot_error_analysis(
            df_results,
            list_errors=df_results.columns,
            figsize=(8, 11),
            list_col_titles=[r"$H_{1,1}$", r"$H_{1,2}$", r"$H_{1,3}$"],
            type_1=False,
            alpha=np.array(alpha_h1),
        )
        fig.savefig(save_dir + "/error_analysis_t2.png", bbox_inches="tight", dpi=300)
    # save config
    save_dir_config = os.path.join(save_dir, "config.pkl")
    # save as csv also
    save_dir_config_json = os.path.join(save_dir, "config.json")
    # save config  as json
    with open(save_dir_config_json, "w") as f:
        config_s = make_serializable(config)
        json.dump(config_s, f)
    with open(save_dir_config, "wb") as f:
        pickle.dump(config, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create configuration for calibration test"
    )
    parser.add_argument(
        "--exp_name", type=str, default="gp", help="name of the experiment"
    )
    parser.add_argument(
        "--cal_test",
        type=str,
        default="npbe_test_ensemble_v2",
        help="calibration test",
    )
    parser.add_argument("--miscal_stats", "--names-list", nargs="+", default=[])
    parser.add_argument("--optim", type=str, default="mlp", help="optimization method")
    parser.add_argument("--n_samples", type=int, default=2000, help="number of samples")
    parser.add_argument(
        "--n_resamples", type=int, default=100, help="number of resamples"
    )
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--n_members", type=int, default=10, help="number of members")
    parser.add_argument("--n_epochs", type=int, default=250, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--patience", type=int, default=100, help="patience")
    parser.add_argument(
        "--hidden_layers", type=int, default=3, help="number of hidden layers"
    )
    parser.add_argument(
        "--reg", type=bool, default=False, help="adds extra termn in the loss"
    )
    parser.add_argument("--hidden_dim", type=int, hiddendefault=16, help="hidden dimension")
    parser.add_argument("--x_dep", type=bool, default=True, help="x_dep")
    parser.add_argument("--deg", type=int, default=2, help="degree")
    parser.add_argument(
        "--x_bound", type=list, default=[0.0, 5.0], help="range of the instance values"
    )
    parser.add_argument(
        "--prefix", type=str, default="results_dirichlet_mlp_t1t2_10_10_100"
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--verbose", type=bool, default=False, help="verbose")
    parser.add_argument("--types_test", "--types-list", nargs="+", default=[])
    args = parser.parse_args()

    main_t1_t2(args)
