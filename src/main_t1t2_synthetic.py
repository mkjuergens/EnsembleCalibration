import os
import json
import pickle
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Any, Dict
import argparse
from inspect import signature


from src.config.config_synthetic import ExperimentConfig, build_objective_map
from src.data.synthetic.sampling import (
    sample_dirichlet_experiment,
    sample_gp_experiment,
)
from src.data import MLPDataset

from src.cal_test import (
    npbe_test_ensemble_v0,
    npbe_test_ensemble_v2,
    npbe_test_ensemble,
)
from src.utils.helpers import (
    save_results_rowwise,
    make_serializable,
    call_with_filtered_kwargs,
)
from src.utils.plot_functions import (
    read_and_plot_error_analysis_full,
)

BOOTSTRAP_TESTS = {
    "v0": npbe_test_ensemble_v0,
    "v2": npbe_test_ensemble_v2,
}

# -----------------------------------------------------------------------------
# Simulation runner
# -----------------------------------------------------------------------------


def simulate(
    cfg: ExperimentConfig,
    obj_map: Dict[str, Dict],
    sampler,
    bootstrap_test,
    h0: bool,
    x_dep: bool,
    alpha: List[float],
) -> List[List[float]]:
    """Run one full simulation block and return a list of rows (one per objective)."""
    results = {name: np.zeros(len(alpha)) for name in obj_map}

    for _ in tqdm(
        range(cfg.n_resamples),
        desc=f"{'H0' if h0 else 'H1'}-{ 'x' if x_dep else 'const'}",
    ):
        # ----- sample synthetic data -----
        x, P, _pbar, y, _ = call_with_filtered_kwargs(
            sampler,
            vars(cfg),
            h0=h0,
            x_dep=x_dep,
            deg_h1=None if h0 else cfg.deg_h1_current,
        )

        # ----- run each requested objective -----
        for name, entry in obj_map.items():
            full = {
                "alpha": alpha,
                "x_inst": x,
                "p_preds": P,
                "y_labels": y,
                "params": entry["params"],
                "verbose": cfg.verbose,
                "use_val": True,
                "use_test": True,
            }
            sig = signature(bootstrap_test)
            kwargs = {k: v for k, v in full.items() if k in sig.parameters}
            decision, _pvals, _stat = bootstrap_test(**kwargs)
            dec_arr = np.array(decision, dtype=float)
            if h0:
                results[name] += dec_arr  # Type‑I error count
            else:
                results[name] += 1.0 - dec_arr  # Power (reject proportion)

    # Average over resamples and return row‑wise list (ordered by obj_map keys)
    return [(results[name] / cfg.n_resamples).tolist() for name in obj_map]


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------


def main(args: Any) -> None:
    cfg = ExperimentConfig.from_dict(vars(args))
    objectives = args.objectives or ["LP", "KL", "MMD", "SKCE"]
    obj_map = build_objective_map(cfg, objectives)

    sampler = (
        sample_dirichlet_experiment
        if cfg.experiment == "dirichlet"
        else sample_gp_experiment
    )
    bootstrap_test = BOOTSTRAP_TESTS[args.bootstrap_test]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # save directory: results_dir/experiment/bootstrap_test/timestamp
    save_dir = os.path.join(
        args.results_dir,  # e.g. "results"
        cfg.experiment,  # "gp"  or  "dirichlet"
        args.bootstrap_test,  # "v0", "v2", or "vanilla"
        timestamp,  # timestamp
    )
    os.makedirs(save_dir, exist_ok=True)

    # ---- H0 simulations ----
    alpha_h0 = [0.07 * i for i in range(1, 13)]
    cfg.verbose = args.verbose

    res_const = simulate(cfg, obj_map, sampler, bootstrap_test, True, False, alpha_h0)
    # save_results(res_const, save_dir, "h0_const.csv", list(obj_map.keys()))
    save_results_rowwise(res_const, save_dir, "h0_const.csv", list(obj_map), alpha_h0)

    res_xdep = simulate(cfg, obj_map, sampler, bootstrap_test, True, True, alpha_h0)
    # save_results(res_xdep, save_dir, "h0_xdep.csv", list(obj_map.keys()))
    save_results_rowwise(res_xdep, save_dir, "h0_xdep.csv", list(obj_map), alpha_h0)
    # ---- H1 simulations ----
    alpha_h1 = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    for deg in cfg.deg_h1_list:
        cfg.deg_h1_current = deg
        res_h1 = simulate(cfg, obj_map, sampler, bootstrap_test, False, True, alpha_h1)
        # save_results(res_h1, save_dir, f"h1_deg{deg}.csv", list(obj_map.keys()))
        save_results_rowwise(res_h1, save_dir, f"h1_{deg}.csv", list(obj_map), alpha_h1)
    # ---- Concatenate and plot ----
    csv_names = ["h0_const.csv", "h0_xdep.csv"] + [
        f"h1_{d}.csv" for d in cfg.deg_h1_list
    ]
    df_full = pd.concat(
        [pd.read_csv(os.path.join(save_dir, n)) for n in csv_names],
        ignore_index=True,
    )
    full_csv = os.path.join(save_dir, "full_results.csv")
    df_full.to_csv(full_csv, index=False)

    read_and_plot_error_analysis_full(
        file_path=full_csv,
        output_path=save_dir,
        list_col_titles=[
            r"$\lambda = const$",
            r"$\lambda = f(x)$",
        ]
        + [rf"$H_{{1,\,\delta={d}}}$" for d in cfg.deg_h1_list],
        title_1="Type I error",
        title_2="Power (1 − Type II)",
        alpha_1=np.asarray(alpha_h0),
        alpha_2=np.asarray(alpha_h1),
    )

    # ---- Persist config ----
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(make_serializable(vars(args)), f, indent=2)
    with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
        pickle.dump(vars(args), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic calibration-test run")

    # ------------ sampling parameters ------------
    parser.add_argument("--experiment", choices=["gp", "dirichlet"], default="gp")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--n_resamples", type=int, default=100)
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--n_members", type=int, default=10)
    parser.add_argument("--x_bound", type=float, nargs=2, default=[0.0, 5.0])
    parser.add_argument("--deg", type=int, default=2)
    parser.add_argument(
        "--bounds_p",
        type=float,
        nargs=2,
        action="append",
        default=[[0.5, 0.7], [0.6, 0.8]],
        help="Repeat `--bounds_p min max` for each GP predictor",
    )

    # ------------ training / optimiser -----------
    parser.add_argument("--optim", choices=["mlp", "COBYLA", "SLSQP"], default="mlp")
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=32)

    # ------------ loss / objective selector ------
    parser.add_argument("--cal_weight", type=float, default=0.0)
    parser.add_argument("--p_param", type=int, default=2)

    # ------------ scenario sweep -----------------
    parser.add_argument(
        "--deg_h1_list", type=float, nargs="*", default=[0.02, 0.1, 0.15]
    )

    # ------------ meta ---------------------------
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--verbose", action="store_true")

    # ------------ bootstrap & objectives ---------
    parser.add_argument("--bootstrap_test", choices=["v0", "v2"], default="v0")
    parser.add_argument(
        "--objectives",
        nargs="*",
        choices=["LP", "KL", "MMD", "SKCE"],
        help="Subset of objectives to evaluate (default: all four)",
    )

    args = parser.parse_args()
    main(args)
