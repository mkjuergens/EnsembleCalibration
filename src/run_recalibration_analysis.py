import os
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import csv

from ensemblecalibration.config import load_config
from ensemblecalibration.data.synthetic import create_synthetic_dataset
from ensemblecalibration.meta_model import (
    CredalSetCalibrator,
    MLPCalW,
    DirichletCalibrator,
    TemperatureScalingCalibrator,
)
from ensemblecalibration.losses.proper_losses import (
    GeneralizedBrierLoss,
    GeneralizedLogLoss,
)
from ensemblecalibration.meta_model.train import train_model
from ensemblecalibration.utils.plot_functions import plot_ens_comb_cal
from ensemblecalibration.cal_estimates import (
    brier_obj,
    mmd_kce,
    get_skce_ul,
    get_ece_kde,
    kl_kde_obj,
)


def run_experiment_full(
    config: dict,
    losses: list = [GeneralizedBrierLoss(), GeneralizedLogLoss()],
    train_modes: list = ["joint", "alternating", "avg_then_calibrate"],
    calibrator_classes: list = [DirichletCalibrator, TemperatureScalingCalibrator],
):
    """
    "Full approach": in each repeat, we randomly re-generate a synthetic dataset,
    train the calibration strategies, measure calibration errors, and store the results.
    After all repeats, we average the results and save them to a CSV.

    Parameters
    ----------
    config : dict
        A dictionary that includes dataset configuration, training parameters,
        output directory, etc.

    Example fields in config:
        {
          "output_dir": "./results",
          "dataset": {
             "method": "gp",  # or "logistic"
             "n_samples": 2000,
             ...
          },
          "train_params": {
             "n_repeats": 10,
             "n_epochs": 100,
             "lr": 1e-3,
             "batch_size": 64,
             ...
          },
          ...
        }
    """
    output_dir = config.get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)
    verbose = config.get("verbose", True)

    # read train_params from config
    train_params = config["train_params"]
    n_repeats = train_params.get("n_repeats", 20)
    n_epochs = train_params.get("n_epochs", 2000)
    lr = train_params.get("lr", 1e-4)
    batch_size = train_params.get("batch_size", 64)
    n_samples = config["dataset"]["n_samples"]

    # Suppose you have a list of calibration approaches you want to try:
    # e.g. losses = [GeneralizedBrierLoss(), GeneralizedLogLoss()]
    # train_modes = ["joint","alternating","avg_then_calibrate"]
    # calibrator_classes = [DirichletCalibrator, TemperatureScalingCalibrator, ...]

    # losses = config["losses"]
    # train_modes = config["train_modes"]
    # calibrator_classes = config["calibrator_classes"]

    # We'll store results in a dict:
    # results[(loss_name, train_mode, cal_name)] = [metric_dict_1, metric_dict_2, ...]
    #   one per repeated dataset
    results = {}
    # save scores for ground truth
    results_gt = {}
    exp_name = config["experiment_name"]

    # use tqdm to iterate over repeats
    for repeat_idx in tqdm(range(n_repeats)):
        print(f"=== Generating dataset for repeat {repeat_idx+1}/{n_repeats} ===")

        # 1) Re-generate a new synthetic dataset
        dataset_cfg = config["dataset"]
        if dataset_cfg["method"] in ["gp", "logistic"]:
            train_set, val_set, test_set = create_synthetic_dataset(dataset_cfg)
        else:
            raise NotImplementedError(
                f"Dataset method {dataset_cfg['method']} not implemented yet."
            )

        # extract needed arrays from val_set
        p_pred_val = val_set.dataset.p_probs[val_set.indices]  # shape (N,K,2)
        p_true_val = val_set.dataset.p_true[val_set.indices]  # shape (N,2)
        y_val = val_set.dataset.y_true[val_set.indices]  # shape (N,)
        x_val = val_set.dataset.x_train[val_set.indices]  # shape (N,1)

        # shapes
        n_ens, n_classes = p_pred_val.shape[1], p_pred_val.shape[2]
        in_channels = x_val.shape[1]

        # 2) For each combination of (loss, train_mode, calibrator), train + measure
        for loss_obj in losses:
            loss_name = loss_obj.__class__.__name__
            for train_mode in train_modes:
                for CalClass in calibrator_classes:
                    cal_name = CalClass.__name__
                    print(
                        f"  [Repeat={repeat_idx+1}] {loss_name} / {train_mode} / {cal_name}"
                    )

                    # Build your model (e.g. CredalSetCalibrator, etc.)
                    model = CredalSetCalibrator(
                        comb_model=MLPCalW,
                        cal_model=CalClass,
                        in_channels=in_channels,
                        n_classes=n_classes,
                        n_ensembles=n_ens,
                        hidden_dim=train_params["hidden_dim"],
                        hidden_layers=train_params["hidden_layers"],
                    )

                    # 3) Train model
                    # (You likely have a train_model function.)
                    model, loss_train, loss_val = train_model(
                        model=model,
                        dataset_train=train_set,
                        dataset_val=val_set,
                        loss_fn=loss_obj,
                        train_mode=train_mode,
                        n_epochs=n_epochs,
                        lr=lr,
                        batch_size=batch_size,
                        early_stopping=True,
                        patience=train_params["patience"],
                        verbose=verbose,
                        subepochs_cal=train_params["subepochs_cal"],
                        subepochs_comb=train_params["subepochs_comb"],
                    )

                    # 4) Evaluate on val
                    #    a) get p_cal
                    with torch.no_grad():
                        if train_mode in ["joint", "alternating"]:
                            p_cal_val, p_bar_val, weights_val = model(x_val, p_pred_val)
                        else:
                            # "avg_then_calibrate"
                            p_bar_val = p_pred_val.mean(dim=1)
                            p_cal_val = model.cal_model(p_bar_val)

                    # b) measure your calibration metrics
                    # p_cal_np = p_cal_val.detach().cpu().numpy()  # shape (N,2)
                    # y_val_np = y_val # shape (N,)
                    # p_true_np = p_true_val.cpu().numpy()  # shape (N,2)

                    metric_dict = measure_calibration_metrics(
                        p_cal_val, y_val, dict_params=config
                    )

                    # accuracy
                    preds = (p_cal_val[:, 1] > 0.5).int().cpu().numpy()
                    acc = np.mean(preds == y_val.cpu().numpy())
                    metric_dict["accuracy"] = float(acc)

                    # store in results
                    key = (loss_name, train_mode, cal_name)
                    if key not in results:
                        results[key] = []
                    results[key].append(metric_dict)

        # 5) Evaluate on ground truth
        metric_gt = measure_calibration_metrics(p_true_val, y_val, dict_params=config)
        # accuracy
        preds = (p_true_val[:, 1] > 0.5).int().cpu().numpy()
        acc = np.mean(preds == y_val.cpu().numpy())
        metric_gt["accuracy"] = float(acc)

        key = ("ground_truth", "ground_truth", "ground_truth")
        if key not in results_gt:
            results_gt[key] = []
        results_gt[key].append(metric_gt)

    # 5) Now we average over n_repeats
    final_scores = {}
    for (loss_name, tm, cal_name), list_of_dicts in results.items():
        if not list_of_dicts:
            continue
        # gather keys from first dict
        metric_keys = list(list_of_dicts[0].keys())
        agg = {}
        for mk in metric_keys:
            vals = [d[mk] for d in list_of_dicts]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            agg[f"{mk}_mean"] = float(mean_val)
            agg[f"{mk}_std"] = float(std_val)
        final_scores[(loss_name, tm, cal_name)] = agg

    # append the ground truth results
    for (loss_name, tm, cal_name), list_of_dicts in results_gt.items():
        if not list_of_dicts:
            continue
        # gather keys from first dict
        metric_keys = list(list_of_dicts[0].keys())
        agg = {}
        for mk in metric_keys:
            vals = [d[mk] for d in list_of_dicts]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            agg[f"{mk}_mean"] = float(mean_val)
            agg[f"{mk}_std"] = float(std_val)
        final_scores[(loss_name, tm, cal_name)] = agg

    # 6) Write final CSV
    csv_path = os.path.join(
        output_dir, f"calibration_scores_{exp_name}_{n_repeats}_{n_samples}.csv"
    )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # figure out all metric fields
    metric_fields = set()
    for agg in final_scores.values():
        metric_fields.update(agg.keys())
    metric_fields = sorted(
        metric_fields
    )  # e.g. brier_mean, brier_std, ece_mean, ece_std, etc.

    columns = ["loss", "train_mode", "cal_model"] + list(metric_fields)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for (loss_name, tm, cal_name), agg_dict in final_scores.items():
            row = [loss_name, tm, cal_name] + [
                agg_dict.get(mf, "") for mf in metric_fields
            ]
            writer.writerow(row)

    print(f"Saved final calibration scores to {csv_path}")


def measure_calibration_metrics(p_cal, y, dict_params: dict = {}):
    """
    Returns a dict of calibration metrics, e.g. {"brier":..., "mmd":..., "skce":..., "ece_kde":...}
    Here you can do the absolute-value trick or any param passing.
    """
    # e.g.
    brier_val = brier_obj(p_cal, y)
    mmd_val = mmd_kce(p_cal, y, bw=dict_params["dict_mmd"]["bw"], take_square=False)
    skce_val = get_skce_ul(
        p_cal, y, bw=dict_params["dict_skce"]["bw"], take_square=False
    )
    ece_val = get_ece_kde(
        p_cal,
        y,
        p=dict_params["dict_kde_ece"]["p"],
        bw=dict_params["dict_kde_ece"]["bw"],
    )
    kl_val = kl_kde_obj(p_bar=p_cal, y=y, params=dict_params["dict_kl"])

    # store them as absolute floats if you prefer
    d = {}
    d["brier"] = float(abs(brier_val))
    d["mmd"] = float(abs(mmd_val))
    d["skce"] = float(abs(skce_val))
    d["ece_kde"] = float(abs(ece_val))
    d["kl_kde"] = float(abs(kl_val))
    return d


def main(config: dict):

    config = load_config(config)
    run_experiment_full(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    main(args.config)
