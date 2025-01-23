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

    # read train_params from config
    train_params = config["train_params"]
    n_repeats = train_params.get("n_repeats", 10)
    n_epochs = train_params.get("n_epochs", 200)
    lr = train_params.get("lr", 1e-3)
    batch_size = train_params.get("batch_size", 64)

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

    for repeat_idx in range(n_repeats):
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
                        hidden_dim=train_params['hidden_dim'],
                        hidden_layers=train_params['hidden_layers'],
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
                        patience=40,
                        verbose=True,
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
        metric_gt =measure_calibration_metrics(
            p_true_val, y_val, dict_params=config
        )
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
    csv_path = os.path.join(output_dir, f"calibration_scores_{exp_name}_{n_repeats}.csv")
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


# def run_experiment(
#     dataset_train,
#     dataset_val,
#     x_val: torch.tensor,  # or you can feed DataLoader, etc.
#     p_pred_val: torch.tensor,  # shape (N, K, 2) or something, for real data or synthetic
#     y_val: torch.tensor,  # shape (N,)
#     losses: list = [
#         GeneralizedBrierLoss(),
#         GeneralizedLogLoss(),
#     ],  # e.g. [GeneralizedBrierLoss(), GeneralizedLogLoss()]
#     train_modes: list = ["joint", "alternating", "avg_then_calibrate"],
#     comb_model_class=MLPCalW,  # e.g. ["joint", "alternating", "avg_then_calibrate"]
#     calibrator_classes: list = [
#         DirichletCalibrator,
#         TemperatureScalingCalibrator,
#     ],  # e.g. [DirichletCalibrator, TemperatureScalingCalibrator, ...]
#     n_repeats: int = 20,
#     n_epochs: int = 100,
#     lr: float = 1e-3,
#     batch_size: int = 64,
#     hidden_dim: int = 64,
#     hidden_layers: int = 2,
#     output_dir="./results",
#     save_plots: bool = False,
#     config_dict: dict = {},
# ):
#     """
#     A generalized pipeline that:
#       1) Repeats training n_repeats times,
#       2) For each repeat, trains all combos of (loss, train_mode, calibrator),
#       3) Evaluates calibration metrics on val set,
#       4) Saves mean/std of scores to CSV in output_dir,
#       5) Optionally saves plots to output_dir.

#     dataset_train : Dataset
#     dataset_val   : Dataset
#       - Provide the needed data for training/validation.
#     x_val, p_pred_val, y_val
#       - For the final forward pass on the validation to compute p_cal or do averaging.
#     losses : list
#       - e.g. [GeneralizedBrierLoss(), GeneralizedLogLoss()]
#     train_modes : list
#       - e.g. ["joint", "alternating", "avg_then_calibrate"]
#     comb_model_class : nn.Module
#         the ensemble combination model class, e.g. MLPCalW or MLPCalConv
#     calibrator_classes : list
#       - e.g. [DirichletCalibrator, TemperatureScalingCalibrator, LinearCalibrator]
#     n_repeats : int
#       - how many times to re-run everything (to measure avg/stdev)
#     output_dir : str
#       - directory to save csv and (optionally) plots
#     save_plots : bool
#       - whether to plot and save them
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # We'll store results in a nested structure:
#     # results[ (loss_name, train_mode, cal_model_name) ] = list of metric dicts over repeats
#     # each metric dict has e.g. {"brier":..., "mmd":..., "skce":..., "ece_kde":...}
#     results = {}
#     n_ens, n_classes = p_pred_val.shape[1], p_pred_val.shape[2]
#     in_channels = x_val.shape[1]
#     exp_name = config_dict["experiment_name"]
#     # use tqdm to show progress
#     for repeat_idx in tqdm(range(n_repeats)):
#         print(f"=== Repeat {repeat_idx+1}/{n_repeats} ===")

#         for loss_obj in losses:
#             loss_name = loss_obj.__class__.__name__
#             for train_mode in train_modes:
#                 for CalClass in calibrator_classes:
#                     cal_name = CalClass.__name__

#                     print(
#                         f"[Repeat={repeat_idx+1}] Training {loss_name}, {train_mode}, {cal_name}"
#                     )

#                     # 1) Construct the model
#                     model = CredalSetCalibrator(
#                         comb_model=comb_model_class,
#                         cal_model=CalClass,
#                         in_channels=in_channels,
#                         n_classes=n_classes,
#                         n_ensembles=n_ens,
#                         hidden_dim=hidden_dim,
#                         hidden_layers=hidden_layers,
#                     )

#                     # 2) Train
#                     model, loss_train, loss_val = train_model(
#                         model=model,
#                         dataset_train=dataset_train,
#                         dataset_val=dataset_val,
#                         loss_fn=loss_obj,
#                         train_mode=train_mode,
#                         n_epochs=n_epochs,
#                         lr=lr,
#                         batch_size=batch_size,
#                         early_stopping=True,
#                         patience=50,
#                         verbose=False,
#                     )

#                     # 3) Evaluate on validation:
#                     # either model(x_val, p_pred_val) or average + cal_model
#                     with torch.no_grad():
#                         if train_mode in ["joint", "alternating"]:
#                             p_cal_val, p_bar_val, weights_val = model(x_val, p_pred_val)
#                         else:
#                             # avg_then_calibrate
#                             p_bar_val = p_pred_val.mean(dim=1)
#                             p_cal_val = model.cal_model(p_bar_val)

#                     # 4) compute metrics
#                     metric_dict = measure_calibration_metrics(
#                         p_cal_val, y_val, dict_params=config_dict
#                     )
#                     # measure average accuracy
#                     # e.g. predict class 1 if p_cal_np[:,0]>0.5
#                     preds = (p_cal_val[:, 1] > 0.5).int().cpu().numpy()
#                     acc = np.mean(preds == y_val.cpu().numpy())
#                     print(acc)
#                     metric_dict["accuracy"] = float(acc)

#                     # store in results dict
#                     key = (loss_name, train_mode, cal_name)
#                     if key not in results:
#                         results[key] = []
#                     results[key].append(metric_dict)

#                     # 5) Optionally plot
#                     if save_plots:
#                         out_plot_name = f"credal_set_log_{loss_name}_{train_mode}_{cal_name}_rpt{repeat_idx+1}.png"
#                         full_path = os.path.join(output_dir, out_plot_name)
#                         if train_mode in ["joint", "alternating"]:
#                             plot_ens_comb_cal(
#                                 x_inst=x_val,
#                                 p_true=...,  # if you have the true prob
#                                 ens_preds=p_pred_val,
#                                 model=model,
#                                 file_name=out_plot_name,
#                                 output_path=output_dir,
#                                 alpha_comb=1.0,
#                                 output_pbar="weighted",
#                             )
#                         else:
#                             # avg_then_calibrate
#                             plot_ens_comb_cal(
#                                 x_inst=x_val,
#                                 p_true=...,
#                                 ens_preds=p_pred_val,
#                                 model=model,
#                                 file_name=out_plot_name,
#                                 output_path=output_dir,
#                                 alpha_comb=1.0,
#                                 output_pbar="average",
#                             )

#     # now results dict is populated for each repeat
#     # e.g. results[ (loss,mode,cal) ] = [ { "brier":0.1, "mmd":..., }, { "brier":0.11,...}, ...]

#     # 6) compute mean/std over repeats
#     # let's say measure_calibration_metrics returns keys: brier, skce, ece_kde, mmd
#     # we'll create a final dict: final_scores[(loss,mode,cal)] = { brier_mean, brier_std, ... }
#     final_scores = {}
#     for (loss_name, tm, cal_name), list_of_dicts in results.items():
#         # list_of_dicts is a list of metrics per repeat
#         # we can gather each metric into a list
#         if not list_of_dicts:
#             continue

#         # gather the metrics
#         metric_keys = list(
#             list_of_dicts[0].keys()
#         )  # e.g. ["brier","mmd","skce","ece_kde"]
#         agg = {}
#         for mk in metric_keys:
#             vals = [d[mk] for d in list_of_dicts]
#             mean_val = np.mean(vals)
#             std_val = np.std(vals)
#             agg[f"{mk}_mean"] = float(mean_val)
#             agg[f"{mk}_std"] = float(std_val)
#         final_scores[(loss_name, tm, cal_name)] = agg

#     # 7) Now we can save final_scores to CSV for each metric if you prefer pivot
#     # Or we can just do one CSV with columns [loss,train_mode,cal_model,brier_mean,brier_std, ...]
#     # We'll do a single CSV:
#     csv_path = os.path.join(
#         output_dir, f"calibration_scores_{n_repeats}_{exp_name}.csv"
#     )
#     columns = ["loss", "train_mode", "cal_model"]  # plus each metric_mean / metric_std
#     # figure out all metric fields
#     # e.g. from final_scores, gather all keys
#     # but we know it might be brier_mean, brier_std, mmd_mean, mmd_std, ...
#     metric_fields = set()
#     for agg in final_scores.values():
#         metric_fields.update(agg.keys())
#     metric_fields = sorted(
#         metric_fields
#     )  # e.g. ["brier_mean","brier_std","ece_kde_mean","ece_kde_std", ...]

#     # full columns = base + metric_fields
#     columns += metric_fields

#     with open(csv_path, "w", newline="") as f:
#         import csv

#         writer = csv.writer(f)
#         writer.writerow(columns)

#         for (loss_name, tm, cal_name), metric_agg in final_scores.items():
#             row = [loss_name, tm, cal_name] + [
#                 metric_agg.get(mf, "") for mf in metric_fields
#             ]
#             writer.writerow(row)

#     print(f"Saved final calibration scores to {csv_path}")


def measure_calibration_metrics(p_cal, y, dict_params: dict = {}):
    """
    Returns a dict of calibration metrics, e.g. {"brier":..., "mmd":..., "skce":..., "ece_kde":...}
    Here you can do the absolute-value trick or any param passing.
    """
    # e.g.
    brier_val = brier_obj(p_cal, y)
    mmd_val = mmd_kce(p_cal, y, bw=dict_params["dict_mmd"]["bw"])
    skce_val = get_skce_ul(p_cal, y, bw=dict_params["dict_skce"]["bw"])
    ece_val = get_ece_kde(
        p_cal,
        y,
        p=dict_params["dict_kde_ece"]["p"],
        bw=dict_params["dict_kde_ece"]["bw"],
    )

    # store them as absolute floats if you prefer
    d = {}
    d["brier"] = float(abs(brier_val))
    d["mmd"] = float(abs(mmd_val))
    d["skce"] = float(abs(skce_val))
    d["ece_kde"] = float(abs(ece_val))
    return d


def main(config: dict):

    config = load_config(config)
    # output_dir = config["output_dir"]
    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

    # # 2) Create dataset
    # dataset_cfg = config["dataset"]
    # dataset_method = dataset_cfg["method"]
    # # see if dataset_method is "gp" or "logistic", then call create_synthetic_dataset
    # if dataset_method in ["gp", "logistic"]:
    #     train_set, val_set, test_set = create_synthetic_dataset(dataset_cfg)
    #     p_pred_val = val_set.dataset.p_probs[val_set.indices]
    #     p_true_val = val_set.dataset.p_true[val_set.indices]
    #     y_val = val_set.dataset.y_true[val_set.indices]
    #     x_val = val_set.dataset.x_train[val_set.indices]
    # else:
    #     raise NotImplementedError(
    #         f"Dataset method {dataset_method} not implemented yet."
    #     )

    # run_experiment(
    #     dataset_train=train_set,
    #     dataset_val=val_set,
    #     x_val=x_val,
    #     p_pred_val=p_pred_val,
    #     y_val=y_val,
    #     n_repeats=20,
    #     n_epochs=config["train_params"]["n_epochs"],
    #     lr=config["train_params"]["lr"],
    #     batch_size=config["train_params"]["batch_size"],
    #     hidden_dim=config["train_params"]["hidden_dim"],
    #     hidden_layers=config["train_params"]["hidden_layers"],
    #     output_dir=output_dir,
    #     save_plots=False,
    #     config_dict=config,
    # )
    run_experiment_full(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    main(args.config)
