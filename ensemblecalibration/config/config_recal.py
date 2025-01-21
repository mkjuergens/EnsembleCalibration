import os
import yaml
from torch import optim

from ensemblecalibration.losses.proper_losses import *


def create_config(
    experiment_name="gp_experiment",
    dataset_method="gp",  # or "logistic", "cifar10", "cifar100"
    n_samples=5000,
    n_ens=5,
    scale_noise=0.5,
    kernel_width=0.1,
    # If real data, you might store 'ensemble_preds_file' or other paths here
    ensemble_preds_file=None,
    # Training params
    n_epochs=100,
    lr=1e-3,
    batch_size=128,
    patience=10,
    # Which calibration methods to run
    opt_methods=("joint", "alternating", "avg_then_calibrate"),
    # Metrics to evaluate
    eval_metrics=("brier", "l2", "mmd", "skce"),
    loss_function="brier",  # or "log_loss", etc.
    calibration_map="temperature_scaling",
    calibrator_params=None,
    loss_params=None,
    # Output directory
    output_dir=None,
    # Any additional special arguments
    **kwargs,
):
    """
    Creates a configuration dictionary for a large-scale calibration experiment.
    You can use this for either synthetic (GP/logistic) or real data (CIFAR, etc.).

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (used in output_dir).
    dataset_method : str
        Either 'gp', 'logistic', 'cifar10', 'cifar100', etc.
    n_samples : int
        Number of samples if synthetic. Not used if real data.
    n_ens : int
        Number of ensemble members.
    scale_noise : float
        For synthetic data (GP/logistic), sets amplitude of noise.
    kernel_width : float
        For GP-based data or some kernel-based methods, sets RBF length-scale.
    ensemble_preds_file : str, optional
        Path to the precomputed ensemble predictions for real data.
    n_epochs : int
        Number of training epochs for calibration.
    lr : float
        Learning rate.
    batch_size : int
        Training batch size.
    patience : int
        Patience for early stopping, etc.
    opt_methods : tuple of str
        Which calibration optimization methods to use: 'joint', 'alternating', 'avg_then_calibrate'.
    eval_metrics : tuple of str
        Which metrics to compute in the evaluation step.
    output_dir : str or None
        If None, defaults to "./outputs/<experiment_name>".
    kwargs : dict
        Additional parameters you might want to store.

    Returns
    -------
    dict
        A configuration dictionary that can be written to a .yml file
        or directly passed to your run_experiment function.
    """

    if output_dir is None:
        output_dir = f"./outputs/{experiment_name}"

    # Build the config dictionary
    config = {
        "experiment_name": experiment_name,
        "dataset": {
            "method": dataset_method,
            "n_samples": n_samples,
            "n_ens": n_ens,
            "scale_noise": scale_noise,
            "kernel_width": kernel_width,
            "ensemble_preds_file": ensemble_preds_file,  # for real data
        },
        "train_params": {
            "n_epochs": n_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "patience": patience,
        },
        "optimization_methods": list(opt_methods),
        "loss_function": loss_function,
        "calibration_map": calibration_map,
        "calibrator_params": calibrator_params or {},
        "loss_params": loss_params or {},
        "eval_metrics": list(eval_metrics),
        "output_dir": output_dir,
    }

    # Merge additional kwargs if you have special fields
    for k, v in kwargs.items():
        config[k] = v

    return config


def save_config_as_yaml(config_dict, save_path):
    """
    Utility to write the config dictionary to a .yml file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)
    print(f"Saved config to {save_path}")


if __name__ == "__main__":
    # Creating a GP experiment config
    gp_cfg = create_config(
        experiment_name="gp_experiment",
        dataset_method="gp",
        n_samples=3000,
        scale_noise=0.4,
        kernel_width=0.05,
        n_epochs=150,
        lr=1e-3,
        batch_size=64,

        # We'll run 3 methods: joint, alt, and avg_cal
        methods_to_run=("joint", "alternating", "avg_then_calibrate"),

        # We'll measure brier, mmd, etc.
        eval_metrics=("brier","mmd_calib"),

        # We'll choose "brier" as the loss, and "temperature_scaling" as calibrator
        loss_function="brier",
        calibration_map="temperature_scaling",

        # Example: passing calibrator and loss parameters
        calibrator_params={"init_temp": 2.0},
        loss_params={"epsilon": 1e-8},

        # Additional fields if needed
        output_dir="./outputs/gp_experiment"
    )

    save_config_as_yaml(gp_cfg, "./configs/gp_experiment.yml")

    # Creating a real-data CIFAR config
    cifar_cfg = create_config(
        experiment_name="cifar10_ensemble",
        dataset_method="cifar10",
        ensemble_preds_file="./data/cifar10_ensembles.pt",
        n_epochs=50,
        lr=5e-4,
        batch_size=128,
        methods_to_run=("joint","avg_then_calibrate"),
        eval_metrics=("brier","l1_calib","mmd_calib"),
        loss_function="log_loss",
        calibration_map="dirichlet",
        calibrator_params={"n_classes": 10},  # if your Dirichlet calibrator needs that
        output_dir="./outputs/cifar10_experiment"
    )

    save_config_as_yaml(cifar_cfg, "./configs/cifar10_experiment.yml")