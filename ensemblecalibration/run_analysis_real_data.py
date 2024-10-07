import torch
import os
import argparse

from ensemblecalibration.meta_model.mlp_model import MLPCalWConv
from ensemblecalibration.cal_test import npbe_test_vaicenavicius
from ensemblecalibration.meta_model.train import get_optim_lambda_mlp
from ensemblecalibration.data.ensemble.datasets import load_results
from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.utils.helpers import test_train_val_split
from ensemblecalibration.config.config_cal_test import create_config


def main():
    parser = argparse.ArgumentParser(description="Train MLP calibration model")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./data/ensemble/results",
        help="Directory to load results from",
    )
    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="Dataset to train on"
    )
    parser.add_argument(
        "--ensemble_size", type=int, default=5, help="Number of models in the ensemble"
    )

    parser.add_argument(
        "--model_idx", type=int, default=1, help="Index of the model to train"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=100, help="Hidden dimension of the ConvNet"
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers in the ConvNet",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs to wait before early stopping",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of training data to use for validation",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    # parser.add_argument("--optim", type=str, default="mlp", help="Optimization method")
    parser.add_argument(
        "--verbose", type=bool, default=True, help="Print results and loss"
    )
    args = parser.parse_args()

    # Load predictions on test set, instance features and labels
    p_preds, x_inst, y_labels = load_results(
        dataset_name=args.dataset,
        ensemble_size=args.ensemble_size,
        directory=args.results_dir,
    )

    assert p_preds.shape[0] == x_inst.shape[0] == y_labels.shape[0], "Data mismatch"

    #config = create_config(cal_test=npbe_test_vaicenavicius, optim)
    # Initialize model
    model = MLPCalWConv(
        in_channels=x_inst.shape[1],
        out_channels=p_preds.shape[2],
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        use_relu=True,
    ).to(args.device)

    # split data into train, validation and test (train and val are used to train the MLP)
    data_test, data_train, data_val = test_train_val_split(p_preds, y_labels, x_inst)

    dataset_train = MLPDataset(
        x_train=data_train[0], P=data_train[2], y=data_train[1]
    )
    dataset_val = MLPDataset(x_train=data_val[0], P=data_val[2], y=data_val[1])
    dataset_test = MLPDataset(x_train=data_test[0], P=data_test[2], y=data_test[1])


    # Train model
    optim_l, loss_train, loss_val = get_optim_lambda_mlp(dataset_train=dataset_train,
                         dataset_val=dataset_val,
                         dataset_test=dataset_test,
                         model=model,
                         loss="nll",
                         n_epochs=args.epochs,
                         lr=0.01,
                         batch_size=args.batch_size,
                         patience=args.patience,
                         device=args.device,
                         verbose=args.verbose)
    # Test model
    npbe_test_vaicenavicius(
        alpha=[0.05],
        p_probs=dataset_test
    )
