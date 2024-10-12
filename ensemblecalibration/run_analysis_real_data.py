import torch
import os
import argparse

from ensemblecalibration.meta_model.mlp_model import MLPCalWConv, MLPCalWithPretrainedModel
from ensemblecalibration.cal_test import npbe_test_vaicenavicius
from ensemblecalibration.meta_model.train import get_optim_lambda_mlp
from ensemblecalibration.data.ensemble.dataset_utils import load_results
from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.utils.helpers import test_train_val_split
from ensemblecalibration.config.config_cal_test import create_config_test
from ensemblecalibration.meta_model.losses import LpLoss, SKCELoss, BrierLoss, MMDLoss
from ensemblecalibration.utils.helpers import calculate_pbar


def main(args):
    LIST_ERRORS = ["LP", "Brier", "MMD", "SKCE"]
    LIST_MODELS = ["vgg", "resnet"]
    LIST_N_ENS = [5, 10]

    config = create_config_test(cal_test=npbe_test_vaicenavicius, n_resamples=100,
                                n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                patience=args.patience,
                                hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim,
                                device=args.device,)
    # Load predictions on test set, instance features and labels
    p_preds, x_inst, y_labels = load_results(
        dataset_name=args.dataset,
        model_type=args.model_type,        # Pass the model type (resnet, vgg)
        ensemble_type=args.ensemble_type,  # Pass the ensemble type (deep_ensemble, mc_dropout)
        ensemble_size=args.ensemble_size if args.ensemble_type == "deep_ensemble" else None,  # Ensemble size only for deep ensemble
        directory=args.results_dir
    )
    x_inst = torch.from_numpy(x_inst)
    y_labels = torch.from_numpy(y_labels)
    assert p_preds.shape[0] == x_inst.shape[0], "Data mismatch"


    #config = create_config(cal_test=npbe_test_vaicenavicius, optim)
    # # Initialize model
    model = MLPCalWithPretrainedModel(
        out_channels=p_preds.shape[1],
        hidden_dim=128,
        hidden_layers=1,
        pretrained_model=args.model_type
    )

    # split data into train, validation and test (train and val are used to train the MLP)
    data_test, data_train, data_val = test_train_val_split(p_preds, y_labels, x_inst)

    dataset_train = MLPDataset(
        x_train=data_train[0], P=data_train[2], y=data_train[1]
    )
    dataset_val = MLPDataset(x_train=data_val[0], P=data_val[2], y=data_val[1])
    dataset_test = MLPDataset(x_train=data_test[0], P=data_test[2], y=data_test[1])


    # # Train model
    optim_l, loss_train, loss_val = get_optim_lambda_mlp(
                         dataset_train=dataset_train,
                         dataset_val=dataset_val,
                         dataset_test=dataset_test,
                         model=model,
                         loss=MMDLoss(bw=0.01),
                         n_epochs=args.epochs,
                         lr=args.lr,
                         batch_size=args.batch_size,
                         patience=args.patience,
                         device=args.device,
                         verbose=args.verbose)
    print(f"optim weights: {optim_l}")
    # run test
    alpha=0.05
    p_bar = calculate_pbar(weights_l=optim_l, p_preds=data_test[2])
    y_labels_test = data_test[1]
    print(y_labels_test.shape)
    print(p_bar.shape)
    decision, p_val, stat = npbe_test_vaicenavicius(alpha=alpha,
                                                    p_probs=p_bar,
                                                   y_labels=y_labels_test,
                                                    params=config["MMD"]["params"]
                                                      )

    
    print(f"decision: {decision}")
    print(f"p_val: {p_val}")
    print(f"Stat: {stat}")
    # compare to mean prediction
    mean_preds = data_test[2].mean(axis=1)
    decision, p_val, stat = npbe_test_vaicenavicius(alpha=alpha,
                                                    p_probs=mean_preds,
                                                   y_labels=y_labels_test,
                                                    params=config["LP"]["params"]
                                                      )
    print(f"decision: {decision}")
    print(f"p_val: {p_val}")
    print(f"Stat: {stat}")

if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train MLP calibration model")
        
        parser.add_argument(
            "--results_dir",
            type=str,
            default="./data/ensemble/ensemble_results",
            help="Directory to load results from"
        )
        
        parser.add_argument(
            "--dataset", 
            type=str, 
            default="CIFAR10", 
            help="Dataset to train on"
        )
        
        parser.add_argument(
            "--ensemble_size", 
            type=int, 
            default=5, 
            help="Number of models in the ensemble (only applicable for deep ensembles)"
        )
        
        parser.add_argument(
            "--model_type", 
            type=str, 
            default="resnet", 
            choices=["resnet", "vgg"], 
            help="Model type: resnet or vgg"
        )
        
        parser.add_argument(
            "--ensemble_type", 
            type=str, 
            default="deep_ensemble", 
            choices=["deep_ensemble", "mc_dropout"], 
            help="Ensemble type: deep ensemble or MCDropout"
        )
        
        parser.add_argument(
            "--hidden_dim", 
            type=int, 
            default=100, 
            help="Hidden dimension of the ConvNet"
        )
        
        parser.add_argument(
            "--hidden_layers",
            type=int,
            default=1,
            help="Number of hidden layers in the ConvNet"
        )
        
        parser.add_argument(
            "--epochs", 
            type=int, 
            default=200, 
            help="Number of epochs to train for"
        )
        
        parser.add_argument(
            "--patience", 
            type=int, 
            default=100, 
            help="Number of epochs to wait before early stopping"
        )
        
        parser.add_argument(
            "--batch_size", 
            type=int, 
            default=1024, 
            help="Batch size for training"
        )
        
        parser.add_argument(
            "--val_split",
            type=float,
            default=0.1,
            help="Fraction of training data to use for validation"
        )

        parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="leanring rate for training the convmlp"
        )
        
        parser.add_argument(
            "--device", 
            type=str, 
            default="cuda:0", 
            help="Device to train on"
        )
        
        parser.add_argument(
            "--verbose", 
            type=bool, 
            default=True, 
            help="Print results and loss"
        )
        
        args = parser.parse_args()
        main(args)