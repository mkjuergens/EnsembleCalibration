import torch
import os
import argparse
import pandas as pd

from ensemblecalibration.meta_model.mlp_model import MLPCalWithPretrainedModel
from ensemblecalibration.cal_test import npbe_test_vaicenavicius
from ensemblecalibration.meta_model.train import get_optim_lambda_mlp
from ensemblecalibration.data.ensemble.dataset_utils import load_results
from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.utils.helpers import test_train_val_split
from ensemblecalibration.config.config_cal_test import create_config_test
from ensemblecalibration.utils.helpers import calculate_pbar


def main(args):
    LIST_ERRORS = ["LP", "Brier", "MMD", "SKCE"]
    LIST_MODELS = ["vgg", "resnet"]
    LIST_N_ENS = [5, 10]

    # save results in a dictionary
    results_list = []
    for n_ens in LIST_N_ENS:
        for model_type in LIST_MODELS:
            for error in LIST_ERRORS:
                print(f"Running calibration analysis for {error} and {model_type} with {n_ens} ensemble members")
                config = create_config_test(cal_test=npbe_test_vaicenavicius, n_resamples=100,
                                            n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                            patience=args.patience,
                                            hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim,
                                            device=args.device,)
                # Load predictions on test set, instance features and labels
                p_preds, x_inst, y_labels = load_results(
                    dataset_name=args.dataset,
                    model_type=model_type,        # Pass the model type (resnet, vgg)
                    ensemble_type="deep_ensemble",  # Pass the ensemble type (deep_ensemble, mc_dropout)
                    ensemble_size=n_ens,  # Ensemble size only for deep ensemble
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
                    pretrained_model=model_type
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
                                    loss=config[error]["params"]["loss"],
                                    n_epochs=args.epochs,
                                    lr=args.lr,
                                    batch_size=args.batch_size,
                                    patience=args.patience,
                                    device=args.device,
                                    verbose=args.verbose)
                
                # run test, first with lambda that was found by optimization
                alpha = args.alpha
                p_bar = calculate_pbar(weights_l=optim_l, p_preds=data_test[2])
                y_labels_test = data_test[1]

                decision_lambda, p_val_lambda, stat_lambda = npbe_test_vaicenavicius(alpha=alpha,
                                                                p_probs=p_bar,
                                                                y_labels=y_labels_test,
                                                                params=config[error]["params"]
                                                                )
                print(f"decision: {decision_lambda}")
                print(f"p_val: {p_val_lambda}")
                print(f"Stat: {stat_lambda}")
                # compare to mean prediction
                mean_preds = data_test[2].mean(axis=1)
                decision_mean, p_val_mean, stat_mean = npbe_test_vaicenavicius(alpha=alpha,
                                                                p_probs=mean_preds,
                                                                y_labels=y_labels_test,
                                                                params=config[error]["params"]
                                                                )
                
                print(f"decision with mean prediciton: {decision_mean}")
                print(f"p_val: {p_val_mean}")
                print(f"Stat: {stat_mean}")

                # save results for both the optimized lambda and mean predcition
                results_list.append({
                    "ensemble_size": n_ens,
                    "model": model,
                    "error_metric": error,
                    "decision_lambda": decision_lambda,
                    "p_val_lambda": p_val_lambda,
                    "stat_lambda": stat_lambda,
                    "decision_mean": decision_mean,
                    "p_val_mean": p_val_mean,
                    "stat_mean": stat_mean,
                })
    df_results = pd.DataFrame(results_list)

    # Save results
    csv_path = os.path.join(args.results_dir, "calibration_analysis_results.csv")
    df_results.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train MLP calibration model")

        parser.add_argument(
            "--alpha",
            type=float,
            default=0.05,
            help="Significance level for the test"
        )
        
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
            default=128, 
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