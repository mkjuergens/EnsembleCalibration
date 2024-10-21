import torch
import os
import argparse
import pandas as pd
import numpy as np

from ensemblecalibration.meta_model.mlp_model import MLPCalWithPretrainedModel
from ensemblecalibration.cal_test import npbe_test_vaicenavicius
from ensemblecalibration.meta_model.train import get_optim_lambda_mlp
from ensemblecalibration.data.ensemble.dataset_utils import load_results
from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.utils.helpers import test_train_val_split
from ensemblecalibration.config.config_cal_test import create_config_test
from ensemblecalibration.utils.helpers import calculate_pbar



def main(args):
    LIST_ERRORS = ["Brier", "MMD", "SKCE"]
    LIST_MODELS = ["vgg", "resnet"]
    LIST_N_ENS = [5, 10]
    LIST_DATASETS = ["CIFAR100"]  # Add a list for datasets

    # Save results in a dictionary
    results_list = []
    
    for dataset_name in LIST_DATASETS:  # Loop over datasets
        print(f"Running analysis for {dataset_name}")
        for n_ens in LIST_N_ENS:
            for model_type in LIST_MODELS:
                for error in LIST_ERRORS:
                    print(f"Running calibration analysis for {error}, {model_type}, {dataset_name} with {n_ens} ensemble members")
                    config = create_config_test(cal_test=npbe_test_vaicenavicius, n_resamples=100,
                                                n_epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                                                patience=args.patience,
                                                hidden_layers=args.hidden_layers, hidden_dim=args.hidden_dim,
                                                device=args.device, reg=args.reg)
                    
                    # Load predictions on test set, instance features, and labels for the current dataset
                    p_preds, x_inst, y_labels = load_results(
                        dataset_name=dataset_name,   # Pass the dataset name dynamically
                        model_type=model_type,       # Pass the model type (resnet, vgg)
                        ensemble_type="mc_dropout",  # Pass the ensemble type (deep_ensemble, mc_dropout)
                        ensemble_size=n_ens,  # Ensemble size only for deep ensemble
                        directory=args.results_dir
                    )
                    
                    x_inst = torch.from_numpy(x_inst).to(args.device)
                    y_labels = torch.from_numpy(y_labels).to(args.device)
                    assert p_preds.shape[0] == x_inst.shape[0], "Data mismatch"

                    # Initialize model
                    model = MLPCalWithPretrainedModel(
                        out_channels=p_preds.shape[1],
                        hidden_dim=128,
                        hidden_layers=1,
                        pretrained_model=model_type
                    ).to(args.device)

                    for param in model.parameters():
                        if param.dim() > 1:  # Only initialize weights, not biases (biases are typically 1-dimensional)
                            torch.nn.init.xavier_uniform_(param)

                    # Split data into train, validation, and test (train and val are used to train the MLP)
                    data_test, data_train, data_val = test_train_val_split(p_preds, y_labels, x_inst)

                    dataset_train = MLPDataset(
                        x_train=data_train[0], P=data_train[2], y=data_train[1]
                    )
                    dataset_val = MLPDataset(x_train=data_val[0], P=data_val[2], y=data_val[1])
                    dataset_test = MLPDataset(x_train=data_test[0], P=data_test[2], y=data_test[1])

                    # Train model
                    optim_l, loss_train, loss_val = get_optim_lambda_mlp(
                                        dataset_train=dataset_train,
                                        dataset_val=dataset_val,
                                        dataset_test=dataset_val,
                                        model=model,
                                        loss=config[error]["params"]["loss"],
                                        n_epochs=args.epochs,
                                        lr=args.lr,
                                        batch_size=args.batch_size,
                                        patience=args.patience,
                                        device=args.device,
                                        verbose=args.verbose,
                                        stratified=True)

                    # Run test, first with lambda that was found by optimization
                    alpha = args.alpha
                    
                    # Ensure no gradients are tracked during evaluation
                    with torch.no_grad():
                        # change: evaluation on validation, not test set
                        p_bar = calculate_pbar(weights_l=optim_l, p_preds=data_val[2]).detach()
                        y_labels_test = data_val[1]
                        pred_labels_lambda = torch.argmax(p_bar, dim=1).to(args.device)
                        accuracy_lambda = (pred_labels_lambda == y_labels_test).float().mean().item()

                        decision_lambda, p_val_lambda, stat_lambda = npbe_test_vaicenavicius(alpha=alpha,
                                                                            p_probs=p_bar,
                                                                            y_labels=y_labels_test.cpu().numpy(),
                                                                            params=config[error]["params"])
                        print(f"decision: {decision_lambda}")
                        print(f"p_val: {p_val_lambda}")
                        print(f"Stat: {stat_lambda}")

                        # Compare to mean prediction
                        mean_preds = torch.tensor(data_val[2].mean(axis=1)).to(args.device)
                        pred_labels_mean = torch.argmax(mean_preds, dim=1)
                        accuracy_mean = (pred_labels_mean == y_labels_test).float().mean().item()

                        decision_mean, p_val_mean, stat_mean = npbe_test_vaicenavicius(alpha=alpha,
                                                                        p_probs=mean_preds.cpu().numpy(),
                                                                        y_labels=y_labels_test.cpu().numpy(),
                                                                        params=config[error]["params"])

                        print(f"decision with mean prediction: {decision_mean}")
                        print(f"p_val: {p_val_mean}")
                        print(f"Stat: {stat_mean}")

                    # Save results for both the optimized lambda and mean prediction
                    results_list.append({
                        "dataset": dataset_name,    # Add dataset to the results
                        "ensemble_size": n_ens,
                        "model": model_type,
                        "error_metric": error,
                        "decision_lambda": decision_lambda,
                        "p_val_lambda": p_val_lambda,
                        "stat_lambda": stat_lambda,
                        "accuracy_lambda": accuracy_lambda,
                        "decision_mean": decision_mean,
                        "p_val_mean": p_val_mean,
                        "stat_mean": stat_mean,
                        "accuracy_mean": accuracy_mean
                    })

                    # Free up memory
                    torch.cuda.empty_cache()

    # Convert results to a DataFrame
    df_results = pd.DataFrame(results_list)

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(args.results_dir, f"calibration_analysis_results_va_{args.alpha}_{args.reg}_mc_dropout_100.csv")
    df_results.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train MLP calibration model")

        parser.add_argument(
            "--alpha",
            type=float,
            default=0.01,
            help="Significance level for the test"
        )
        parser.add_argument("--reg",
                            type=bool,
                            default=False,
                            help="Regularization of the optimization")
        
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
            default=300, 
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
            default=512, 
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
            default=0.0005,
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