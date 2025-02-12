import torch
import os
import argparse
import pandas as pd
import numpy as np

from ensemblecalibration.meta_model import (
    CredalSetCalibrator,
    MLPCalWConv,
    DirichletCalibrator,
)
from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.cal_test import npbe_test_vaicenavicius
from ensemblecalibration.meta_model.train import get_optim_lambda_mlp
from ensemblecalibration.data.real.dataset_utils import load_results_real_data
from ensemblecalibration.config.config_cal_test import create_config_proper_losses
from ensemblecalibration.utils.helpers import calculate_pbar, flatten_dict
from ensemblecalibration.losses.proper_losses import (
    GeneralizedBrierLoss,
    GeneralizedLogLoss,
)
class RealDataExperimentCalTest:
    """
    Trains a comb model to learn the optimal combination of ensemble predictions on validation data,
    then tests the calibration on (separate) test data.
    """

    def __init__(
        self,
        dir_predictions: str,
        dataset_name: str,
        model_type: str,
        ensemble_type: str,
        ensemble_size: int,
        device: str = "cuda",
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        hidden_dim: int = 128,
        hidden_layers: int = 1,
        output_dir: str = "./calibration_test_results",
        n_resamples: int = 100,
        pretrained: bool = True,
        pretrained_model: str = "resnet18",
        verbose: bool = False,
        early_stopping: bool = True,
        patience: int = 10,
        cal_test: callable = npbe_test_vaicenavicius,
        prefix: str = "results"
    ):
        self.dir_predictions = dir_predictions
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.ensemble_type = ensemble_type
        self.ensemble_size = ensemble_size
        self.device = device
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.output_dir = output_dir
        self.n_resamples = n_resamples
        self.pretrained = pretrained
        self.pretrained_model = pretrained_model
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patience = patience
        self.cal_test = cal_test
        self.prefix = prefix

        # define losses, train_modes, calibrators
        self.losses = [GeneralizedBrierLoss(), GeneralizedLogLoss()]
        self.train_mode = "comb"

        # metric parameters
        self.dict_mmd = {"bw": 0.01}
        self.dict_skce = {"bw": 0.0001}
        self.dict_kde_ece = {"p": 2, "bw": 0.01}
        self.dict_kde_kl = {"bw": 0.01}

        os.makedirs(output_dir, exist_ok=True)

    def load_datasets(self):
        val_prefix = "val"
        predictions_val, instances_val, labels_val = load_results_real_data(
            dataset_name=self.dataset_name,
            model_type=self.model_type,
            ensemble_type=self.ensemble_type,
            ensemble_size=self.ensemble_size,
            directory=self.dir_predictions,
            file_prefix=val_prefix,
        )
        dataset_1 = MLPDataset(x_train=instances_val, P=predictions_val, y=labels_val)
        test_prefix = "test"
        predictions_test, instances_test, labels_test = load_results_real_data(
            dataset_name=self.dataset_name,
            model_type=self.model_type,
            ensemble_type=self.ensemble_type,
            ensemble_size=self.ensemble_size,
            directory=self.dir_predictions,
            file_prefix=test_prefix,
        )
        dataset_2 = MLPDataset(
            x_train=instances_test,
            P=predictions_test,
            y=labels_test,
        )
        return dataset_1, dataset_2

    def learn_comb(self, dataset_train, dataset_test, loss_fn):

        n_classes = dataset_train.p_probs.shape[2]
        n_ens = dataset_train.p_probs.shape[1]
        print(f"pretrained: {self.pretrained}")
        def comb_builder(**kwargs):
            return MLPCalWConv(
                in_channels=dataset_train.x_train.shape[1],
                out_channels=n_ens,
                hidden_dim=self.hidden_dim,
                hidden_layers=self.hidden_layers,
                pretrained=self.pretrained,
                pretrained_model=self.pretrained_model,
            )

        model = CredalSetCalibrator(
            comb_model=comb_builder,
            cal_model=DirichletCalibrator,
            in_channels=dataset_train.x_train.shape[1],
            n_classes=n_classes,
            n_ensembles=n_ens,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        l_weights, _, _ = get_optim_lambda_mlp(
            dataset_train=dataset_train,
            dataset_val=dataset_test,
            dataset_test=dataset_test,
            model=model,
            loss=loss_fn,
            n_epochs=self.n_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            patience=self.patience,
            device=self.device,
            verbose=self.verbose,
        )

        return l_weights

    def run(self, alpha: float = 0.01):

        # make output directory if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)

        config_params = create_config_proper_losses(
            exp_name="real_data",
            cal_test=self.cal_test,
            n_resamples=self.n_resamples,
            n_epochs=self.n_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            patience=self.patience,
            hidden_layers=self.hidden_layers,
            hidden_dim=self.hidden_dim,
            device=self.device,
        )
        cal_errors = ["LP", "KL"]
        # take only LP and KL rom config_params since the others are too computationally expensive
        config_params = {k: v for k, v in config_params.items() if k in cal_errors}

        results = {}

        dataset_1, dataset_2 = self.load_datasets()
        for loss_fn in self.losses:
            loss_fn_name = loss_fn.__name__
            results[loss_fn_name] = {}
            # iterate over keys in config_params (cal erros)
            l_weights_test = self.learn_comb(dataset_2, dataset_1, loss_fn).detach().cpu()

            p_bar = calculate_pbar(
                weights_l=l_weights_test, p_preds=dataset_1.p_probs
            ).detach()
            y_labels_test = dataset_1.y_true
            pred_labels_lambda = torch.argmax(p_bar, dim=1) #.to(self.device)
            accuracy_lambda = (
                (pred_labels_lambda == y_labels_test).float().mean().item()
            )
            print(f"Accuracy with learned lambda: {accuracy_lambda}")
            results[loss_fn_name]["accuracy_lambda"] = accuracy_lambda

            # Compare to mean prediction
            mean_preds = torch.tensor(dataset_1.p_probs.mean(axis=1)) #.to(self.device)
            pred_labels_mean = torch.argmax(mean_preds, dim=1)
            accuracy_mean = (pred_labels_mean == y_labels_test).float().mean().item()
            print(f"Accuracy with mean prediction: {accuracy_mean}")
            results[loss_fn_name]["accuracy_mean"] = accuracy_mean

            # use random sub sample of size 1000 for the test
            idx = np.random.choice(
                dataset_1.x_train.shape[0], 1000, replace=False

            )
            p_probs_test = p_bar[idx]
            y_labels_test = y_labels_test[idx]
            mean_preds_test = mean_preds[idx]
            for cal_error in config_params.keys():
                decision_lambda, p_val_lambda, stat_lambda = self.cal_test(
                    alpha=alpha,
                    p_probs=p_probs_test.cpu().numpy(),
                    y_labels=y_labels_test.cpu().numpy(),
                    params=config_params[cal_error]["params"],
                )
                print(f"decision for {cal_error}: {decision_lambda}")
                print(f"p_val: {p_val_lambda}")
                print(f"Stat: {stat_lambda}")
                results[loss_fn_name][cal_error] = {
                    "decision_lambda": decision_lambda,
                    "p_val_lambda": p_val_lambda,
                    "stat_lambda": stat_lambda,
                }

                decision_mean, p_val_mean, stat_mean = self.cal_test(
                    alpha=alpha,
                    p_probs=mean_preds_test.cpu().numpy(),
                    y_labels=y_labels_test.cpu().numpy(),
                    params=config_params[cal_error]["params"],
                )

                print(f"decision with mean prediction, {cal_error}: {decision_mean}")
                print(f"p_val: {p_val_mean}")
                print(f"Stat: {stat_mean}")

                results[loss_fn_name][cal_error]["decision_mean"] = decision_mean
                results[loss_fn_name][cal_error]["p_val_mean"] = p_val_mean
                results[loss_fn_name][cal_error]["stat_mean"] = stat_mean

        # create a dataframe
        rows = []
        for loss_name, metrics in results.items():
            flat_metrics = flatten_dict(metrics)
            flat_metrics["loss_fn"] = loss_name
            rows.append(flat_metrics)

        df_results = pd.DataFrame(rows)
        csv_file = f"{self.prefix}_{alpha}_{self.dataset_name}_{self.model_type}_{self.ensemble_type}_{self.ensemble_size}.csv"
        csv_path = os.path.join(self.output_dir, csv_file)
        df_results.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP calibration model and test")

    parser.add_argument(
        "--alpha", type=float, default=0.01, help="Significance level for the test"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="CIFAR100", help="Dataset name."
    )
    parser.add_argument("--model_type", type=str, default="resnet", help="Model type.")
    # parser.add_argument(
    #     "--reg", type=bool, default=False, help="Regularization of the optimization"
    # )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="#models in ensemble or #MC passes.",
    )
    parser.add_argument(
        "--ensemble_type",
        type=str,
        default="deep_ensemble",
        help="deep_ensemble or mc_dropout.",
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for calibration."
    )
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--n_resamples", type=int, default=100)
    parser.add_argument(
        "--pretrained", type=bool, default=True, help="Use pretrained comb model."
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="resnet18",
        help="Which pretrained model.",
    )
    parser.add_argument(
        "--dir_predictions",
        type=str,
        default="ensemble_results",
        help="Dir where val/test ensemble predictions are saved.",
    )
    parser.add_argument("--output_dir", type=str, default="./cal_test_results")
    parser.add_argument(
        "--verbose", type=bool, default=True, help="whether to output training losses"
    )
    parser.add_argument(
        "--early_stopping",
        type=bool,
        default=True,
        help="whether to use early stopping",
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="patience for early stopping"
    )
    parser.add_argument("--prefix", type=str, default="results")

    args = parser.parse_args()
    experiment = RealDataExperimentCalTest(
        dir_predictions=args.dir_predictions,
        dataset_name=args.dataset_name,
        model_type=args.model_type,
        ensemble_type=args.ensemble_type,
        ensemble_size=args.ensemble_size,
        device=args.device,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        output_dir=args.output_dir,
        n_resamples=args.n_resamples,
        pretrained=args.pretrained,
        pretrained_model=args.pretrained_model,
        verbose=args.verbose,
        early_stopping=args.early_stopping,
        patience=args.patience,
        cal_test=npbe_test_vaicenavicius,
        prefix=args.prefix
    )
    experiment.run(alpha=args.alpha)

    # parser.add_argument(
    #     "--preds_dir",
    #     type=str,
    #     default="./data/real/cal_results",
    #     help="Directory to load results from",
    # )

    # parser.add_argument(
    #     "--dataset", type=str, default="CIFAR10", help="Dataset to train on"
    # )

    # parser.add_argument(
    #     "--ensemble_size",
    #     type=int,
    #     default=5,
    #     help="Number of models in the ensemble (only applicable for deep ensembles)",
    # )

    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     default="resnet",
    #     choices=["resnet", "vgg"],
    #     help="Model type: resnet or vgg",
    # )

    # parser.add_argument(
    #     "--ensemble_type",
    #     type=str,
    #     default="deep_ensemble",
    #     choices=["deep_ensemble", "mc_dropout"],
    #     help="Ensemble type: deep ensemble or MCDropout",
    # )

    # parser.add_argument(
    #     "--hidden_dim",
    #     type=int,
    #     default=128,
    #     help="Hidden dimension of the ConvNet",
    # )

    # parser.add_argument(
    #     "--hidden_layers",
    #     type=int,
    #     default=3,
    #     help="Number of hidden layers in the ConvNet",
    # )

    # parser.add_argument(
    #     "--epochs", type=int, default=300, help="Number of epochs to train for"
    # )

    # parser.add_argument(
    #     "--patience",
    #     type=int,
    #     default=100,
    #     help="Number of epochs to wait before early stopping",
    # )

    # parser.add_argument(
    #     "--batch_size", type=int, default=512, help="Batch size for training"
    # )

    # parser.add_argument(
    #     "--val_split",
    #     type=float,
    #     default=0.1,
    #     help="Fraction of training data to use for validation",
    # )

    # parser.add_argument(
    #     "--lr",
    #     type=float,
    #     default=0.0005,
    #     help="leanring rate for training the convmlp",
    # )

    # parser.add_argument(
    #     "--device", type=str, default="cuda:0", help="Device to train on"
    # )

    # parser.add_argument(
    #     "--verbose", type=bool, default=True, help="Print results and loss"
    # )
    # parser.add_argument(
    #     "--split_data",
    #     type=bool,
    #     default=False,
    #     help="whether to split the data into train, val, test",
    # )

    # parser.add_argument(
    #     "--n_samples",
    #     type=int,
    #     default=2000,
    #     help="number of samples to be used on the test set",
    # )

    # parser.add_argument(
    #     "--stratified",
    #     type=bool,
    #     default=False,
    #     help="whether to do stratified sampling on the train data",
    # )
    # args = parser.parse_args()
    # main(args)


# "dataset": dataset_name,  # Add dataset to the results
#                             "ensemble_size": n_ens,
#                             "model": model_type,
#                             "error_metric": error,
#                             "decision_lambda": decision_lambda,
#                             "p_val_lambda": p_val_lambda,
#                             "stat_lambda": stat_lambda,
#                             "accuracy_lambda": accuracy_lambda,
#                             "decision_mean": decision_mean,
#                             "p_val_mean": p_val_mean,
#                             "stat_mean": stat_mean,
#                             "accuracy_mean": accuracy_mean,


# def main(args):
#     LIST_ERRORS = ["LP", "Brier", "MMD", "SKCE"]
#     LIST_MODELS = ["vgg", "resnet"]
#     LIST_N_ENS = [5, 10]
#     LIST_DATASETS = ["CIFAR10"]  # Add a list for datasets

#     # Save results in a dictionary
#     results_list = []

#     for dataset_name in LIST_DATASETS:  # Loop over datasets
#         print(f"Running analysis for {dataset_name}")
#         for n_ens in LIST_N_ENS:
#             for model_type in LIST_MODELS:
#                 for error in LIST_ERRORS:
#                     print(
#                         f"Running calibration analysis for {error}, {model_type}, {dataset_name} with {n_ens} ensemble members"
#                     )
#                     config = create_config_test(
#                         cal_test=npbe_test_vaicenavicius,
#                         n_resamples=100,
#                         n_epochs=args.epochs,
#                         lr=args.lr,
#                         batch_size=args.batch_size,
#                         patience=args.patience,
#                         hidden_layers=args.hidden_layers,
#                         hidden_dim=args.hidden_dim,
#                         device=args.device,
#                         reg=args.reg,
#                     )

#                     # Load predictions on test set, instance features, and labels for the current dataset
#                     p_preds, x_inst, y_labels = load_results(
#                         dataset_name=dataset_name,  # Pass the dataset name dynamically
#                         model_type=model_type,  # Pass the model type (resnet, vgg)
#                         ensemble_type=args.ensemble_type,  # Pass the ensemble type (deep_ensemble, mc_dropout)
#                         ensemble_size=n_ens,  # Ensemble size only for deep ensemble
#                         directory=args.results_dir,
#                     )

#                     x_inst = torch.from_numpy(x_inst).to(args.device)
#                     y_labels = torch.from_numpy(y_labels).to(args.device)
#                     assert p_preds.shape[0] == x_inst.shape[0], "Data mismatch"

#                     # Initialize model
#                     model = MLPCalWithPretrainedModel(
#                         out_channels=p_preds.shape[1],
#                         hidden_dim=128,
#                         hidden_layers=1,
#                         pretrained_model=model_type,
#                     ).to(args.device)

#                     for param in model.parameters():
#                         if (
#                             param.dim() > 1
#                         ):  # Only initialize weights, not biases (biases are typically 1-dimensional)
#                             torch.nn.init.xavier_uniform_(param)

#                     # Split data into train, validation, and test (train and val are used to train the MLP)
#                     if args.split_data:
#                         if args.verbose:
#                             print("Splitting data...")
#                         data_test, data_train, data_val = test_train_val_split(
#                             p_preds, y_labels, x_inst
#                         )
#                     else:
#                         data_train = (x_inst, y_labels, p_preds)
#                         data_val = data_train
#                         data_test = data_val
#                     # select n_samples randomly from the test set
#                     idx = np.random.choice(
#                         data_test[0].shape[0], args.n_samples, replace=False
#                     )
#                     data_test = (
#                         data_test[0][idx],
#                         data_test[1][idx],
#                         data_test[2][idx],
#                     )
#                     if args.verbose:
#                         print(f"Size of test set: {data_test[0].shape[0]}")

#                     dataset_train = MLPDataset(
#                         x_train=data_train[0], P=data_train[2], y=data_train[1]
#                     )
#                     dataset_val = MLPDataset(
#                         x_train=data_val[0], P=data_val[2], y=data_val[1]
#                     )
#                     dataset_test = MLPDataset(
#                         x_train=data_test[0], P=data_test[2], y=data_test[1]
#                     )

#                     # Train model
#                     optim_l, loss_train, loss_val = get_optim_lambda_mlp(
#                         dataset_train=dataset_train,
#                         dataset_val=dataset_val,
#                         dataset_test=dataset_test,
#                         model=model,
#                         loss=config[error]["params"]["loss"],
#                         n_epochs=args.epochs,
#                         lr=args.lr,
#                         batch_size=args.batch_size,
#                         patience=args.patience,
#                         device=args.device,
#                         verbose=args.verbose,
#                         stratified=args.stratified,
#                     )

#                     # Run test, first with lambda that was found by optimization
#                     alpha = args.alpha

#                     # Ensure no gradients are tracked during evaluation
#                     with torch.no_grad():
#                         # change: evaluation on validation, not test set
#                         p_bar = calculate_pbar(
#                             weights_l=optim_l, p_preds=data_test[2]
#                         ).detach()
#                         y_labels_test = data_test[1]
#                         pred_labels_lambda = torch.argmax(p_bar, dim=1).to(args.device)
#                         accuracy_lambda = (
#                             (pred_labels_lambda == y_labels_test).float().mean().item()
#                         )

#                         decision_lambda, p_val_lambda, stat_lambda = (
#                             npbe_test_vaicenavicius(
#                                 alpha=alpha,
#                                 p_probs=p_bar,
#                                 y_labels=y_labels_test.cpu().numpy(),
#                                 params=config[error]["params"],
#                             )
#                         )
#                         print(f"decision: {decision_lambda}")
#                         print(f"p_val: {p_val_lambda}")
#                         print(f"Stat: {stat_lambda}")

#                         # Compare to mean prediction
#                         mean_preds = torch.tensor(data_test[2].mean(axis=1)).to(
#                             args.device
#                         )
#                         pred_labels_mean = torch.argmax(mean_preds, dim=1)
#                         accuracy_mean = (
#                             (pred_labels_mean == y_labels_test).float().mean().item()
#                         )

#                         decision_mean, p_val_mean, stat_mean = npbe_test_vaicenavicius(
#                             alpha=alpha,
#                             p_probs=mean_preds.cpu().numpy(),
#                             y_labels=y_labels_test.cpu().numpy(),
#                             params=config[error]["params"],
#                         )

#                         print(f"decision with mean prediction: {decision_mean}")
#                         print(f"p_val: {p_val_mean}")
#                         print(f"Stat: {stat_mean}")

#                     # Save results for both the optimized lambda and mean prediction
#                     results_list.append(
#                         {
#                             "dataset": dataset_name,  # Add dataset to the results
#                             "ensemble_size": n_ens,
#                             "model": model_type,
#                             "error_metric": error,
#                             "decision_lambda": decision_lambda,
#                             "p_val_lambda": p_val_lambda,
#                             "stat_lambda": stat_lambda,
#                             "accuracy_lambda": accuracy_lambda,
#                             "decision_mean": decision_mean,
#                             "p_val_mean": p_val_mean,
#                             "stat_mean": stat_mean,
#                             "accuracy_mean": accuracy_mean,
#                         }
#                     )

#                     # Free up memory
#                     torch.cuda.empty_cache()

#     # Convert results to a DataFrame
#     df_results = pd.DataFrame(results_list)

#     # Save the DataFrame to a CSV file
#     csv_path = os.path.join(
#         args.results_dir,
#         f"calibration_analysis_results_{args.alpha}_{args.reg}_{args.ensemble_type}_{LIST_DATASETS[0]}_{args.split_data}.csv",
#     )
#     df_results.to_csv(csv_path, index=False)

#     print(f"Results saved to {csv_path}")
