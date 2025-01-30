import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.data.real.dataset_utils import load_results_real_data
from ensemblecalibration.meta_model import (
    CredalSetCalibrator,
    MLPCalWConv,
    DirichletCalibrator,
    TemperatureScalingCalibrator,
)
from ensemblecalibration.losses.proper_losses import (
    GeneralizedBrierLoss,
    GeneralizedLogLoss,
)
from ensemblecalibration.meta_model.train import train_model
from ensemblecalibration.cal_estimates import (
    brier_obj,
    mmd_kce,
    get_skce_ul,
    get_ece_kde,
)


class RealDataExperiment:
    """
    Trains a comb/cal model on validation data, then evaluates calibration metrics on test data.
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
        output_dir: str = "./calibration_results",
        n_repeats: int = 1,
        pretrained: bool = False,
        pretrained_model: str = "resnet18",
        verbose: bool = False,
        early_stopping: bool = True,
        patience: int = 10,
        subepochs_cal: int = 1,
        subepochs_comb: int = 1,
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
        self.n_repeats = n_repeats
        self.pretrained = pretrained
        self.pretrained_model = pretrained_model
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.patience = patience
        self.subepochs_cal = subepochs_cal
        self.subepochs_comb = subepochs_comb

        # define losses, train_modes, calibrators
        self.losses = [GeneralizedBrierLoss(), GeneralizedLogLoss()]
        self.train_modes = ["joint", "alternating", "avg_then_calibrate"]
        self.calibrators = [DirichletCalibrator, TemperatureScalingCalibrator]

        # metric parameters
        self.dict_mmd = {"bw": 0.1}
        self.dict_skce = {"bw": 0.001}
        self.dict_kde_ece = {"p": 2, "bw": 0.01}

        os.makedirs(output_dir, exist_ok=True)

    def measure_calibration_metrics(
        self, p_cal: torch.Tensor, y_tensor: torch.Tensor, max_mmd_samples: int = 1000
    ):
        """
        Return a dict: {brier, mmd, skce, ece_kde, accuracy},
        using random subset for MMD if needed.
        """
        # brier
        brier_val = brier_obj(p_cal, y_tensor)
        # skce
        skce_val = get_skce_ul(p_cal, y_tensor, bw=self.dict_skce["bw"], take_square=False)
        # ece
        ece_val = get_ece_kde(
            p_cal, y_tensor, p=self.dict_kde_ece["p"], bw=self.dict_kde_ece["bw"],
        )
        # subset for MMD
        N = p_cal.shape[0]
        if N > max_mmd_samples:
            indices = torch.randperm(N, device=p_cal.device)[:max_mmd_samples]
            p_cal_mmd = p_cal[indices]
            y_mmd = y_tensor[indices]
        else:
            p_cal_mmd = p_cal
            y_mmd = y_tensor
        mmd_val = mmd_kce(p_cal_mmd, y_mmd, bw=self.dict_mmd["bw"], take_square=False)

        d = {}
        d["brier"] = float(abs(brier_val))
        d["mmd"] = float(abs(mmd_val))
        d["skce"] = float(abs(skce_val))
        d["ece_kde"] = float(abs(ece_val))
        return d

    def run(self):
        # 1) Load validation data from *val* .npy files
        # e.g. "CIFAR100_resnet_deep_ensemble_10_val_predictions.npy" etc.
        val_prefix = "val"
        predictions_val, instances_val, labels_val = load_results_real_data(
            dataset_name=self.dataset_name,
            model_type=self.model_type,
            ensemble_type=self.ensemble_type,
            ensemble_size=self.ensemble_size,
            directory=self.dir_predictions,
            file_prefix=val_prefix,
        )

        # 2) Build MLPDataset for *val*
        dataset_1 = MLPDataset(x_train=instances_val, P=predictions_val, y=labels_val)

        # 3) Load test data from *test* .npy files
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

        # We'll store final results (only tested on test set)
        results = {}  # {(loss_name,train_mode,calibrator) : [metric_dict, ...]}
        # calculate accuracy and scores on ground truth
        acc_val = np.mean(np.argmax(predictions_val.mean(axis=1), axis=1) == labels_val)
        print(f"Accuracy on Validation set: {acc_val}")
        #caliubration scores
        metric_d = self.measure_calibration_metrics(
            torch.from_numpy(predictions_val.mean(axis=1)).float(),
            torch.from_numpy(labels_val).long(),
        )
        print(f"Calibration scores on Validation set: {metric_d}")

        # 4) multiple repeats if you want random re-initialization seeds
        for repeat_idx in range(self.n_repeats):
            print(f"\n=== Repetition {repeat_idx+1}/{self.n_repeats} ===")

            # We do not build separate "val" for calibrator's early-stopping.
            # We'll pass None to dataset_val in train_model if we want no separate split.
            dataset_train = dataset_2
            dataset_test = dataset_1
            # dataset_val2 = None

            for loss_fn in self.losses:
                loss_name = loss_fn.__class__.__name__
                for train_mode in self.train_modes:
                    for CalCls in self.calibrators:
                        cal_name = CalCls.__name__
                        print(f" -> {loss_name} / {train_mode} / {cal_name}")

                        # shape check
                        n_ens = predictions_val.shape[1]
                        n_classes = predictions_val.shape[2]

                        # Build comb model (CNN-based if you'd like)
                        def comb_builder(**kwargs):
                            return MLPCalWConv(
                                in_channels=instances_val.shape[1],
                                out_channels=n_ens,
                                hidden_dim=self.hidden_dim,
                                hidden_layers=self.hidden_layers,
                                pretrained=self.pretrained,
                                pretrained_model=self.pretrained_model,
                            )

                        model = CredalSetCalibrator(
                            comb_model=comb_builder,
                            cal_model=CalCls,
                            in_channels=instances_val.shape[1],
                            n_classes=n_classes,
                            n_ensembles=n_ens,
                            hidden_dim=self.hidden_dim,
                            hidden_layers=self.hidden_layers,
                        )

                        # 5) Train on validation set
                        trained_model, _, _ = train_model(
                            model=model,
                            dataset_train=dataset_train,
                            dataset_val=dataset_test,
                            loss_fn=loss_fn,
                            train_mode=train_mode,
                            device=self.device,
                            n_epochs=self.n_epochs,
                            lr=self.lr,
                            batch_size=self.batch_size,
                            verbose=self.verbose,
                            early_stopping=self.early_stopping,
                            patience=self.patience,
                            subepochs_comb=self.subepochs_comb,
                            subepochs_cal=self.subepochs_cal
                        )

                        # 6) Evaluate on test set
                        p_preds_test_tensor = (
                            torch.from_numpy(predictions_val).float().to(self.device)
                        )
                        x_test_tensor = (
                            torch.from_numpy(instances_val).float().to(self.device)
                        )
                        y_test_tensor = torch.from_numpy(labels_val).long()

                        with torch.no_grad():
                            if train_mode in ["joint", "alternating"]:
                                p_cal_test, _, _ = trained_model(
                                    x_test_tensor, p_preds_test_tensor
                                )
                            else:
                                # avg-then-calibrate
                                p_bar_test = p_preds_test_tensor.mean(dim=1)
                                p_cal_test = trained_model.cal_model(p_bar_test)
                        p_cal_test = p_cal_test.cpu()

                        # measure calibration metrics
                        metric_d = self.measure_calibration_metrics(
                            p_cal_test, y_test_tensor
                        )

                        # measure accuracy
                        preds_test = torch.argmax(p_cal_test, dim=1).cpu().numpy()
                        acc_test = np.mean(preds_test == labels_val)
                        metric_d["accuracy"] = float(acc_test)
                        print(f"Accuracy on Test set: {acc_test}")

                        key = (loss_name, train_mode, cal_name)
                        if key not in results:
                            results[key] = []
                        results[key].append(metric_d)

        # 7) Summarize results (over repeats)
        final_scores = {}
        for key, list_of_dicts in results.items():
            metric_keys = sorted(
                list_of_dicts[0].keys()
            )  # e.g. [accuracy, brier, ece_kde, mmd, skce]
            agg = {}
            for mk in metric_keys:
                vals = [d[mk] for d in list_of_dicts]
                agg[f"{mk}_mean"] = float(np.mean(vals))
                agg[f"{mk}_std"] = float(np.std(vals))
            final_scores[key] = agg

        # 8) Write to CSV
        csv_path = os.path.join(
            self.output_dir,
            f"calibration_scores_{self.dataset_name}_{self.model_type}_{self.ensemble_type}_{self.ensemble_size}.csv",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # columns
        metric_fields = sorted(final_scores[next(iter(final_scores))].keys())
        columns = ["loss", "train_mode", "cal_model"] + metric_fields

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for (loss_name, tm, cal_name), agg_dict in final_scores.items():
                row = [loss_name, tm, cal_name] + [agg_dict[mf] for mf in metric_fields]
                writer.writerow(row)

        print(f"Saved final calibration results to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="CIFAR100", help="Dataset name."
    )
    parser.add_argument("--model_type", type=str, default="resnet", help="Model type.")
    parser.add_argument(
        "--ensemble_type",
        type=str,
        default="deep_ensemble",
        help="deep_ensemble or mc_dropout.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="#models in ensemble or #MC passes.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for calibration."
    )
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./calibration_results")
    parser.add_argument("--n_repeats", type=int, default=1)
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
    parser.add_argument(
        "--verbose", type=bool, default=True, help="whether to output training losses"
    )
    parser.add_argument(
        "--early_stopping", type=bool, default=True, help="whether to use early stopping"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="patience for early stopping"
    )
    parser.add_argument(
        "--subepochs_cal", type=int, default=5, help="subepochs for calibration model"
    )
    parser.add_argument(
        "--subepochs_comb", type=int, default=5, help="subepochs for comb model"
    )

    args = parser.parse_args()

    runner = RealDataExperiment(
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
        n_repeats=args.n_repeats,
        pretrained=args.pretrained,
        pretrained_model=args.pretrained_model,
        verbose=args.verbose,
        early_stopping=args.early_stopping,
        patience=args.patience,
        subepochs_cal=args.subepochs_cal,
        subepochs_comb=args.subepochs_comb,
    )
    runner.run()


if __name__ == "__main__":
    main()
