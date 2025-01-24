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
        verbose: bool = False
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

        # define losses, train_modes, calibrators as needed
        self.losses = [GeneralizedBrierLoss(), GeneralizedLogLoss()]
        self.train_modes = ["joint", "alternating", "avg_then_calibrate"]
        # if you only want e.g. Dirichlet + Temperature
        self.calibrators = [DirichletCalibrator, TemperatureScalingCalibrator]

        # metric params
        self.dict_mmd = {"bw": 0.1}
        self.dict_skce = {"bw": 0.05}
        self.dict_kde_ece = {"p": 2, "bw": 0.05}

        os.makedirs(output_dir, exist_ok=True)

    def measure_calibration_metrics(self, p_cal, y_tensor):
        """
        Return a dict: {brier, mmd, skce, ece_kde}
        """
        # brier
        brier_val = brier_obj(p_cal, y_tensor)
        mmd_val = mmd_kce(p_cal, y_tensor, bw=self.dict_mmd["bw"])
        skce_val = get_skce_ul(p_cal, y_tensor, bw=self.dict_skce["bw"])
        ece_val = get_ece_kde(
            p_cal, y_tensor, p=self.dict_kde_ece["p"], bw=self.dict_kde_ece["bw"]
        )
        d = {}
        d["brier"] = float(abs(brier_val))
        d["mmd"] = float(abs(mmd_val))
        d["skce"] = float(abs(skce_val))
        d["ece_kde"] = float(abs(ece_val))
        return d

    def run(self):
        # 1) Load predictions, instances, labels
        predictions_np, instances_np, labels_np = load_results_real_data(
            dataset_name=self.dataset_name,
            model_type=self.model_type,
            ensemble_type=self.ensemble_type,
            ensemble_size=self.ensemble_size,
            directory=self.dir_predictions,  # or pass from outside
        )

        # 2) Build MLPDataset, DataLoader
        dataset_val = MLPDataset(
            x_train=instances_np,  # shape (N, C, H, W) for images
            P=predictions_np,  # shape (N, M, K)
            y=labels_np,  # shape (N,)
        )

        # We store results over repeats
        results = {}  # {(loss_name, mode, cal_name): list_of_dicts}

        for repeat_idx in range(self.n_repeats):
            print(f"\n=== Repetition {repeat_idx+1}/{self.n_repeats} ===")
            # We'll treat loader_val as both "train" and "val" for calibrator
            # Or build a small "val" subset if you'd like
            dataset_train = dataset_val  # same data for calibration
            dataset_val2 = None  # if you want no separate val

            for loss_obj in self.losses:
                loss_name = loss_obj.__class__.__name__
                for train_mode in self.train_modes:
                    for CalCls in self.calibrators:
                        cal_name = CalCls.__name__
                        print(f" -> {loss_name} / {train_mode} / {cal_name}")

                        # Build the model
                        # shape (N, M, K) => M = # ensemble members, K = # classes
                        n_ens = predictions_np.shape[1]
                        n_classes = predictions_np.shape[2]

                        # if train_mode in ["joint", "alternating"]:
                        #     # comb_model => MLPCalWConv
                        #     # we define a lambda to pass into CredalSetCalibrator
                        #     # or just build it directly
                        def comb_builder(**kwargs):
                            return MLPCalWConv(
                                in_channels=instances_np.shape[1],  # 3 for CIFAR
                                out_channels=n_ens,
                                hidden_dim=self.hidden_dim,
                                hidden_layers=self.hidden_layers,
                                pretrained=self.pretrained,
                                pretrained_model=self.pretrained_model,
                            )

                        # else:
                        #     # "avg_then_calibrate" => comb model won't be used,
                        #     # but we still pass something: dummy
                        #     def comb_builder(**kwargs):
                        #         return MLPCalWConv(
                        #             in_channels=3,  # dummy
                        #             out_channels=n_ens,
                        #             pretrained=False,
                        #         )

                        model = CredalSetCalibrator(
                            comb_model=comb_builder,
                            cal_model=CalCls,
                            in_channels=instances_np.shape[1],  # e.g. 3
                            n_classes=n_classes,
                            n_ensembles=n_ens,
                            hidden_dim=self.hidden_dim,
                            hidden_layers=self.hidden_layers,
                        )

                        # Train
                        # Reuse your train_model
                        model, _, _ = train_model(
                            model=model,
                            dataset_train=dataset_train,
                            dataset_val=dataset_val2,
                            loss_fn=loss_obj,
                            train_mode=train_mode,
                            device=self.device,
                            n_epochs=self.n_epochs,
                            lr=self.lr,
                            batch_size=self.batch_size,
                            verbose=self.verbose,
                            early_stopping=False,
                        )

                        # Evaluate on the entire dataset
                        p_preds_tensor = (
                            torch.from_numpy(predictions_np).float().to(self.device)
                        )
                        x_tensor = (
                            torch.from_numpy(instances_np).float().to(self.device)
                        )
                        y_tensor = torch.from_numpy(labels_np).long().to(self.device)

                        with torch.no_grad():
                            if train_mode in ["joint", "alternating"]:
                                p_cal, p_bar, weights = model(x_tensor, p_preds_tensor)
                            else:
                                # average
                                p_bar = p_preds_tensor.mean(dim=1)  # (N, K)
                                p_cal = model.cal_model(p_bar)

                        # measure calibration metrics
                        metric_d = self.measure_calibration_metrics(p_cal.cpu().numpy(), y_tensor)
                        # measure accuracy
                        preds = torch.argmax(p_cal, dim=1).cpu().numpy()
                        acc = np.mean(preds == labels_np)
                        metric_d["accuracy"] = float(acc)

                        key = (loss_name, train_mode, cal_name)
                        if key not in results:
                            results[key] = []
                        results[key].append(metric_d)

        # Summarize
        final_scores = {}
        for key, list_of_dicts in results.items():
            metric_keys = list(list_of_dicts[0].keys())
            agg = {}
            for mk in metric_keys:
                vals = [d[mk] for d in list_of_dicts]
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                agg[f"{mk}_mean"] = float(mean_val)
                agg[f"{mk}_std"] = float(std_val)
            final_scores[key] = agg

        # Write CSV
        csv_path = os.path.join(
            self.output_dir,
            f"calibration_scores_{self.dataset_name}_{self.model_type}_{self.ensemble_type}_{self.ensemble_size}.csv",
        )
        os.makedirs(self.output_dir, exist_ok=True)
        # gather metric fields
        metric_fields = sorted(final_scores[next(iter(final_scores))].keys())
        columns = ["loss", "train_mode", "cal_model"] + metric_fields
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for (loss_name, tm, cal_name), aggdict in final_scores.items():
                row = [loss_name, tm, cal_name] + [aggdict[mf] for mf in metric_fields]
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
        help="Number of ensemble models or #MC passes.",
    )
    # parser.add_argument(
    #     "--ensemble_dir",
    #     type=str,
    #     default="ensemble_results",
    #     help="Directory where ensemble predictions, instances, labels are saved.",
    # )

    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for calibration."
    )
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./calibration_results")
    parser.add_argument("--n_repeats", type=int, default=1)
    parser.add_argument(
        "--pretrained", type=bool, default=True, help="Use pretrained model for the comb model."
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="resnet18",
        help="Pretrained model to use.",
    )
    parser.add_argument(
        "--dir_predictions",
        type=str,
        default="ensemble_results",
        help="Directory where ensemble predictions, instances, labels are saved.",
    )
    parser.add_argument("--verbose", type=bool, default=False, help="whether to output training losses")

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
    )
    runner.run()


if __name__ == "__main__":
    main()
