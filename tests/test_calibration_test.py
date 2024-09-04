import unittest
import numpy as np
import torch

from ensemblecalibration.config import create_config
from ensemblecalibration.config.config_cal_test import config_binary_const_weights
from ensemblecalibration.cal_test import npbe_test_vaicenavicius, npbe_test_ensemble
from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.meta_model.losses import LpLoss, MMDLoss, SKCELoss
from ensemblecalibration.cal_estimates import ece_kde_obj, mmd_kce_obj, skce_obj
from ensemblecalibration.data.gp_binary import exp_gp


class TestCalibrationTest(unittest.TestCase):

    def setUp(self) -> None:
        # generate data for experiment for case when null hypothesis is true
        self.n_samples = 2000
        self.bounds_p = [[0.4, 0.6], [0.8, 1.0]]
        self.alphas = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
        self.test = npbe_test_ensemble
        self.n_iter = 10
        # create configuration for calibration test
        self.params = create_config(
            exp_name="gp",
            cal_test=self.test,
            loss=SKCELoss,
            n_samples=self.n_samples,
            n_resamples=100,
            obj=mmd_kce_obj,
            n_epochs=250,
            lr=0.0001,
            patience=200,
            hidden_layers=3,
            hidden_dim=64,
            bw=0.5,
            lambda_bce=.1,
            batch_size=32,
        )["params"]
        self.params_const = config_binary_const_weights["LP"]["params"]

    def test_null_alternative_hypothesis(self):
        results_h0 = np.zeros((self.n_iter, len(self.alphas)))
        results_h1 = np.zeros((self.n_iter, len(self.alphas)))

        p_vals_h0 = []
        p_vals_h1 = []

        for i in range(self.n_iter):
            # sample data
            x_inst_h0, p_preds, p_bar, y_labels_h0, _ = exp_gp(
                n_samples=self.n_samples,
                bounds_p=self.bounds_p,
                h0=True,
                x_dep=True,
                deg=2,
                gamma=0.01,
            )
            # run test
            result, p_val, stat = self.test(
                alpha=self.alphas,
                x_inst=x_inst_h0,
                p_preds=p_preds,
                y_labels=y_labels_h0,
                params=self.params_const,
            )
            results_h0[i] = result
            print(f"H0: {result}")
            p_vals_h0.append(p_val)
        # compute average results
        avg_results_h0 = np.mean(results_h0, axis=0)
        print(f"Average results for H0: {avg_results_h0}")

        for i in range(self.n_iter):
            # sample data
            (
                x_inst_h1,
                p_preds,
                p_bar_h1,
                y_labels_h1,
            ) = exp_gp(
                n_samples=self.n_samples,
                bounds_p=self.bounds_p,
                h0=False,
                x_dep=True,
                deg=2,
                gamma=0.01,
            )
            # run test
            result, p_val, stat = self.test(
                alpha=self.alphas,
                x_inst=x_inst_h1,
                p_preds=p_preds,
                y_labels=y_labels_h1,
                params=self.params_const,
            )
            results_h1[i] = result
            print(f"H1: {result}")
            p_vals_h1.append(p_val)

        # compute average results
        avg_results_h1 = np.mean(results_h1, axis=0)
        print(f"Average results for H1: {avg_results_h1}")

        # compare p values: for the alternative hypothesis, the p-values should be smaller
        self.assertGreater(
            np.mean(p_vals_h0), np.mean(p_vals_h1)
        ), f"p-values are not as expected, {np.mean(p_vals_h0)} vs {np.mean(p_vals_h1)}"


if __name__ == "__main__":
    unittest.main()
