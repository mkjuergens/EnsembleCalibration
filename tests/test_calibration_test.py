import unittest
import numpy as np
import torch

from ensemblecalibration.config import create_config_binary_classification
from ensemblecalibration.cal_test import npbe_test_vaicenavicius
from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.meta_model.losses import LpLoss, MMDLoss
from ensemblecalibration.cal_estimates import ece_kde_obj, mmd_kce_obj
from ensemblecalibration.data.gp_binary import exp_gp


class TestCalibrationTest(unittest.TestCase):

    def setUp(self) -> None:
        # generate data for experiment for case when null hypothesis is true
        self.n_samples = 1000
        self.bounds_p = [[0.4, 0.6], [0.7, 0.9]]
        self.alphas = [0.01, 0.1, 0.5, 0.9, 0.99]
        self.test = npbe_test_vaicenavicius
        self.n_iter = 10
        # create configuration for calibration test
        self.params = create_config_binary_classification(
            cal_test=self.test,
            loss=MMDLoss,
            n_samples=self.n_samples,
            n_resamples=100,
            obj=mmd_kce_obj,
            n_epochs=100,
            lr=0.01,
            patience=15,
            bw=0.01,
        )["params"]

    def test_null_alternative_hypothesis(self):
        results_h0 = np.zeros((self.n_iter, len(self.alphas)))
        results_h1 = np.zeros((self.n_iter, len(self.alphas)))

        p_vals_h0 = []
        p_vals_h1 = []

        for i in range(self.n_iter):
            # sample data
            x_inst_h0, p_preds, p_bar_h0, y_labels_h0, _ = exp_gp(
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
                p_probs=p_bar_h0,
                y_labels=y_labels_h0,
                params=self.params,
            )
            results_h0[i] = result
            print(f"H0: {result}")
            p_vals_h0.append(p_val)

        for i in range(self.n_iter):
            # sample data
            x_inst_h1, p_preds, p_bar_h1, y_labels_h1, = exp_gp(
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
                p_probs=p_bar_h1,
                y_labels=y_labels_h1,
                params=self.params,
            )
            results_h1[i] = result
            print(f"H1: {result}")
            p_vals_h1.append(p_val)

        # compare p values: for the alternative hypothesis, the p-values should be smaller
        self.assertGreater(
            np.mean(p_vals_h0), np.mean(p_vals_h1)
        ), f"p-values are not as expected, {np.mean(p_vals_h0)} vs {np.mean(p_vals_h1)}"


if __name__ == "__main__":
    unittest.main()
