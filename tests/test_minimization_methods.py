import unittest

from ensemblecalibration.data.gp_binary import exp_gp
from ensemblecalibration.utils.minimization import minimize_const_weights
from ensemblecalibration.config.config_cal_test import config_binary_const_weights


class TestMinimizationOutput(unittest.TestCase):

    def setUp(self):
        self.n_samples = 1000
        self.bounds_p = [[0.4, 0.6], [0.8, 1.0]]
        self.params = config_binary_const_weights["SKCE"]["params"]

    def test_minimization_output(self):
        x_inst, p_preds, p_bar, y_labels, weights_l = exp_gp(
            n_samples=self.n_samples,
            bounds_p=self.bounds_p,
            h0=True,
            x_dep=False,
            deg=2,
            gamma=0.01,
        )
        real_stat = self.params["obj"](p_bar, y_labels, self.params)
        print(real_stat)
        results, stat = minimize_const_weights(p_probs=p_preds, y=y_labels, params=self.params,
                                                enhanced_output=True)
        print(results, stat)
        print(weights_l)
        self.assertEqual(results.shape, (2,))
         

if __name__ == "__main__":
    unittest.main()
