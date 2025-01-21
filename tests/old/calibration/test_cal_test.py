import unittest
import numpy as np

from ensemblecalibration.old.calibration.cal_test_new import _npbe_test_v3_alpha
from ensemblecalibration.old.calibration.config import config_new_v3
from ensemblecalibration.old.calibration.experiments import experiment_h0
from ensemblecalibration.old.sampling.lambda_sampling import uniform_weight_sampling

class TestCalTestv3(unittest.TestCase):

    def setUp(self) -> None:
        # generate
        self.n_classes = 3
        self.n_members = 5
        p_probs, y_labels = experiment_h0(1000, self.n_members, self.n_classes, u=0.01)
        test_params = config_new_v3["SKCEuq"]["params"]
        self.p_probs = p_probs
        self.y_labels = y_labels
        self.params = test_params

    def test_out(self):
        alpha = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
        result = _npbe_test_v3_alpha(p_probs=self.p_probs, y_labels=self.y_labels, alpha=alpha,
                                     params=self.params)
        print(result)

    def test_out_dirichlet(self):
        weights_l = np.random.dirichlet([1]*self.n_classes, size=1)[0, :]
        preds_1 = weights_l @ self.p_probs
        preds_2 = []
        weights_l_rep = np.repeat(weights_l.reshape(-1,1), self.n_members, axis=1)
        for n in range(self.p_probs.shape[0]):
            preds_2_n = np.sum(self.p_probs[n]*weights_l_rep, axis=0)
            preds_2.append(preds_2_n)
        preds_2 = np.stack(preds_2)

        self.assertTrue(np.allclose(preds_1, preds_2))

if __name__ == "__main__":
    unittest.main()
