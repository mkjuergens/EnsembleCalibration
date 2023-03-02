import unittest

import numpy as np

from ensemblecalibration.calibration.minimization import solve_cobyla
from ensemblecalibration.calibration.test_objectives import confece_obj_new

class TestCobylaMethod(unittest.TestCase):

    def setUp(self) -> None:
        self.y = np.ones(3)
        P = [np.eye(3) for _ in range(3)]
        self.P = np.stack(P, axis=0)
        self.params = {"obj": confece_obj_new, "n_bins": 1}

    def test_optim_lambda(self):
        lambda_optim = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
        lambda_real = solve_cobyla(self.P, self.y, self.params)

        diff = np.sum(lambda_optim - lambda_real)
        self.assertLessEqual(diff, 0.01)





if __name__ == "__main__":
    unittest.main()