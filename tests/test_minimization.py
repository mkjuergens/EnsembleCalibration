import unittest

import numpy as np

from ensemblecalibration.calibration.minimization import (
    solve_cobyla2D,
    solve_neldermead2D,
    solve_pyswarm,
)
from ensemblecalibration.calibration.calibration_estimates.helpers import (
    calculate_pbar,
    classece_obj_new,
    skce_ul_obj_new,
    skce_uq_obj_new,
)
from ensemblecalibration.calibration.calibration_estimates.helpers import (
    calculate_pbar,
    classece_obj_new,
    skce_ul_obj_new,
)


class TestCobylaMethod(unittest.TestCase):
    def setUp(self) -> None:
        """
        setup the test environment: examplary P tensor and labels, parameters for testing
        """
        self.y = np.ones(3, dtype=int)
        P = [np.eye(3) for _ in range(3)]
        self.P = np.stack(P, axis=0)
        self.params = {"obj": classece_obj_new, "n_bins": 5}
        self.weights_cobyla = solve_cobyla2D(self.P, self.y, self.params)

    def test_bounds(self):
        weights_l = solve_cobyla2D(self.P, self.y, self.params)
        p_bar = calculate_pbar(weights_l=self.weights_cobyla, P=self.P)
        # test if difference of row sums to 1 is less or equal a small epsilon
        print(weights_l)
        self.assertLessEqual((np.sum(p_bar, axis=1) - 1).sum(), 0.001)
        self.assertTrue(np.all((p_bar >= 0.0) | (p_bar <= 1.0)))
        self.assertTrue(np.all((weights_l >= 0.0) & (weights_l <= 1.0)))

    def test_constraints(self):
        p_bar = calculate_pbar(weights_l=self.weights_cobyla, P=self.P)
        weights_l = self.weights_cobyla.reshape(self.P.shape[0], -1)
        self.assertLessEqual((np.sum(weights_l, axis=1) - 1).sum(), 0.001)
        self.assertLessEqual((np.sum(p_bar, axis=1) - 1).sum(), 0.001)


class TestNelderMeadMethod(unittest.TestCase):
    def setUp(self) -> None:
        self.y = np.ones(3, dtype=int)
        P = [np.eye(3) for _ in range(3)]
        self.P = np.stack(P, axis=0)
        self.params = {"obj": classece_obj_new, "n_bins": 5}

    def test_bounds(self):
        weights_l = solve_neldermead2D(self.P, self.y, self.params)
        p_bar = calculate_pbar(weights_l=weights_l, P=self.P)
        # test if difference of row sums to 1 is less or equal a small epsilon
        print(weights_l)
        self.assertLessEqual((np.sum(p_bar, axis=1) - 1).sum(), 0.001)
        self.assertTrue(np.all((p_bar >= 0.0) | (p_bar <= 1.0)))
        self.assertTrue(np.all((weights_l >= 0.0) & (weights_l <= 1.0)))


"""
class TestPySwarmMethod(unittest.TestCase):

    def setUp(self) -> None:
        self.y = np.ones(3, dtype=int)
        P = [np.eye(3) for _ in range(3)]
        self.P = np.stack(P, axis=0)
        self.params = {"obj": classece_obj_new, "n_bins": 5}

    def test_optim_lambda(self):
        lambda_optim = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0])
        lambda_real = solve_pyswarm(self.P, self.y, self.params, maxiter=100)
        obj_optim = classece_obj_new(lambda_optim, self.P, self.y, self.params)
        obj_real = classece_obj_new(lambda_real, self.P, self.y, self.params)
        print(f"Real min objective: {obj_optim}, calculated min objective Pyswarm: {obj_real}")

        diff = np.sum(np.abs(lambda_optim - lambda_real))
        self.assertLessEqual(diff, 0.01)
        
"""

if __name__ == "__main__":
    unittest.main()
