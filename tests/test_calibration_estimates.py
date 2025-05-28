import unittest

import torch
from src.cal_estimates import (
    get_skce_ul,
    get_skce_uq,
    get_ece_kde,
    mmd_kce,
)


class TestCalibrationEstimates(unittest.TestCase):
    def setUp(self) -> None:
        self.p_bar = torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]])
        self.y = torch.tensor([0, 1, 0, 1])
        self.bandwidth = 0.1
        self.device = "cpu"

    def test_output_shape(self):
        # squared kernel calibration error
        skce_ul_stat = get_skce_ul(self.p_bar, self.y, bw=self.bandwidth)
        skce_uq_stat = get_skce_uq(self.p_bar, self.y, sigma=self.bandwidth)

        # Lp kernel calibration error
        ece_kde_stat = get_ece_kde(
            p_bar=self.p_bar, y=self.y, bw=self.bandwidth, device=self.device
        )

        # mmd kernel calibration error
        mmd_kce_stat = mmd_kce(p_bar=self.p_bar, y=self.y, bw=self.bandwidth)

        self.assertIsInstance(skce_ul_stat, torch.Tensor)
        self.assertIsInstance(skce_uq_stat, torch.Tensor)
        self.assertIsInstance(ece_kde_stat, torch.Tensor)
        self.assertIsInstance(mmd_kce_stat, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
