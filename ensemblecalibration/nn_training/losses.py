from typing import Optional

import torch
import numpy as np
from torch import nn

from torchvision.ops import focal_loss

from ensemblecalibration.nn_training.distances import (
    skce_ul_tensor,
    tv_distance_tensor,
    median_heuristic,
    skce_uq_tensor,
)
from ensemblecalibration.nn_training.helpers import calculate_pbar_torch
from ensemblecalibration.calibration.calibration_estimates.ece_kde import get_ece_kde, get_bandwidth


class SKCELoss(nn.Module):
    def __init__(
        self,
        use_median_bw: bool = False,
        bw: float = 2.0,
        dist_fct=tv_distance_tensor,
        use_square: bool = True,
        tensor_miscal: torch.Tensor = skce_ul_tensor,
    ) -> None:
        """_summary_

        Parameters
        ----------
        use_median_bw : bool, optional
            whether to use the median heuristic for the bandwidth of the kernel, by default False
        bw : float, optional
            bandwidth of the kernel, by default 2.0
        dist_fct : optional
            distance function used in the kernel, by default tv_distance.
            Remark: needs to be in torch tensor format
        use_square : bool, optional
            whether to square the loss function, by default True
        tensor_miscal : torch.Tensor, optional
            tensor version of miscalibration measure, by default skce_ul_tensor
        """
        super().__init__()
        self.bw = bw
        self.use_median_bw = use_median_bw
        self.dist_fct = dist_fct
        self.use_square = use_square
        self.tensor_miscal = tensor_miscal

    def forward(self, p_preds: torch.Tensor, weights_l: torch.Tensor, y: torch.Tensor):
        """forward pass of the loss function

        Parameters
        ----------
        p_preds : torch.Tensor
            tensor of shape (n_samples, n_predictors, n_classes) containing probabilistic predictions
        weights_l : torch.Tensor
            tensor of shape (n_samples, n_predictors) containing weight coefficients
        y : torch.Tensor
            tensor of shape (n_samples,) containing labels

        Returns
        -------
        loss
            value of loss function
        """
        # calculate matrix of point predictions of the convex combination
        p_bar = calculate_pbar_torch(
            weights_l=weights_l, p_preds=p_preds, reshape=False
        )
        if self.use_median_bw:
            bw = median_heuristic(p_hat=p_bar, y_labels=y)
        else:
            bw = self.bw
        # get tensor of SKCE values which is to be summed over
        hat_skce_ul = self.tensor_miscal(
            p_bar=p_bar, y=y, dist_fct=self.dist_fct, sigma=bw
        )
        # calculate mean
        loss = torch.mean(hat_skce_ul)

        if self.use_square:
            loss = torch.square(loss)

        return loss
    

class LpLoss(nn.Module):
    """Lp Calibration error as a loss function, see also Poporadanoska et al. (2022)

    """

    def __init__(self, p: int, bw: Optional[float] = None, device: str = 'cpu',
                 ) -> None:
        """
        Parameters
        ----------
        p : int
            order of the Lp norm used in the calibration error
        bw : Optional[float], optional
            badnwidth of the kernel. If None, a heuristic method is used, by default None
        device : str, optional
            device on which the calculations are performed, by default 'cpu'
        
        """
        super().__init__()
        self.p = p
        self.bw = bw
        self.device = device

    def forward(self, p_preds: torch.Tensor, weights_l: torch.Tensor, y: torch.Tensor):

        # calculate convex combination
        p_bar = calculate_pbar_torch(
            weights_l=weights_l, p_preds=p_preds, reshape=False
        )
        p_bar = p_bar.float()
        if self.bw is None:
            bw = get_bandwidth(p_bar, y)
        else:
            bw = self.bw
        assert np.isnan(p_bar.detach()).sum() == 0, f"p_bar contains {np.isnan(p_bar).sum()} NaNs"
        lp_er = get_ece_kde(p_bar, y, bw, self.p, device=self.device, mc_type="canonical")

        return lp_er



class FocalLoss(nn.Module):


    def __init__(self, alpha=1.0, gamma=2.0, reduction: str = "mean"):

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self, p_preds: torch.Tensor, weights_l: torch.Tensor, y_labels: torch.Tensor
    ):
        p_bar = calculate_pbar_torch(
            weights_l=weights_l, p_preds=p_preds, reshape=False
        )
        # use predictions for class 1
        p_pred = p_bar[:, 1]
        y_labels = y_labels.float()

        loss = focal_loss.sigmoid_focal_loss(
            p_pred, y_labels, self.alpha, self.gamma, reduction=self.reduction
        )
        return loss
    

if __name__ == "__main__":
    loss = SKCELoss()
    loss_focal = FocalLoss()
    p = torch.from_numpy(np.random.dirichlet([1] * 2, size=(100, 2)))
    lambdas = torch.from_numpy(np.random.dirichlet([1] * 2, size=100))
    y = torch.randint(2, size=(100,))
    out = loss(p, lambdas, y)
    print(out)
    loss_2 = SKCELoss(tensor_miscal=skce_uq_tensor)
    out_2 = loss_2(p, lambdas, y)
    print(out_2)
    out_focal = loss_focal(p, lambdas, y)
    print(out_focal)
    loss = LpLoss(p=2, bw=None)
    out_lp = loss(p, lambdas, y)
    print(out_lp)
