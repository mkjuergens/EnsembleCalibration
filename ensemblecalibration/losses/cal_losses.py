from typing import Optional

import torch
import numpy as np
from torch import nn
from torch.nn import BCELoss, NLLLoss
from torchvision.ops import focal_loss

from ensemblecalibration.utils.distances import tv_distance
from ensemblecalibration.utils.helpers import calculate_pbar
from ensemblecalibration.cal_estimates.kde_ece import get_ece_kde
from ensemblecalibration.cal_estimates.skce import skce_ul_tensor, skce_uq_tensor
from ensemblecalibration.cal_estimates.mmd_kce import mmd_kce, rbf_kernel


class CalibrationLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce_loss = BCELoss()  # TODO: add one hot encoded y
        self.nll_loss = NLLLoss()

    def forward(self, p_preds: torch.Tensor, weights_l: torch.Tensor, y: torch.Tensor):
        pass

    def compute_reg_loss(self, p_bar: torch.Tensor, y: torch.Tensor):
        if p_bar.shape[1] > 2:
            # take logs of probabilities
            epsilon = 1e-10
            p_bar = torch.log(p_bar + epsilon)
            loss = self.nll_loss(p_bar, y)
        else:
            loss = self.bce_loss(p_bar[:, 1], y.float())
        return loss


class SKCELoss(CalibrationLoss):
    def __init__(
        self,
        bw: float = 2.0,
        dist_fct=tv_distance,
        use_square: bool = True,
        tensor_miscal: torch.Tensor = skce_ul_tensor,
        lambda_bce: float = 0.0,
    ) -> None:
        """loss of the squared kernel calibration error

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
        lambda_bce : float, optional
            weight of the BCE loss, or NLL loss if n_classes > 2, by default 0.0
        """
        super().__init__()
        self.bw = bw
        self.dist_fct = dist_fct
        self.use_square = use_square
        self.tensor_miscal = tensor_miscal
        self.lambda_bce = lambda_bce

    def forward(
        self,
        p_preds: torch.Tensor,
        weights_l: torch.Tensor,
        y: torch.Tensor,
    ):
        """forward pass of the loss function

        Parameters
        ----------
        p_preds : torch.Tensor
            tensor of shape (n_samples, n_predictors, n_classes) containing probabilistic
            predictions
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
        p_bar = calculate_pbar(weights_l=weights_l, p_preds=p_preds, reshape=False)
        bw = self.bw
        # get tensor of SKCE values which is to be summed over
        hat_skce_ul = self.tensor_miscal(
            p_bar=p_bar, y=y, dist_fct=self.dist_fct, bw=bw
        )
        #   if self.use_square:
        #      hat_skce_ul = torch.square(hat_skce_ul)
        # calculate mean
        loss = torch.mean(hat_skce_ul)

        if self.use_square:
            loss = torch.square(loss)  # TODO: check if this is correct

        if self.lambda_bce > 0:
            # check if n_classes > 2
            reg_loss = self.compute_reg_loss(p_bar, y)
            loss += self.lambda_bce * reg_loss

        return loss


class LpLoss(CalibrationLoss):
    """Lp Calibration error as a loss function, see also Poporadanoska et al. (2022)"""

    def __init__(
        self,
        bw: float,
        p: int = 2,
        lambda_bce: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        p : int
            order of the Lp norm used in the calibration error
        bw : float
            badnwidth of the kernel
        device : str, optional
            device on which the calculations are performed, by default 'cpu'

        """
        super().__init__()
        self.p = p
        self.bw = bw
        self.lambda_bce = lambda_bce

    def forward(
        self,
        p_preds: torch.Tensor,
        weights_l: torch.Tensor,
        y: torch.Tensor,
        debug: bool = False,
    ):

        device = weights_l.device
        # print(f"weights min: {weights_l.min()}, weights_max: {weights_l.max()}")
        # clip p_preds for numerical stability
        # p_preds = torch.clamp(p_preds, 1e-12, 1 - 1e-12)
        # weights_l = torch.clamp(weights_l, 1e-15, 1.0)
        # check for nans
        assert (
            np.isnan(p_preds.detach().cpu()).sum() == 0
        ), f"p_preds contains {np.isnan(p_preds.cpu().detach()).sum()} NaNs"
        assert (
            np.isnan(weights_l.detach().cpu()).sum() == 0
        ), f"weights contain {np.isnan(weights_l.cpu().detach()).sum()} NaNs"

        # cehck max and min values of weights
        # print(f"max: {torch.max(weights_l)}, min: {torch.min(weights_l)}")
        # calculate convex combination
        p_bar = calculate_pbar(weights_l=weights_l, p_preds=p_preds, reshape=False)
        bw = self.bw
        assert (
            np.isnan(p_bar.detach().cpu()).sum() == 0
        ), f"p_bar contains {np.isnan(p_bar.cpu().detach()).sum()} NaNs"
        # check that all y's lie in {0,1}
        loss_ece = get_ece_kde(
            p_bar, y, bw, self.p, device=device
        )  # changed: not taking only the second column
        # print(f"loss:{loss_ece}")
        if self.lambda_bce > 0:
            reg_loss = self.compute_reg_loss(p_bar, y)
            loss = loss_ece + self.lambda_bce * reg_loss
            if debug:
                return loss, loss_ece, reg_loss
            else:
                return loss

        else:
            return loss_ece


class MMDLoss(CalibrationLoss):

    def __init__(
        self, bw: float, kernel_fct=rbf_kernel, lambda_bce: float = 0.0
    ) -> None:
        super().__init__()
        self.bw = bw
        self.kernel_fct = kernel_fct
        self.lambda_bce = lambda_bce

    def forward(
        self,
        p_preds: torch.Tensor,
        weights_l: torch.Tensor,
        y: torch.Tensor,
        debug: bool = False,
    ):
        p_bar = calculate_pbar(weights_l=weights_l, p_preds=p_preds, reshape=False)
        bw = self.bw
        loss_mmd = mmd_kce(p_bar, y, kernel_fct=self.kernel_fct, bw=bw)

        if self.lambda_bce > 0:
            reg_loss = self.compute_reg_loss(p_bar, y)
            loss = loss_mmd + self.lambda_bce * reg_loss
            if debug:
                return loss, loss_mmd, reg_loss
            else:
                return loss
        else:
            return loss_mmd


class FocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0, reduction: str = "mean"):

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self, p_preds: torch.Tensor, weights_l: torch.Tensor, y_labels: torch.Tensor
    ):
        p_bar = calculate_pbar(weights_l=weights_l, p_preds=p_preds, reshape=False)
        # use predictions for class 1
        p_pred = p_bar[:, 1]
        y_labels = y_labels.float()

        loss = focal_loss.sigmoid_focal_loss(
            p_pred, y_labels, self.alpha, self.gamma, reduction=self.reduction
        )
        return loss


# Brier score
class BrierLoss(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, p_preds: torch.Tensor, weights_l: torch.Tensor, y: torch.Tensor):

        p_bar = calculate_pbar(weights_l=weights_l, p_preds=p_preds, reshape=False)
        # one-hot encoding of labels
        y_onehot = torch.eye(p_bar.shape[1]).to(y.device)[y, :]
        # calculate brier score
        brier = torch.mean(torch.sum((p_bar - y_onehot) ** 2, dim=1))

        return brier


class GeneralizedBrierLoss(nn.Module):
    """
    A flexible Brier loss class that can handle:
      1) Directly taking the "final" probabilities p_bar and labels y;
      2) Or taking ensemble preds p_preds & mixture weights, computing p_bar internally.

    Optionally applies an internal calibrator if provided.
    """

    def __init__(self, calibrator: nn.Module = None):
        """
        Parameters
        ----------
        calibrator : nn.Module, optional
            If not None, a module (e.g. small neural net)
                           that takes p_bar and outputs calibrated probabilities.
        """
        super().__init__()
        self.calibrator = calibrator

    def forward(
        self,
        y: torch.Tensor,
        p_bar: torch.Tensor = None,
        p_preds: torch.Tensor = None,
        weights_l: torch.Tensor = None,
    ) -> torch.Tensor:
        """computes the Brier score for the two scenarios
        1) directly taking p_bar and y,
        2) taking p_preds and weights_l and computing p_bar internally, optionally calibrating it.

        Usage scenarios:
            - if you already have p_bar, y, use: loss(y, p_bar)
            - if you have p_preds, weights_l, and want to compute p_bar internally, use: loss(y, p_preds, weights

        Parameters
        ----------
        y : torch.Tensor
            tensor of shape (n_samples,) containing labels
        p_bar : torch.Tensor, optional
            tensor of predictions of shape (n_samples, n_classes), by default None
        p_preds : torch.Tensor, optional
            tensor of (ensemble) predictions of shape (n_smaples, n_classes, n_ens),
              by default None
        weights_l : torch.Tensor, optional
            tensor of weights of shape (n_smaples, n_ens), by default None

        Returns
        -------
        torch.Tensor
            Brier score
        """
        # 1) If p_bar is not given, compute from ensemble preds
        if p_bar is None:
            if p_preds is None or weights_l is None:
                raise ValueError("Must provide either p_bar, or (p_preds & weights_l).")
            p_bar = calculate_pbar(weights_l=weights_l, p_preds=p_preds, reshape=False)

        # 2) If we have a calibrator, apply it
        if self.calibrator is not None:
            # calibrator should output shape (batch_size, C)
            p_bar = self.calibrator(p_bar)

        # 3) Compute the multi-class Brier score
        # one-hot encode labels
        num_classes = p_bar.shape[1]
        y_onehot = torch.eye(num_classes, device=p_bar.device)[y, :]
        # sum of squared differences
        diff_sq = (p_bar - y_onehot) ** 2
        per_sample_sum = diff_sq.sum(dim=1)  # sum over classes
        brier = per_sample_sum.mean(dim=0)  # average over batch

        return brier


if __name__ == "__main__":
    loss = SKCELoss()
    loss_focal = FocalLoss()
    p = torch.from_numpy(np.random.dirichlet([1] * 2, size=(100, 2))).float()
    lambdas = torch.from_numpy(np.random.dirichlet([1] * 2, size=100)).float()
    y = torch.randint(2, size=(100,))
    out = loss(p, lambdas, y)
    print(out)
    loss_2 = SKCELoss(tensor_miscal=skce_uq_tensor)
    out_2 = loss_2(p, lambdas, y)
    print(out_2)
    out_focal = loss_focal(p, lambdas, y)
    print(out_focal)
    loss = LpLoss(p=2, bw=0.001)
    out_lp = loss(p, lambdas, y)
    print(out_lp)
