import torch

from torch import nn
from ensemblecalibration.utils.helpers import calculate_pbar



def create_loss_fn(loss_name: str, calibrator=None, epsilon=1e-8) -> nn.Module:
    """
    Returns a loss function instance given the name.
    You can also pass an optional calibrator if needed, 
    but typically the calibrator is separate from the loss.
    """
    loss_name = loss_name.lower()
    if loss_name == "brier":
        return GeneralizedBrierLoss(calibrator=calibrator)
    elif loss_name in ["log_loss", "cross_entropy"]:
        return GeneralizedLogLoss(calibrator=calibrator, epsilon=epsilon)
    # elif ...
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")

class BaseCombinerLoss(nn.Module):
    """
    Abstract base class that handles:
      1) Combining ensemble predictions if needed,
      2) Applying an optional calibrator,
      3) Computing the final loss (defined in compute_loss).
    """

    def __init__(self, calibrator: nn.Module = None):
        """
        Initializes the base combiner loss.

        Parameters
        ----------
        calibrator : nn.Module, optional
            If provided, this module is applied to p_bar (e.g. a small neural net)
            to produce calibrated probabilities, by default None.
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
        """
        Combines the predictions (if needed), applies a calibrator (if provided),
        and computes the final loss (via compute_loss).

        Usage scenarios:
            1) If you already have `p_bar` and `y`, use:
               loss_fn(y, p_bar=p_bar)
            2) If you have `(p_preds, weights_l)` and want to compute p_bar internally:
               loss_fn(y, p_preds=p_preds, weights_l=weights_l)

        Parameters
        ----------
        y : torch.Tensor
            Tensor of shape (batch_size,) containing labels (integer class indices).
        p_bar : torch.Tensor, optional
            Tensor of predictions of shape (batch_size, n_classes). If this is provided,
            p_preds and weights_l are ignored. By default None.
        p_preds : torch.Tensor, optional
            Tensor of shape (batch_size, K, n_classes) containing ensemble predictions.
            If p_bar is None, we expect p_preds + weights_l to be provided. By default None.
        weights_l : torch.Tensor, optional
            Tensor of shape (batch_size, K) containing mixture weights. If p_bar is None,
            we combine p_preds with these weights. By default None.

        Returns
        -------
        torch.Tensor
            The computed loss as a scalar tensor.

        Raises
        ------
        ValueError
            If neither p_bar nor (p_preds, weights_l) is provided.
        """
        # 1) If p_bar is not given, compute from ensemble predictions
        if p_bar is None:
            if p_preds is None or weights_l is None:
                raise ValueError("Must provide either p_bar or (p_preds & weights_l).")
            p_bar = calculate_pbar(weights_l=weights_l, p_preds=p_preds, reshape=False)

        # 2) If we have a calibrator, apply it
        if self.calibrator is not None:
            p_bar = self.calibrator(p_bar)  # shape (batch_size, n_classes)

        # 3) Compute the final loss (child class implements compute_loss)
        return self.compute_loss(y, p_bar)

    def compute_loss(self, y: torch.Tensor, p_bar: torch.Tensor) -> torch.Tensor:
        """
        Virtual method for computing the loss.

        Parameters
        ----------
        y : torch.Tensor
            Labels of shape (batch_size,).
        p_bar : torch.Tensor
            Probabilities of shape (batch_size, n_classes).

        Returns
        -------
        torch.Tensor
            The scalar loss.

        Raises
        ------
        NotImplementedError
            Must be overridden in subclasses.
        """
        raise NotImplementedError("Child classes must implement compute_loss.")


class GeneralizedBrierLoss(BaseCombinerLoss):
    """
    Child class for computing the multi-class Brier score.
    Inherits from the base combiner, so it reuses the 'forward' logic
    for combining p_preds, applying calibrator, etc.
    """
    def __init__(self, calibrator: nn.Module = None):
        """
        Initializes the generalized Brier loss module.

        Parameters
        ----------
        calibrator : nn.Module, optional
            Module for calibrating probabilities, by default None
        """
        super().__init__(calibrator=calibrator)
        self.__name__ = "brier loss"


    def compute_loss(self, y: torch.Tensor, p_bar: torch.Tensor) -> torch.Tensor:
        """
        Computes the Brier score for the two scenarios:
            1) directly taking p_bar and y,
            2) taking p_preds and weights_l and computing p_bar internally (optionally calibrating it).

        Usage scenarios:
            - if you already have p_bar, y, call: loss_fn(y, p_bar=p_bar).
            - if you have p_preds, weights_l, call: loss_fn(y, p_preds=p_preds, weights_l=weights_l).

        Parameters
        ----------
        y : torch.Tensor
            Tensor of shape (batch_size,) containing labels (integer class indices).
        p_bar : torch.Tensor
            Tensor of shape (batch_size, n_classes) with predicted probabilities.

        Returns
        -------
        torch.Tensor
            The Brier score (scalar).
        """
        num_classes = p_bar.shape[1]
        # one-hot encode y
        y_onehot = torch.eye(num_classes, device=p_bar.device)[y]
        diff_sq = (p_bar - y_onehot) ** 2
        per_sample_sum = diff_sq.sum(dim=1)  # sum over classes
        brier = per_sample_sum.mean()  # average over batch
        return brier


class GeneralizedLogLoss(BaseCombinerLoss):
    """
    Child class for computing the multi-class negative log-likelihood (cross-entropy).
    """

    def __init__(self, calibrator: nn.Module = None, epsilon: float = 1e-8):
        """
        Initializes the generalized log-loss module.

        Parameters
        ----------
        calibrator : nn.Module, optional
            Module for calibrating probabilities, by default None
        epsilon : float, optional
            Small constant to avoid log(0), by default 1e-8
        """
        super().__init__(calibrator=calibrator)
        self.epsilon = epsilon
        self.__name__ = "log loss"

    def compute_loss(self, y: torch.Tensor, p_bar: torch.Tensor) -> torch.Tensor:
        """
        Computes the multi-class log-loss (cross-entropy) for the two scenarios:
            1) directly taking p_bar and y,
            2) taking p_preds and weights_l and computing p_bar internally (optionally calibrating it).

        Usage scenarios:
            - if you already have p_bar, y, call: loss_fn(y, p_bar=p_bar).
            - if you have p_preds, weights_l, call: loss_fn(y, p_preds=p_preds, weights_l=weights_l).

        Parameters
        ----------
        y : torch.Tensor
            Tensor of shape (batch_size,) containing labels (integer class indices).
        p_bar : torch.Tensor
            Tensor of shape (batch_size, n_classes) with predicted probabilities.

        Returns
        -------
        torch.Tensor
            Negative log-likelihood (scalar).
        """
        # Avoid log(0)
        log_p_bar = torch.log(p_bar + self.epsilon)
        # Gather log-prob of correct class
        nll = -log_p_bar.gather(dim=1, index=y.unsqueeze(1))  # shape (batch_size, 1)
        loss = nll.mean()
        return loss
