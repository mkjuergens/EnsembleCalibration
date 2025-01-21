import torch
from torch import nn
import torch.nn.functional as F
from ensemblecalibration.utils.helpers import calculate_pbar


def create_calibrator(
    calibrator_name: str, n_classes: int, *args, **kwargs
) -> nn.Module:

    calibrator_name = calibrator_name.lower()
    if calibrator_name == "linear":
        return LinearCalibrator(n_classes=n_classes, *args, **kwargs)
    elif calibrator_name == "dirichlet":
        return DirichletCalibrator(n_classes=n_classes, *args, **kwargs)
    elif calibrator_name == "temperature":
        return TemperatureScalingCalibrator(n_classes=n_classes, *args, **kwargs)
    # elif ...
    else:
        raise ValueError(f"Unknown calibrator name: {calibrator_name}")


class LinearCalibrator(nn.Module):
    """calibrater network that maps from (combined) probability vector to he

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(self, n_classes: int, *args, **kwargs):
        super().__init__()

        # model to learn convex combination
        self.n_classes = n_classes
        # use sigmoid as additional layer
        self.linear = nn.Linear(n_classes, n_classes)

    def forward(self, p: torch.Tensor):
        """forward pass of the calibration mapping. Takes as input a probability vector,
        outputs a recalibrated probability vector.

        Parametersl
        ----------
        p : torch.Tensor
            tensor of shape (n_samples, n_classes) containing the probabilities of the
            base model.

        Returns
        -------
        torch.Tensor
            tensor of shape (n_samples, n_classes) containing the recalibrated probabilities.
        """
        # apply linear layer and sigmoid
        out = self.linear(p)
        out = F.softmax(out, dim=-1)

        return out


class DirichletCalibrator(nn.Module):

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.W = nn.Parameter(torch.eye(n_classes, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(n_classes, dtype=torch.float32))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: shape (batch_size, C) in [0,1], sums to 1
        return: shape (batch_size, C), re-calibrated probabilities
        """
        eps = 1e-12
        # log transform probabilities
        log_p = torch.log(p + eps)  # (batch_size, C)
        # (batch_size, C) x (C x C) -> (batch_size, C)
        # matrix multiplication for each sample can be done via log_p @ W
        z = log_p @ self.W + self.b  # shape (batch_size, C)
        p_cal = F.softmax(z, dim=-1)
        return p_cal


class TemperatureScalingCalibrator(nn.Module):
    """
    Multi-class temperature scaling when you only have probabilities, not raw logits.
    We'll transform p -> log(p) -> scale -> softmax -> p_cal.
    """

    def __init__(self, init_temp=1.0, **kwargs):
        super().__init__()
        # We'll learn log_temperature to ensure positivity
        self.log_temp = nn.Parameter(
            torch.log(torch.tensor([init_temp], dtype=torch.float32))
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        p: shape (batch_size, C), each row sums to ~1.0
        Returns p_cal: shape (batch_size, C)
        """
        eps = 1e-12
        # Convert probabilities to "pseudo-logits"
        # z = log(p_c + eps)
        z = torch.log(p + eps)
        # scale
        temp = torch.exp(self.log_temp)  # ensure positivity
        z_scaled = z / (temp + eps)
        # final calibrated prob
        p_cal = F.softmax(z_scaled, dim=-1)
        return p_cal


class CredalSetCalibrator(nn.Module):

    def __init__(
        self,
        comb_model: nn.Module,
        cal_model: nn.Module,
        in_channels: int,
        n_classes: int,
        n_ensembles: int,
        hidden_dim: int,
        hidden_layers: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.comb_model = comb_model(
            in_channels=in_channels,
            out_channels=n_ensembles,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            **kwargs,
        )
        self.cal_model = cal_model(n_classes=n_classes, **kwargs)
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_ensembles = n_ensembles

    def forward(self, x: torch.Tensor, p_preds: torch.Tensor):

        # 1) convex combinations given by the comb_model
        weights_l = self.comb_model(x)
        # 2) calculate convex combination
        p_bar = calculate_pbar(weights_l=weights_l, p_preds=p_preds)
        # 3) recalibrate using cal_model
        p_cal = self.cal_model(p_bar)

        return p_cal, p_bar, weights_l
