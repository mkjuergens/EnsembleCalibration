import math

import numpy as np
import torch


def get_soft_binning_ece_tensor(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int,
    soft_binning_temp: float,
    use_decay: bool,
    soft_binning_decay_factor: float,
):
    # use equal length binning for easier implementation: set interval boundaries
    soft_anchors = torch.Tensor(
        np.arange(1.0 / (2.0 * n_bins), 1.0, 1.0 / n_bins), dtype=torch.Tensor.float32
    )

    # prediction_tile
    prediction_tile = torch.tile(
        torch.unsqueeze(predictions, 1), [1, soft_anchors.shape[0]]
    ).unsqueeze(2)
    bin_achors_tile = torch.tile(
        torch.unsqueeze(soft_anchors, 0), [predictions.shape[0], 1]
    )

    bin_achors_tile = torch.unsqueeze(bin_achors_tile, 2)

    if use_decay:
        soft_binning_temp = 1 / (math.log(soft_binning_decay_factor) * n_bins * n_bins)
    return


def calculate_pbar_torch(
    weights_l: torch.Tensor, p_preds: torch.Tensor, reshape: bool = False
):
    """calculates the matrix of convex combinations

    Parameters
    ----------
    weights_l : torch.Tensor
        _description_
    p_preds : torch.Tensor
        _description_
    reshape : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    n_rows = p_preds.shape[0]
    if reshape:
        assert (
            len(weights_l) % n_rows == 0
        ), " weight vector needs to be a multiple of the "
        "number of rows"
        weights_l = weights_l.reshape(n_rows, -1)

    assert (
        weights_l.shape[0] == p_preds.shape[0]
    ), " number of samples need to be the same for P and weights_l"
    assert (
        weights_l.shape[1] == p_preds.shape[1]
    ), " number of ensemble members need to be the same for P and weights_l"

    p_bar = (weights_l.unsqueeze(2) * p_preds).sum(-2)

    return p_bar


if __name__ == "__main__":
    p = torch.from_numpy(np.random.dirichlet([1] * 3, size=(100, 10)))
    lambdas = torch.from_numpy(np.random.dirichlet([1] * 10, size=100))
    p_bar = calculate_pbar_torch(weights_l=lambdas, p_preds=p)
    print(p_bar.shape)
