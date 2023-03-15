import math

import numpy as np
import torch


def get_soft_binning_ece_tensor(predictions: torch.Tensor, labels: torch.Tensor, n_bins: int, soft_binning_temp: float,
                                use_decay: bool, soft_binning_decay_factor: float):
    
    # use equal length binning for easier implementation: set interval boundaries
    soft_anchors = torch.tensor(np.arange(1.0/(2.0 * n_bins), 1.0, 1.0 / n_bins), dtype=torch.float32)

    #prediction_tile
    prediction_tile = torch.tile(torch.unsqueeze(predictions, 1), [1, soft_anchors.shape[0]]).unsqueeze(2)
    bin_achors_tile = torch.tile(torch.unsqueeze(soft_anchors, 0), [predictions.shape[0], 1])

    bin_achors_tile = torch.unsqueeze(bin_achors_tile, 2)

    if use_decay:
        soft_binning_temp = 1 / (
            math.log(soft_binning_decay_factor) * n_bins * n_bins
        )
    return