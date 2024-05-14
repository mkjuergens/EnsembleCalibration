import torch
import numpy as np

def tv_distance(p_1: torch.Tensor, p_2: torch.Tensor):
    """total variation distance between two point predictions.

    Parameters
    ----------
    p_1 : torch.Tensor
        point estimate of shape (n_classes,)
    p_2 : torch.Tensor
        second point estimate of shape (n_classes,)

    Returns
    -------
    float
        variation distance
    """

    return 0.5*torch.sum(torch.abs(p_1-p_2))

def l2_distance(p_1: torch.Tensor, p_2: torch.Tensor):
    """L" distance between two point predictions given as torch.Tensors.

    Parameters
    ----------
    p_1 : torch.Tensor
        first point prediction
    p_2 : torch.Tensor
        second point porediction

    Returns
    -------
    float
        distance
    """

    return torch.sqrt(torch.sum((p_1 - p_2)**2))