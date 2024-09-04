import torch
import numpy as np

from ensemblecalibration.utils.helpers import calculate_pbar

 
def brier_obj(p_bar: np.ndarray, y: np.ndarray, params: dict):
    if not isinstance(p_bar, torch.Tensor):
        p_bar = torch.tensor(p_bar, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.int64)
    # one hot encoding of labels
    y_onehot = torch.eye(p_bar.shape[1])[y, :]
    # calculate brier score
    brier_score = torch.mean(torch.sum((p_bar - y_onehot) ** 2, dim=1))

    return brier_score


def brier_obj_lambda(weights_l, p_probs, y_labels, params, x_dep: bool = False):
    if x_dep:
        p_bar = calculate_pbar(weights_l, p_probs, reshape=True, n_dims=2)
    else:
        p_bar = calculate_pbar(weights_l, p_probs, reshape=False, n_dims=1)

    obj = brier_obj(p_bar, y_labels, params)
    # convert to numpy array if needed
    if isinstance(obj, torch.Tensor):
        obj = obj.numpy()

    return obj