import torch
import numpy as np
from scipy.stats import multinomial


def matrix_kernel(p: np.ndarray, q: np.ndarray, dist_fct, sigma: float = 2.0):
    """returns the matrix-valued kernel evaluated at two point predictions

    Parameters
    ----------
    p : np.ndarray
        first point prediction
    q : np.ndarray
        second point prediction
    sigma : float
        bandwidth
    dist_fct : _type_
        distance measure. Options: {tv_distance, l2_distance}

    Returns
    -------
    np.ndarray
        _description_
    """
    p = p.squeeze()
    q = q.squeeze()

    assert len(p) == len(q), "vectors need to be of the same length"
    id_k = np.eye(len(p))  # identity matrix
    return np.exp((-1 / sigma) * (dist_fct(p, q) ** 2) * id_k)


def multinomial_label_sampling(probs: np.ndarray, tensor: bool = False):
    """draws a sample y from the categorical distribution
    defined by a probaibility vector.

    Parameters
    ----------
    probs : np.ndarray
        probability vector that sums up to one

    Returns
    -------
    np.ndarray or torch.tensor
        
    """
    try:
        draws = multinomial(1, probs).rvs(size=1)[0, :]
        y = np.argmax(draws)

    except ValueError as e:
        y = np.argmax(probs)
    
    if tensor:
        y = torch.tensor(y, dtype=torch.long)

    return y


def calculate_pbar(
    weights_l: torch.Tensor, p_preds: torch.Tensor, reshape: bool = False
):
    """function to calculate the tensor of convex combinations. Taeks as input
    the weights (per instance) and the tensor of probabilistic predictions.

    Parameters
    ----------
    weights_l : torch.Tensor
        weight tensor of shape (n_samples, n_predictors) or just (n_predictors,)
    p_preds : torch.Tensor
        tensor containing all predictions per instance and predictor,
        of shape (n_samples, n_predictors, n_classes)
    reshape : bool, optional
        whether to reshape the weights. Only needed in case of instance-dependency.
         By default False

    Returns
    -------
    torch.Tensor
        tensor of shape (n_samples, n_classes) containing the convex combinations
    """

    # number of samples for which we have predictions
    n_inst = p_preds.shape[0]
    # make arrays tensors if they are not
   # weights_l = torch.tensor(weights_l) if weights_l is not torch.Tensor else weights_l
   # p_preds = torch.tensor(p_preds) if p_preds is not torch.Tensor else p_preds

    if reshape:
        assert (
            len(weights_l) % n_inst == 0
        ), " weight vector needs to be a multiple of the number of rows"
        weights_l = weights_l.reshape(n_inst, -1)

    assert (
        weights_l.shape[:2] == p_preds.shape[:2],
        " number of samples need to be the same for P and weights_l",
    )

    # calculate convex combination: sum over the second dimension
    p_bar = torch.sum(weights_l.unsqueeze(2) * p_preds, dim=1)

    return p_bar
