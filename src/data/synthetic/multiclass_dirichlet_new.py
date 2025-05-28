from typing import Union
import numpy as np
import torch

from src.utils.helpers import calculate_pbar, sample_lambda
from src.utils.polytope import get_boundary


def syn_exp_multiclass(
    n_samples: int,
    n_classes: int,
    n_predictors: int,
    x_bound: list = [0.0, 5.0],
    h0: bool = True,
    x_dep: bool = True,
    uncertainty: float = 0.1,
    deg_h1: int = 0.1,
    deg_pol_weights: int = 2,
):
    x_inst = torch.tensor(
        np.random.uniform(x_bound[0], x_bound[1], n_samples), dtype=torch.float32
    )
    p_preds, dir_params = const_preds_ensembles(
        x_inst, n_predictors, n_classes, uncertainty=uncertainty
    )
    if h0:
        # Sample weights for null hypothesis
        weights_l = sample_lambda(
            x_inst, n_predictors, x_dep=x_dep, deg=deg_pol_weights
        )
        p_bar = calculate_pbar(weights_l, p_preds)
        y_labels = torch.distributions.Categorical(probs=p_bar).sample()
        x_inst = x_inst.view(-1, 1)


        return x_inst, p_preds, p_bar, y_labels, weights_l
    else:
        p_bar = sample_p_bar_h1_deg(p_preds[0], dir_params=dir_params, deg=deg_h1)
        p_bar = p_bar.repeat(n_samples, 1)
        y_labels = torch.distributions.Categorical(probs=p_bar).sample()
        x_inst = x_inst.view(-1, 1)
        return x_inst, p_preds, p_bar, y_labels


def const_preds_ensembles(
    x_inst, n_predictors: int, n_classes: int, uncertainty: float = 0.5
):
    """function for generating constant probabilisitc predictions for each ensemble member.
    Parameters
    ----------
    x_inst : torch.tensor
        tensor of instance values
    n_ens : int
        number of ensemble members
    n_classes : int
        number of classes
    uncertainty : float
        uncertainty level (the higher, the less certain)

    Returns
    -------
    torch.tensor, torch.tensor
        probabilistic predictions, parameters of the underlying Dirichlet distribution


    """
    # sample dirichlet prior (one for all instances)
    dir_prior = torch.distributions.Dirichlet(torch.ones(n_classes)).sample()
    dir_params = dir_prior * dir_prior.shape[0] / uncertainty
    # sample one prediction per ensemble member, repeat for all instances
    p_preds = torch.distributions.Dirichlet(dir_params).sample((n_predictors,))
    p_preds = p_preds.repeat(x_inst.shape[0], 1, 1)
    return p_preds, dir_params


def sample_p_bar_h1(
    x_inst: torch.Tensor,
    p_preds: torch.Tensor,
    dir_params: torch.Tensor,
    deg_h1: float = None,
    setting: int = None,
):
    if setting is not None:
        return sample_p_bar_h1_fixed(x_inst, p_preds, dir_params, setting)
    elif deg_h1 is not None:
        return sample_p_bar_h1_deg(x_inst, p_preds, dir_params, deg_h1)
    else:
        raise NotImplementedError(
            "Please provide a setting or a degree for the experiment"
        )


def sample_p_bar_h1_fixed(
    p_preds: torch.Tensor, dir_params: torch.Tensor, setting: int
):
    n_samples, n_ens, n_classes = p_preds.shape
    p_mean = torch.distributions.Dirichlet(dir_params).mean

    c = (
        torch.argmax(p_mean, dim=1)
        if setting == 1
        else torch.randint(0, n_classes, (n_samples,))
    )
    p_c = torch.eye(n_classes, device=p_preds.device)[c]

    p_b = get_boundary(p_preds, p_mean, p_c)
    l = torch.rand(n_samples, 1, device=p_preds.device)
    p_l = l * p_b + (1 - l) * p_c

    y_labels = torch.multinomial(p_l, 1).squeeze()

    return p_l, y_labels


def sample_p_bar_h1_deg(p_preds: torch.Tensor, dir_params: torch.Tensor, deg: float):
    if p_preds.dim() == 2:
        # If p_preds is of shape (M, K)
        n_ens, n_classes = p_preds.shape
        p_mean = torch.distributions.Dirichlet(dir_params).mean

        # Sample random class per instance
        c = torch.randint(0, n_classes, (1,), device=p_preds.device).item()
        p_c = torch.eye(n_classes, device=p_preds.device)[c]

        # Get boundary and calculate p_l
        p_b = get_boundary(p_preds, p_mean, p_c)
        p_l = deg * p_c + (1 - deg) * p_b

    elif p_preds.dim() == 3:
        # If p_preds is of shape (N, M, K)
        n_samples, n_ens, n_classes = p_preds.shape
        p_mean = torch.distributions.Dirichlet(dir_params).mean

        # Sample random class for each instance in batch
        c = torch.randint(0, n_classes, (n_samples,), device=p_preds.device)
        p_c = torch.eye(n_classes, device=p_preds.device)[c]

        # Get boundary and calculate p_l
        p_b = get_boundary(p_preds, p_mean, p_c)
        p_l = deg * p_c + (1 - deg) * p_b

    else:
        raise ValueError("p_preds must be either of shape (M, K) or (N, M, K)")

    return p_l

