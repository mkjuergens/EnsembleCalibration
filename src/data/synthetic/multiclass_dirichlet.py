from typing import Union

import numpy as np
import torch

from ensemblecalibration.utils.helpers import (
    calculate_pbar,
    sample_lambda
)
from ensemblecalibration.utils.polytope import get_boundary


def exp_dirichlet(
    n_samples: int,
    n_classes: int,
    n_members: int = 5,
    x_bound: list = [0.0, 5.0],
    h0: bool = True,
    x_dep: bool = True,
    uc: float = 0.5,
    setting: int = 1,
    deg_pol: int = 2,
    deg_h1: float = None,
):
    """generates synhetic data for the multiclass case, where the predictions are sampled from
    a Dirichlet distribution. The weights of the convex combination are sampled from a Dirichlet

    Parameters
    ----------
    n_samples : int
        number of instances to sample
    n_classes : int
        number of classes
    n_members : int, optional
        number of predictors, by default 5
    x_bound : list, optional
        interval defining the instance value range, by default [0.0, 5.0]
    h0 : bool, optional
        whether the null hypothesis is true, by default True
    x_dep : bool, optional
        whether in case h0==True, the convex combination is a non-constant function 0in the
        instance space, by default True
    uc : float, optional
        uncertainty budget, by default 0.5
    setting: int, optional
        setting in the case h1==True, by default 1. Options: {1, 2}.
    deg_pol : int, optional
        degree of the polynomial function used to sample the weights, by default 2
    deg_h1 : float, optional
        degree of distance from the polytope in case h1==True, by default None

    Returns
    -------
    x_inst, p_preds, p_bar, y_labels, weights_l or x_inst, p_preds, p_bar, y_labels
        instance values, probabilistic predictions, convex combination, labels, weights of cc

    Raises
    ------
    NotImplementedError
        in case h1 is chosen
    """
    # Sample instances uniformly
    x_inst = torch.FloatTensor(n_samples, 1).uniform_(*x_bound)

    # Sample ensemble predictions
    p_preds, dir_params = sample_ensemble_preds(
        x_inst, n_ens=n_members, n_classes=n_classes, uncertainty=uc
    )

    if h0:
        # Sample weights for null hypothesis
        weights_l = sample_lambda(x_inst, n_members, x_dep=x_dep, deg=deg_pol)
        p_bar = calculate_pbar(weights_l, p_preds)

        # Create a Categorical distribution and sample labels
        y_labels = torch.distributions.Categorical(probs=p_bar).sample()

        return x_inst, p_preds, p_bar, y_labels, weights_l
    else:
        # Alternative hypothesis
        p_bar, y_labels = sample_p_bar_h1(
            x_inst=x_inst,
            p_preds=p_preds,
            dir_params=dir_params,
            setting=setting,
            deg_h1=deg_h1,
        )
        return x_inst, p_preds, p_bar, y_labels


def sample_ensemble_preds(
    x_inst: torch.Tensor, n_ens: int, n_classes: int, uncertainty: Union[float, callable]
):
    # Sample dirichlet prior for each instance
    dir_prior = torch.distributions.Dirichlet(torch.ones(n_classes)).sample((x_inst.shape[0],))

    # Get Dirichlet parameters
    dir_params = sample_dir_params(x_inst, dir_prior=dir_prior, uncertainty=uncertainty)

    # Sample predictions for each ensemble member
    p_preds = torch.distributions.Dirichlet(dir_params).sample((n_ens,)).permute(1, 0, 2)

    return p_preds, dir_params


def sample_dir_params(
    x_inst: torch.tensor, dir_prior: torch.tensor, uncertainty: Union[float, callable]
):
    """samples parameters of the Dirichlet distribution per instance based on a given prior
    and uncertainty level

    Parameters
    ----------
    x_inst : torch.tensor
        tensor of instance values
    dir_prior : torch.tensor
        tensor of shape (n_classes,) containing the prior
    uncertainty : function
        uncertainty level (the higher, the less certain), defined as a function on the instance space

    Returns
    -------
    torch.tensor of shape (n_samples, n_classes)
        tensor of parameters for the Dirichlet distribution from which to sample
    """
    # evaluate function on x_inst if it is a function
    if callable(uncertainty):
        # if uncertainty is a function on the instance space, evaluate it
        uncertainty = uncertainty(x_inst)
    else:
        # if uncertainty is a constant, repeat it for each instance
        uncertainty = torch.full((x_inst.shape[0],), uncertainty, device=x_inst.device)
    params_m = dir_prior * dir_prior.shape[1] / uncertainty.unsqueeze(1)
    return params_m


def sample_p_bar_h1(
    x_inst: torch.tensor,
    p_preds: torch.tensor,
    dir_params: torch.tensor,
    deg_h1: float = None,
    setting: int = None,
):
    """generates data for the Dirichlet synthetic experiment for the case when the null hypothesis is false,
    i.e. the calibrated prediction lies outside the polytope spanned by the predictors.

    Parameters
    ----------
    x_inst : torch.tensor
        instance values
    p_preds : torch.tensor
        probabilistic predictions of the ensemble members
    dir_params : torch.tensor
        parameters of the Dirichlet distribution from which the predictions are sampled
    deg_h1: float,
        degree of distance from the polytope. Has to be in [0,1] or None.
        If None, setting has to be provided.

    setting : bool, optional
        setting for the experiment, by default 1
        Setting 1:
        Setting 2:

    Returns
    -------
    p_preds, p_bar, y_labels
        ensemble predictons, calibrated predictions, labels
    """
    if setting is not None:

        p_bar, y_labels = sample_p_bar_h1_fixed(
            x_inst=x_inst, p_preds=p_preds, dir_params=dir_params, setting=setting
        )

    elif deg_h1 is not None:
        # sample p_bar and y_labels at a certain degree of distance from the polytope
        p_bar, y_labels = sample_p_bar_h1_deg(
            x_inst=x_inst, p_preds=p_preds, dir_params=dir_params, deg=deg_h1
        )

    else:
        raise NotImplementedError(
            "Please provide a setting or a degree for the experiment"
        )

    return p_bar, y_labels


def sample_p_bar_h1_fixed(
    x_inst: torch.tensor, p_preds: torch.tensor, dir_params: torch.tensor, setting: int
):

    n_ens = p_preds.shape[1]
    n_classes = p_preds.shape[2]
    y_labels, p_bar = [], []
    for n in range(x_inst.shape[0]):
        # get center of underlying Dirichlet distribution (mean)
        p_mean = torch.distributions.Dirichlet(dir_params[n]).mean
        preds_n = p_preds[n, :, :]
        # sample predictions from Dirichlet distribution for each ensemble member
        # preds_n = torch.distributions.Dirichlet(p_mean).sample((n_ens,))
        # p_preds.append(preds_n)
        if setting == 1:
            # get class with highest probability
            c = torch.argmax(p_mean).item()
        elif setting == 2:
            # randomly select class/corner
            c = torch.randint(0, n_classes, (1,)).item()
        else:
            raise NotImplementedError(
                "Only setting 1 and 2 are implemented for the Dirichlet experiment"
            )
        # get corner of the simplex (as one-hot encoded vector of the respective class)
        p_c = torch.eye(n_classes)[c, :]
        p_b = get_boundary(preds_n, p_mean, p_c)
        # sample from connecting line between p_c and p_b
        l = torch.rand(1)[0]
        p_l = l * p_b + (1 - l) * p_c
        try:
            y_l = torch.multinomial(p_l, 1)[0]
        except ValueError as e:
            y_l = torch.argmax(p_l)
        y_labels.append(y_l)
        p_bar.append(p_l)

    y_labels = torch.stack(y_labels)
    p_bar = torch.stack(p_bar)

    return p_bar, y_labels


def sample_p_bar_h1_deg(
    x_inst: torch.tensor, p_preds: torch.tensor, dir_params: torch.tensor, deg: float
):
    n_ens = p_preds.shape[1]
    n_classes = p_preds.shape[2]
    y_labels, p_bar = [], []
    for n in range(x_inst.shape[0]):
        # get center of underlying Dirichlet distribution (mean)
        p_mean = torch.distributions.Dirichlet(dir_params[n]).mean
        preds_n = p_preds[n, :, :]
        # sample predictions from Dirichlet distribution for each ensemble member
        # preds_n = torch.distributions.Dirichlet(p_mean).sample((n_ens,))
        # p_preds.append(preds_n)
        # sample random class
        c = torch.randint(0, n_classes, (1,)).item()
        # c = torch.argmax(p_mean).item()

        # get corner of the simplex (as one-hot encoded vector of the respective class)
        p_c = torch.eye(n_classes)[c, :]

        p_b = get_boundary(preds_n, p_mean, p_c)
        # calculate p_l, i.e. the prediction that is deg distance from the polytope
        p_l = deg * p_c + (1 - deg) * p_b
        try:
            y_l = torch.multinomial(p_l, 1)[0]
        except ValueError as e:
            y_l = torch.argmax(p_l)
        y_labels.append(y_l)
        p_bar.append(p_l)

    y_labels = torch.stack(y_labels)
    p_bar = torch.stack(p_bar)

    return p_bar, y_labels


if __name__ == "__main__":
    x_inst = torch.distributions.Uniform(0, 5).sample((100,))
    weights_l = sample_lambda(x_inst, 5, x_dep=True, deg=2)
    print(weights_l)
    # print sum of weights
    print(weights_l.sum(dim=1))