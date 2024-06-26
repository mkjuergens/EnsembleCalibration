import numpy as np
import torch

from ensemblecalibration.utils.helpers import (
    calculate_pbar,
    multinomial_label_sampling,
    sample_function,
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
    deg: int = 2,
):
    """generates sytnhetic data for the multiclass case, where the predictions are sampled from
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
    deg : int, optional
        degree of the polynomial function used to sample the weights, by default 2

    Returns
    -------
    x_inst, p_preds, p_bar, y_labels, weights_l or x_inst, p_preds, p_bar, y_labels
        instance values, probabilistic predictions, convex combination, labels, weights of cc

    Raises
    ------
    NotImplementedError
        in case h1 is chosen
    """
    # sample instances uniformly
    x_inst = torch.tensor(
        np.random.uniform(x_bound[0], x_bound[1], n_samples), dtype=torch.float32
    )
    # sort
    p_preds, dir_params = sample_ensemble_preds(
        x_inst, n_ens=n_members, n_classes=n_classes, uncertainty=uc
    )
    if h0:
        # sample weights
        weights_l = sample_weights_h0(x_inst, n_members, x_dep=x_dep, deg=deg)
        p_bar = calculate_pbar(weights_l, p_preds)

        y_labels = torch.stack(
            [
                multinomial_label_sampling(p, tensor=True)
                for p in torch.unbind(p_bar, dim=0)
            ]
        )

    else:
        p_bar, y_labels = sample_p_bar_h1(
            x_inst=x_inst,
            p_preds=p_preds,
            dir_params=dir_params,
            setting=setting,
        )
    x_inst = x_inst.view(-1, 1)
    return (
        (x_inst, p_preds, p_bar, y_labels, weights_l)
        if h0
        else (x_inst, p_preds, p_bar, y_labels)
    )


def sample_weights_h0(
    x_inst, n_members, x_dep: bool = True, deg: int = 2, variance: int = 5
):
    n_samples = x_inst.shape[0]
    weights_l = torch.zeros((n_samples, n_members))
    if x_dep:
        for i in range(n_members):
            ivl = np.random.uniform(0, variance, 2)
            weights_l[:, i] = torch.tensor(sample_function(x_inst, deg=deg, ivl=ivl))
        # normalize
        weights_l = torch.nn.functional.softmax(weights_l, dim=1)
    else:
        # set all rows to same value, sampled from dirichlet distribution
        weights_l = (
            torch.distributions.Dirichlet(torch.ones(n_members))
            .sample()
            .repeat(n_samples, 1)
        )
    return weights_l


def sample_dir_params(
    x_inst: torch.tensor, dir_prior: torch.tensor, uncertainty: float = 0.5
):
    """samples parameters of the Dirichlet distribution per instance based on a given prior
    and uncertainty level

    Parameters
    ----------
    x_inst : torch.tensor
        tensor of instance values
    dir_prior : torch.tensor
        tensor of shape (n_classes,) containing the prior
    uncertainty : float, optional
        uncertainty level (the higher, the less certain), by default 0.5

    Returns
    -------
    torch.tensor of shape (n_samples, n_classes)
        tensor of parameters for the Dirichlet distribution from which to sample
    """
    params_m = np.zeros((x_inst.shape[0], dir_prior.shape[0]))
    for c in range(dir_prior.shape[0]):
        params_m[:, c] = (dir_prior[c] * dir_prior.shape[0]) / uncertainty
    params_m = torch.tensor(params_m)
    return params_m


def sample_ensemble_preds(
    x_inst: torch.tensor, n_ens: int, n_classes: int, uncertainty: float = 0.5
):
    p_preds = torch.zeros((x_inst.shape[0], n_ens, n_classes))
    # sample (same) prior for all ensemble members
    dir_prior = torch.distributions.Dirichlet(
        torch.ones(n_classes) / (int(n_classes / 2))
    ).sample()
    dir_params = sample_dir_params(x_inst, dir_prior=dir_prior, uncertainty=uncertainty)

    p_preds = (
        torch.distributions.Dirichlet(dir_params)
        .sample((n_ens,))
        .view(x_inst.shape[0], n_ens, n_classes)
    )
    # for i in range(n_ens):
    #     # dir_params = sample_dir_params(
    #     #     x_inst, dir_prior=dir_prior, uncertainty=uncertainty
    #     # )
    #     p_preds[:, i, :] = torch.distributions.Dirichlet(dir_params).sample()
    return p_preds, dir_params


def sample_p_bar_h1(
    x_inst: torch.tensor,
    p_preds: torch.tensor,
    dir_params: torch.tensor,
    setting: int = 1,
):
    """generates data for the Dirichlet synthetic experiment for the case when the null hypothesis is false,
    i.e. the calibrated prediction lies outside the polytope spanned by the predictors.

    Parameters
    ----------
    x_inst : torch.tensor
        instance values
    n_ens : int
        number of ensemble members
    n_classes : int
        number of classes
    uncertainty : float, optional
        "uncertainty buidget"", by default 0.5
    setting : bool, optional
        setting for the experiment, by default 1
        Setting 1:
        Setting 2:
        TODO: add more fine-grained description, and settings

    Returns
    -------
    p_preds, p_bar, y_labels
        ensemble predictons, calibrated predictions, labels
    """
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
        assert preds_n.shape == (n_ens, n_classes)
        assert p_mean.shape == (n_classes,)
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
    # p_preds = torch.stack(p_preds)
    y_labels = torch.stack(y_labels)
    p_bar = torch.stack(p_bar)

    return p_bar, y_labels
