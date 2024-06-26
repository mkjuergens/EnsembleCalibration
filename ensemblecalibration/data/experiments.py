from typing import Optional
import torch

from ensemblecalibration.data.gp_binary import exp_gp
from ensemblecalibration.data.multiclass_dirichlet import exp_dirichlet
from ensemblecalibration.data.dataset import MLPDataset


def get_experiment(
    config: dict,
    h0: bool = True,
    x_dep: bool = True,
    setting: int = 1,
    batch_size: Optional[int] = None,
    **kwargs
):
    """generate data for the experiment, depending on the configuration

    Parameters
    ----------
    config : dict
        configuration for the experiment
    h0 : bool, optional
        whether the null hypothesis is true, by default True
    s_1 : bool, optional
        whether the GP sample is close to the boundary of the probability simplex (in case the
        null hypothesis is false), by default False
    **kwargs : dict
        additional keyword arguments (e.g. for the kernel used in the GP)

    Returns
    -------
    loader, dataset
        dataloader and dataset
    """

    if config["experiment"] == "gp":
        # sample data
        data = exp_gp(
            n_samples=config["params"]["n_samples"],
            bounds_p=config["params"]["bounds_p"],
            h0=h0,
            x_dep=x_dep,
            deg=config["params"]["deg"],
            setting=setting,
            **kwargs,
        )

    elif config["experiment"] == "dirichlet":
        # sample data
        data = exp_dirichlet(
            n_samples=config["params"]["n_samples"],
            n_classes=config["params"]["n_classes"],
            n_members=config["params"]["n_members"],
            x_bound=config["params"]["x_bound"],
            x_dep=x_dep,
            h0=h0,
            setting=setting
        )
    else:
        raise ValueError("Experiment not implemented")
    
    # dataset
    dataset = MLPDataset(data[0], data[1], data[3])
    dataset.weights_l = data[4] if h0 else None
    if batch_size is None:
        # check if batch size is defined in the config
        # check if key exists
        if "batch_size" in config["params"]:
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=config["params"]["batch_size"], shuffle=True
            )
        else:
            loader = None
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

    return data, loader, dataset

