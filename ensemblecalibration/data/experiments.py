from typing import Optional
import torch

from ensemblecalibration.data.gp_binary import exp_gp
from ensemblecalibration.data.dataset import MLPDataset


def get_experiment(config: dict, h0: bool = True, batch_size: Optional[int] = None, **kwargs):
    """generate data for the experiment, depending on the configuration

    Parameters
    ----------
    config : dict
        configuration for the experiment
    h0 : bool, optional
        whether the null hypothesis is true, by default True
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
            x_dep=config["params"]["x_dep"],
            deg=config["params"]["deg"],
            **kwargs,
        )
            
        # data
        dataset = MLPDataset(data[0], data[1], data[3])
        dataset.weights_l = data[4] if h0 else None
        if batch_size is None:
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=config["params"]["batch_size"], shuffle=True
            )
        else:
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )

    else:
        raise ValueError("Experiment not implemented")
    
    return data, loader, dataset
