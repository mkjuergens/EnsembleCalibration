from ensemblecalibration.cal_estimates import *
from ensemblecalibration.cal_test import *
from ensemblecalibration.meta_model.losses import *


def create_config_binary_classification(
    exp_name: str = "gp",
    cal_test=npbe_test_ensemble,
    loss: CalibrationLossBinary = LpLoss,
    bw: float = 0.01,
    n_samples: int = 2000,
    n_resamples: int = 100,
    obj=ece_kde_obj,
    n_epochs: int = 500,
    lr: float = 1e-4,
    batch_size: int = 16,
    patience: int = 100,
    hidden_layers: int = 3,
    hidden_dim: int = 64,
    x_dep: bool = True,
    deg: int = 2,
    bounds_p: list = [[0.5, 0.6], [0.8, 1.0]],
    **kwargs
):
    """function to create dictionary with configuration for running the calibration test
    for the setting of binary classification.

    Parameters
    ----------
    exp_name : str, optional
        name of the experiment, by default "gp"
    cal_test : function, optional
        the calibration test used, by default npbe_test_vaicenavicius
    loss : CalibrationLossBinary, optional
        loss function for obtaining the optimal convex combination, by default LpLoss
    n_samples : int, optional
        number of (training) samples, by default 1000
    n_resamples : int, optional
        number of resampling iterations in the bootstrapping step, by default 100
    obj : function, optional
        objective, (i.e. miscalibration estimate). Related to the loss function,
             by default get_ece_kde
    n_epochs : int, optional
        number of epochs to train the meta learner, by default 100
    lr : float, optional
        learning rate, by default 0.01
    patience : int, optional
        patience until to perform Early Stopping, by default 15
    hidden_layers : int, optional
        number of hiddne layers in the meta model, by default 1
    hidden_dim : int, optional
        hidden dimension in the meta model, by default 32
    **kwargs : dict
        additional keyword arguments for the loss function

    Returns
    -------
    dict
        dictionary with configuration for running the calibration test
    """

    config = {
        "experiment": exp_name,
        "test": cal_test,
        "params": {
            "optim": "mlp",
            "n_samples": n_samples,
            "n_resamples": n_resamples,
            "obj": obj,
            "bw": bw,
            "loss": loss(bw=bw, **kwargs),
            "n_epochs": n_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "patience": patience,
            "hidden_layers": hidden_layers,
            "hidden_dim": hidden_dim,
            "x_dep": x_dep,
            "deg": deg,
            "bounds_p": bounds_p,
        },
    }
    for key, value in kwargs.items():
        config["params"][key] = value
    return config


config_binary_clasification = {
    "LP": create_config_binary_classification(
        exp_name="gp",
        cal_test=npbe_test_ensemble,
        loss=LpLoss,
        bw=0.02,
        n_samples=2000,
        n_resamples=100,
        obj=ece_kde_obj,
        n_epochs=300,
        lr=1e-4,
        batch_size=1000,
        patience=100,
        hidden_layers=3,
        hidden_dim=64,
        x_dep=True,
        deg=2,
        bounds_p=[[0.4, 0.6], [0.8, 1.0]],
        p=2,
        lambda_bce=.5, # TODO: check this
    ),
    "SKCE": create_config_binary_classification(
        exp_name="gp",
        cal_test=npbe_test_ensemble,
        loss=SKCELoss,
        bw=0.1,
        n_samples=2000,
        n_resamples=200,
        obj=skce_obj,
        n_epochs=600,
        lr=1e-4,
        batch_size=1000,
        patience=300,
        hidden_layers=3,
        hidden_dim=64,
        x_dep=True,
        deg=2,
        bounds_p=[[0.4, 0.6], [0.8, 1.0]],
        lambda_bce=0.01, # TODO: check this
    ),
    "MMD": create_config_binary_classification(
        exp_name="gp",
        cal_test=npbe_test_ensemble,
        loss=MMDLoss,
        bw=0.1,
        n_samples=2000,
        n_resamples=100,
        obj=mmd_kce_obj,
        n_epochs=600,
        lr=1e-4,
        batch_size=1000,
        patience=300,
        hidden_layers=3,
        hidden_dim=64,
        x_dep=True,
        deg=2,
        bounds_p=[[0.4, 0.6], [0.8, 1.0]],
        lambda_bce=0.01, # TODO: check this
    ),
}

if __name__ == "__main__":
    print(create_config_binary_classification(p=2, bw=0.01))
