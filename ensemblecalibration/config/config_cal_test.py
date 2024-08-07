from ensemblecalibration.cal_estimates import *
from ensemblecalibration.cal_test import *
from ensemblecalibration.meta_model.losses import *


def create_config_mlp(
    exp_name: str = "gp",
    cal_test=npbe_test_ensemble,
    loss: CalibrationLossBinary = LpLoss,
    bw: float = 0.01,
    lambda_bce: float = 0.0,
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
            "loss": loss(bw=bw, lambda_bce=lambda_bce),
            "n_epochs": n_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "patience": patience,
            "hidden_layers": hidden_layers,
            "hidden_dim": hidden_dim,
            "x_dep": x_dep,
            "deg": deg
        },
    }
    for key, value in kwargs.items():
        config["params"][key] = value
    return config


config_binary_classification_mlp = {
    "LP": create_config_mlp(
        exp_name="gp",
        cal_test=npbe_test_ensemble,
        loss=LpLoss,
        bw=0.002,
        n_samples=2000,
        n_resamples=100,
        obj=ece_kde_obj,
        n_epochs=300,
        lr=1e-4,
        batch_size=512,
        patience=100,
        hidden_layers=3,
        hidden_dim=64,
        x_dep=True,
        deg=2,
        bounds_p=[[0.5, 0.8], [0.7, .9]],
        p=2,
        lambda_bce=0.5,  # TODO: check this
    ),
    "SKCE": create_config_mlp(
        exp_name="gp",
        cal_test=npbe_test_ensemble,
        loss=SKCELoss,
        bw=0.05,
        n_samples=2000,
        n_resamples=100,
        obj=skce_obj,
        n_epochs=400,
        lr=1e-4,
        batch_size=512,
        patience=300,
        hidden_layers=3,
        hidden_dim=64,
        x_dep=True,
        deg=2,
        bounds_p=[[0.5, 0.8], [0.7, .9]],
        lambda_bce=0.01,  
    ),
    "MMD": create_config_mlp(
        exp_name="gp",
        cal_test=npbe_test_ensemble,
        loss=MMDLoss,
        bw=0.1,
        n_samples=2000,
        n_resamples=100,
        obj=mmd_kce_obj,
        n_epochs=400,
        lr=1e-4,
        batch_size=512,
        patience=300,
        hidden_layers=3,
        hidden_dim=64,
        x_dep=True,
        deg=2,
        bounds_p=[[0.5, 0.8], [0.7, .9]],
        lambda_bce=0.01,  
    ),
}


config_binary_const_weights = {
    "LP": {
        "experiment": "gp",
        "test": npbe_test_ensemble,
        "params": {
            "optim": "COBYLA",
            "n_samples": 2000,
            "n_resamples": 100,
            "obj": ece_kde_obj,
            "obj_lambda": ece_kde_obj_lambda,
            "bw": 0.002,
            "x_dep": False,
            "deg": 2,
            "bounds_p": [[0.5, 0.8], [0.7, .9]],
            "p": 2,
        },
    },
    "SKCE": {
        "experiment": "gp",
        "test": npbe_test_ensemble,
        "params": {
            "optim": "COBYLA",
            "n_samples": 2000,
            "n_resamples": 100,
            "obj": skce_obj,
            "obj_lambda": skce_obj_lambda,
            "bw": 0.1,
            "x_dep": False,
            "deg": 2,
            "bounds_p": [[0.5, 0.8], [0.7, .9]],
        },
    },
    "MMD": {
        "experiment": "gp",
        "test": npbe_test_ensemble,
        "params": {
            "optim": "COBYLA",
            "n_samples": 2000,
            "n_resamples": 100,
            "obj": mmd_kce_obj,
            "obj_lambda": mmd_kce_obj_lambda,
            "bw": 0.1,
            "x_dep": False,
            "deg": 2,
            "bounds_p": [[0.5, 0.8], [0.7, .9]],
        },

    },
}
 

config_dirichlet_mlp = {
    "LP": create_config_mlp(
        exp_name="dirichlet",
        cal_test=npbe_test_ensemble,
        loss=LpLoss,
        bw=0.002,
        n_samples=2000,
        n_classes=3,
        n_members=5,
        n_resamples=100,
        obj=ece_kde_obj,
        n_epochs=300,
        lr=1e-4,
        batch_size=512,
        patience=100,
        hidden_layers=3,
        hidden_dim=64,
        x_dep=True,
        deg=2,
        x_bound = [0.0, 5.0],
        p=2,
        lambda_bce=0.5,  
    ),
    "SKCE": create_config_mlp(
        exp_name="dirichlet",
        cal_test=npbe_test_ensemble,
        loss=SKCELoss,
        bw=0.05,
        n_samples=2000,
        n_classes=3,
        n_members=5,
        n_resamples=100,
        obj=skce_obj,
        n_epochs=400,
        lr=1e-4,
        batch_size=512,
        patience=300,
        hidden_layers=3,
        hidden_dim=64,
        x_dep=True,
        deg=2,
        x_bound = [0.0, 5.0],
        lambda_bce=0.01,  
    ),
    "MMD": create_config_mlp(
        exp_name="dirichlet",
        cal_test=npbe_test_ensemble,
        loss=MMDLoss,
        bw=0.1,
        n_samples=2000,
        n_classes=3,
        n_members=5,
        n_resamples=100,
        obj=mmd_kce_obj,
        n_epochs=400,
        lr=1e-4,
        batch_size=512,
        patience=300,
        hidden_layers=3,
        hidden_dim=64,
        x_dep=True,
        deg=2,
        x_bound = [0.0, 5.0],
        lambda_bce=0.01,  
    ),
}


config_dirichlet_const_weights = {
    "LP": {
        "experiment": "dirichlet",
        "test": npbe_test_ensemble,
        "params": {
            "optim": "COBYLA",
            "n_samples": 2000,
            "n_resamples": 100,
            "n_classes": 3,
            "n_members": 5,
            "obj": ece_kde_obj,
            "obj_lambda": ece_kde_obj_lambda,
            "bw": 0.002,
            "x_dep": False,
            "deg": 2,
            "x_bound": [0.0, 5.0],
            "p": 2,
        },
    },
    "SKCE": {
        "experiment": "gp",
        "test": npbe_test_ensemble,
        "params": {
            "optim": "COBYLA",
            "n_samples": 2000,
            "n_resamples": 100,
            "n_classes": 3,
            "n_members": 5,
            "obj": skce_obj,
            "obj_lambda": skce_obj_lambda,
            "bw": 0.1,
            "x_dep": False,
            "deg": 2,
            "x_bound": [0.0, 5.0],
        },
    },
    "MMD": {
        "experiment": "gp",
        "test": npbe_test_ensemble,
        "params": {
            "optim": "COBYLA",
            "n_samples": 2000,
            "n_resamples": 100,
            "n_classes": 3,
            "n_members": 5,
            "obj": mmd_kce_obj,
            "obj_lambda": mmd_kce_obj_lambda,
            "bw": 0.1,
            "x_dep": False,
            "deg": 2,
            "x_bound": [0.0, 5.0],
        },

    },
}



if __name__ == "__main__":
    print(create_config_mlp(p=2, bw=0.01))
