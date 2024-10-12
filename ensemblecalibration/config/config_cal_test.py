from ensemblecalibration.cal_estimates import *
from ensemblecalibration.cal_test import *
from ensemblecalibration.meta_model.losses import *


def create_config(
    exp_name: str = "gp",
    cal_test=npbe_test_ensemble,
    # loss: CalibrationLossBinary = LpLoss,
    optim: str = "mlp",
    # bw: float = 0.01,
    # lambda_bce: float = 0.0,
    n_samples: int = 2000,
    n_resamples: int = 100,
    n_classes: int = 3,
    n_members: int = 5,
    n_epochs: int = 500,
    lr: float = 1e-4,
    batch_size: int = 16,
    patience: int = 100,
    hidden_layers: int = 3,
    hidden_dim: int = 64,
    x_dep: bool = True,
    deg: int = 2,
    device: str = "cpu",
    reg: bool = False,
    bounds_p : list = [[0.5, 0.8], [0.7, .9]],
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
    optim: str
        optimization method, by default "mlp". Options: {"mlp", "COBYLA", "SLSQP"}
    device : str, optional
        device on which the calculations are performed, by default "cpu"
    n_samples : int, optional
        number of (training) samples, by default 1000
    n_resamples : int, optional
        number of resampling iterations in the bootstrapping step, by default 100
    n_classes : int, optional
        number of classes, by default 3
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
    x_dep : bool, optional
        whether the optimal weights depend on the instance, by default True
    reg: bool, optional
        whether to regularize the loss function with the cross-entropy loss, by default False
    **kwargs : dict
        additional keyword arguments for the loss function

    Returns
    -------
    dict
        dictionary with configuration for running the calibration test
    """

    # initiate loss with bw and lambda_bce if it takes these as arguments, else just bw
    # if "lambda_bce" in loss.__init__.__code__.co_varnames:
    #     loss = loss(bw=bw, lambda_bce=lambda_bce)
    # else:
    #     loss = loss(bw=bw)
    if reg:
        reg_mmd = 0.1
        reg_lp = 1.0
        reg_skce = 0.0001
    else:
        reg_mmd = 0.0
        reg_lp = 0.0
        reg_skce = 0.0

    config = {
        "MMD": {
            "experiment": exp_name,
            "test": cal_test,
            "params": {
                "optim": optim,
                "n_samples": n_samples,
                "n_resamples": n_resamples,
                "n_classes": n_classes,
                "n_members": n_members,
                "obj": mmd_kce_obj,
                "obj_lambda": mmd_kce_obj_lambda,
                "bw": 0.01, # TODO: check this
                "loss": MMDLoss(bw=0.01, lambda_bce=reg_mmd), # changed!!
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "hidden_layers": hidden_layers,
                "hidden_dim": hidden_dim,
                "x_dep": x_dep,
                "deg": deg,
                "lambda_bce": reg_mmd,
                "device": device,
                "bounds_p": bounds_p,
                **kwargs
        }
        },
        "Brier": {
            "experiment": exp_name,
            "test": cal_test,
            "params": {
                "optim": optim,
                "n_samples": n_samples,
                "n_resamples": n_resamples,
                "n_classes": n_classes,
                "n_members": n_members,
                "bw": 0.1, # TODO: check this
                "obj": brier_obj,
                "obj_lambda": brier_obj_lambda,
                "loss": BrierLoss(), # changed!!
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "hidden_layers": hidden_layers,
                "hidden_dim": hidden_dim,
                "x_dep": x_dep,
                "deg": deg,
                "device": device,
                "bounds_p": bounds_p,
                **kwargs
            }

        },
        "LP": {
            "experiment": exp_name,
            "test": cal_test,
            "params": {
                "optim": optim,
                "n_samples": n_samples,
                "n_resamples": n_resamples,
                "n_classes": n_classes,
                "n_members": n_members,
                "obj": ece_kde_obj,
                "obj_lambda": ece_kde_obj_lambda,
                "bw": 0.00001, # TODO: check this
                "loss": LpLoss(bw=0.00001, lambda_bce=reg_lp), # changed
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "hidden_layers": hidden_layers,
                "hidden_dim": hidden_dim,
                "x_dep": x_dep,
                "deg": deg,
                "p": 2,
                "lambda_bce": reg_lp, # TODO: check this
                "device": device,
                "bounds_p": bounds_p,
                **kwargs
            }
        },
        "SKCE": {
            "experiment": exp_name,
            "test": cal_test,
            "params": {
                "optim": optim,
                "n_samples": n_samples,
                "n_resamples": n_resamples,
                "n_classes": n_classes,
                "n_members": n_members,
                "obj": skce_obj,
                "obj_lambda": skce_obj_lambda,
                "bw": 0.001, # TODO: check this
                "loss": SKCELoss(bw=0.0001, lambda_bce=reg_skce), # changed!!
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "hidden_layers": hidden_layers,
                "hidden_dim": hidden_dim,
                "x_dep": x_dep,
                "deg": deg,
                "lambda_bce": reg_skce, # TODO: check this
                "device": device,
                "bounds_p": bounds_p,
                **kwargs
            }
        }
    }
    return config


def create_config_test(
    cal_test=npbe_test_vaicenavicius,
    n_resamples: int = 100,
    n_classes: int = 3,
    n_members: int = 5,
    n_epochs: int = 500,
    lr: float = 1e-4,
    batch_size: int = 16,
    patience: int = 100,
    hidden_layers: int = 3,
    hidden_dim: int = 64,
    device: str = "cpu",
    reg: bool = False,
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
    optim: str
        optimization method, by default "mlp". Options: {"mlp", "COBYLA", "SLSQP"}
    device : str, optional
        device on which the calculations are performed, by default "cpu"
    n_samples : int, optional
        number of (training) samples, by default 1000
    n_resamples : int, optional
        number of resampling iterations in the bootstrapping step, by default 100
    n_classes : int, optional
        number of classes, by default 3
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
    x_dep : bool, optional
        whether the optimal weights depend on the instance, by default True
    reg: bool, optional
        whether to regularize the loss function with the cross-entropy loss, by default False
    **kwargs : dict
        additional keyword arguments for the loss function

    Returns
    -------
    dict
        dictionary with configuration for running the calibration test
    """

    # initiate loss with bw and lambda_bce if it takes these as arguments, else just bw
    # if "lambda_bce" in loss.__init__.__code__.co_varnames:
    #     loss = loss(bw=bw, lambda_bce=lambda_bce)
    # else:
    #     loss = loss(bw=bw)
    if reg:
        reg_mmd = 0.1
        reg_lp = 1.0
        reg_skce = 0.0001
    else:
        reg_mmd = 0.0
        reg_lp = 0.0
        reg_skce = 0.0

    config = {
        "MMD": {
            "test": cal_test,
            "params": {
                "n_resamples": n_resamples,
                "n_classes": n_classes,
                "n_members": n_members,
                "obj": mmd_kce_obj,
                "obj_lambda": mmd_kce_obj_lambda,
                "bw": 0.01, # TODO: check this
                "loss": MMDLoss(bw=0.01, lambda_bce=reg_mmd), # changed!!
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "hidden_layers": hidden_layers,
                "hidden_dim": hidden_dim,
                "lambda_bce": reg_mmd,
                "device": device,
                **kwargs
        }
        },
        "Brier": {
            "test": cal_test,
            "params": {
                "n_resamples": n_resamples,
                "n_classes": n_classes,
                "n_members": n_members,
                "bw": 0.1, # TODO: check this
                "obj": brier_obj,
                "obj_lambda": brier_obj_lambda,
                "loss": BrierLoss(), # changed!!
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "hidden_layers": hidden_layers,
                "hidden_dim": hidden_dim,
                "device": device,
                **kwargs
            }

        },
        "LP": {
            "test": cal_test,
            "params": {
                "n_resamples": n_resamples,
                "n_classes": n_classes,
                "n_members": n_members,
                "obj": ece_kde_obj,
                "obj_lambda": ece_kde_obj_lambda,
                "bw": 0.00001, # TODO: check this
                "loss": LpLoss(bw=0.0001, lambda_bce=reg_lp), # changed
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "hidden_layers": hidden_layers,
                "hidden_dim": hidden_dim,
                "p": 2,
                "lambda_bce": reg_lp, # TODO: check this
                "device": device,
                **kwargs
            }
        },
        "SKCE": {
            "test": cal_test,
            "params": {
                "n_resamples": n_resamples,
                "n_classes": n_classes,
                "n_members": n_members,
                "obj": skce_obj,
                "obj_lambda": skce_obj_lambda,
                "bw": 0.001, # TODO: check this
                "loss": SKCELoss(bw=0.0001, lambda_bce=reg_skce), # changed!!
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "patience": patience,
                "hidden_layers": hidden_layers,
                "hidden_dim": hidden_dim,
                "lambda_bce": reg_skce, # TODO: check this
                "device": device,
                **kwargs
            }
        }
    }
    return config

"""
config for multiclass experiment with mlp optimization
"""
config_dirichlet_mlp = create_config(
    exp_name="dirichlet",
    cal_test=npbe_test_ensemble,
    optim="mlp",
    n_samples=1000,
    n_resamples=100,
    n_classes=5,
    n_members=5,
    n_epochs=300,
    lr=1e-4,
    batch_size=256,
    patience=100,
    hidden_layers=3,
    hidden_dim=64,
    x_dep=True,
    deg=2,
    x_bound = [0.0, 5.0]

)


"""
config for multiclass experiment with COBYLA optimization
"""
config_dirichlet_cobyla = create_config(
    exp_name="dirichlet",
    cal_test=npbe_test_ensemble,
    optim="COBYLA",
    n_samples=1000,
    n_resamples=100,
    n_classes=5,
    n_members=5,
    hidden_layers=3,
    hidden_dim=64,
    x_dep=True,
    deg=2,
    x_bound = [0.0, 5.0]

)

"""
configurations for binary classification with mlp optimization
"""
config_binary_classification_mlp = create_config(
    exp_name="gp",
    cal_test=npbe_test_ensemble,
    optim="mlp",
    n_samples=1000,
    n_resamples=100,
    n_classes=2,
    n_members=2,
    n_epochs=300,
    lr=1e-4,
    batch_size=256,
    patience=100,
    hidden_layers=3,
    hidden_dim=64,
    x_dep=True,
    deg=2,
    bounds_p=[[0.5, 0.8], [0.7, .9]],
)

"""
configurations for binary classification with COBYLA optimization
"""
config_binary_classification_cobyla = create_config(
    exp_name="gp",
    cal_test=npbe_test_ensemble,
    optim="COBYLA",
    n_samples=1000,
    n_resamples=100,
    n_classes=2,
    n_members=2,
    hidden_layers=3,
    hidden_dim=64,
    x_dep=True,
    deg=2,
    bounds_p=[[0.5, 0.8], [0.7, .9]],
)


# config_binary_classification_mlp = {
#     "Brier": create_config_mlp(
#         exp_name="gp",
#         cal_test=npbe_test_ensemble,
#         loss=BrierLoss,
#         bw=0.01,
#         n_samples=2000,
#         n_resamples=100,
#         obj=brier_obj,
#         n_epochs=300,
#         lr=1e-4,
#         batch_size=256,
#         patience=100,
#         hidden_layers=3,
#         hidden_dim=64,
#         x_dep=True,
#         deg=2,
#         bounds_p=[[0.5, 0.8], [0.7, .9]],
#     ),
#     "LP": create_config_mlp(
#         exp_name="gp",
#         cal_test=npbe_test_ensemble,
#         loss=LpLoss,
#         bw=0.002,
#         n_samples=2000,
#         n_resamples=100,
#         obj=ece_kde_obj,
#         n_epochs=300,
#         lr=1e-4,
#         batch_size=512,
#         patience=100,
#         hidden_layers=3,
#         hidden_dim=64,
#         x_dep=True,
#         deg=2,
#         bounds_p=[[0.5, 0.8], [0.7, .9]],
#         p=2,
#         lambda_bce=0.5,  # TODO: check this
#     ),
#     "SKCE": create_config_mlp(
#         exp_name="gp",
#         cal_test=npbe_test_ensemble,
#         loss=SKCELoss,
#         bw=0.05,
#         n_samples=2000,
#         n_resamples=100,
#         obj=skce_obj,
#         n_epochs=400,
#         lr=1e-4,
#         batch_size=512,
#         patience=300,
#         hidden_layers=3,
#         hidden_dim=64,
#         x_dep=True,
#         deg=2,
#         bounds_p=[[0.5, 0.8], [0.7, .9]],
#         lambda_bce=0.01,  
#     ),
# }


# config_binary_const_weights = {
#     "Brier": {
#         "experiment": "gp",
#         "test": npbe_test_ensemble,
#         "params": {
#             "optim": "COBYLA",
#             "n_samples": 2000,
#             "n_resamples": 100,
#             "obj": brier_obj,
#             "obj_lambda": brier_obj_lambda,
#             "x_dep": False,
#             "deg": 2,
#             "bounds_p": [[0.5, 0.8], [0.7, .9]],
#             "p": 2,
#         }
#     },
#     "LP": {
#         "experiment": "gp",
#         "test": npbe_test_ensemble,
#         "params": {
#             "optim": "COBYLA",
#             "n_samples": 2000,
#             "n_resamples": 100,
#             "obj": ece_kde_obj,
#             "obj_lambda": ece_kde_obj_lambda,
#             "bw": 0.002,
#             "x_dep": False,
#             "deg": 2,
#             "bounds_p": [[0.5, 0.8], [0.7, .9]],
#             "p": 2,
#         },
#     },
#     "SKCE": {
#         "experiment": "gp",
#         "test": npbe_test_ensemble,
#         "params": {
#             "optim": "COBYLA",
#             "n_samples": 2000,
#             "n_resamples": 100,
#             "obj": skce_obj,
#             "obj_lambda": skce_obj_lambda,
#             "bw": 0.1,
#             "x_dep": False,
#             "deg": 2,
#             "bounds_p": [[0.5, 0.8], [0.7, .9]],
#         },
#     },
#     "MMD": {
#         "experiment": "gp",
#         "test": npbe_test_ensemble,
#         "params": {
#             "optim": "COBYLA",
#             "n_samples": 2000,
#             "n_resamples": 100,
#             "obj": mmd_kce_obj,
#             "obj_lambda": mmd_kce_obj_lambda,
#             "bw": 0.1,
#             "x_dep": False,
#             "deg": 2,
#             "bounds_p": [[0.5, 0.8], [0.7, .9]],
#         },

#     },
# }

"""
configurations for multiclass classification
"""
 

# config_dirichlet_mlp = {
#     "Brier": create_config(
#         exp_name="dirichlet",
#         cal_test=npbe_test_ensemble,
#         loss=BrierLoss,
#         n_samples=1000,
#         n_classes=3,
#         n_members=5,
#         n_resamples=100,
#         obj=brier_obj,
#         n_epochs=300,
#         lr=1e-4,
#         batch_size=256,
#         patience=100,
#         hidden_layers=3,
#         hidden_dim=64,
#         x_dep=True,
#         deg=2,
#         x_bound = [0.0, 5.0]),

#     "LP": create_config(
#         exp_name="dirichlet",
#         cal_test=npbe_test_ensemble,
#         loss=LpLoss,
#         bw=0.002,
#         n_samples=1000,
#         n_classes=3,
#         n_members=5,
#         n_resamples=100,
#         obj=ece_kde_obj,
#         n_epochs=300,
#         lr=1e-4,
#         batch_size=512,
#         patience=100,
#         hidden_layers=3,
#         hidden_dim=64,
#         x_dep=True,
#         deg=2,
#         x_bound = [0.0, 5.0],
#         p=2,
#         lambda_bce=0.5,  
#     ),
#     "SKCE": create_config(
#         exp_name="dirichlet",
#         cal_test=npbe_test_ensemble,
#         loss=SKCELoss,
#         bw=0.05,
#         n_samples=1000,
#         n_classes=3,
#         n_members=5,
#         n_resamples=100,
#         obj=skce_obj,
#         n_epochs=400,
#         lr=1e-4,
#         batch_size=512,
#         patience=300,
#         hidden_layers=3,
#         hidden_dim=64,
#         x_dep=True,
#         deg=2,
#         x_bound = [0.0, 5.0],
#         lambda_bce=0.01,  
#     ),
    # "MMD": create_config_mlp(
    #     exp_name="dirichlet",
    #     cal_test=npbe_test_ensemble,
    #     loss=MMDLoss,
    #     bw=0.1,
    #     n_samples=1000,
    #     n_classes=3,
    #     n_members=5,
    #     n_resamples=100,
    #     obj=mmd_kce_obj,
    #     n_epochs=400,
    #     lr=1e-4,
    #     batch_size=512,
    #     patience=300,
    #     hidden_layers=3,
    #     hidden_dim=64,
    #     x_dep=True,
    #     deg=2,
    #     x_bound = [0.0, 5.0],
    #     lambda_bce=0.01,  
#     # ),
# }


# config_dirichlet_const_weights = {
#     "Brier": {
#         "experiment": "dirichlet",
#         "test": npbe_test_ensemble,
#         "params": {
#             "optim": "COBYLA",
#             "n_samples": 1000,
#             "n_resamples": 100,
#             "n_classes": 3,
#             "n_members": 5,
#             "obj": brier_obj,
#             "obj_lambda": brier_obj_lambda,
#             "x_dep": False,
#             "deg": 2,
#             "x_bound": [0.0, 5.0],
#             "p": 2,
#     }
#     },
#     "LP": {
#         "experiment": "dirichlet",
#         "test": npbe_test_ensemble,
#         "params": {
#             "optim": "COBYLA",
#             "n_samples": 1000,
#             "n_resamples": 100,
#             "n_classes": 3,
#             "n_members": 5,
#             "obj": ece_kde_obj,
#             "obj_lambda": ece_kde_obj_lambda,
#             "bw": 0.002,
#             "x_dep": False,
#             "deg": 2,
#             "x_bound": [0.0, 5.0],
#             "p": 2,
#         },
#     },
#     "SKCE": {
#         "experiment": "dirichlet",
#         "test": npbe_test_ensemble,
#         "params": {
#             "optim": "COBYLA",
#             "n_samples": 1000,
#             "n_resamples": 100,
#             "n_classes": 3,
#             "n_members": 5,
#             "obj": skce_obj,
#             "obj_lambda": skce_obj_lambda,
#             "bw": 0.1,
#             "x_dep": False,
#             "deg": 2,
#             "x_bound": [0.0, 5.0],
#         },
#      }
#     "MMD": {
#         "experiment": "dirichlet",
#         "test": npbe_test_ensemble,
#         "params": {
#             "optim": "COBYLA",
#             "n_samples": 1000,
#             "n_resamples": 100,
#             "n_classes": 3,
#             "n_members": 5,
#             "obj": mmd_kce_obj,
#             "obj_lambda": mmd_kce_obj_lambda,
#             "bw": 0.1,
#             "x_dep": False,
#             "deg": 2,
#             "x_bound": [0.0, 5.0],
#         },

# #     },
# }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create configuration for calibration test")
    parser.add_argument("--exp_name", type=str, default="gp", help="name of the experiment")
    parser.add_argument("--cal_test", type=callable,default=npbe_test_ensemble,
                         help="calibration test")
    parser.add_argument("--optim", type=str, default="mlp", help="optimization method")
    parser.add_argument("--n_samples", type=int, default=2000, help="number of samples")
    parser.add_argument("--n_resamples", type=int, default=100, help="number of resamples")
    parser.add_argument("--n_classes", type=int, default=3, help="number of classes")
    parser.add_argument("--n_members", type=int, default=5, help="number of members")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--patience", type=int, default=15, help="patience")
    parser.add_argument("--hidden_layers", type=int, default=1, help="number of hidden layers")
    parser.add_argument("--hidden_dim", type=int, default=32, help="hidden dimension")
    parser.add_argument("--x_dep", type=bool, default=True, help="x_dep")
    parser.add_argument("--deg", type=int, default=2, help="degree")
    args = parser.parse_args()
    config = config = create_config(
        exp_name=args.exp_name,
        cal_test=args.cal_test,
        optim=args.optim,
        n_samples=args.n_samples,
        n_resamples=args.n_resamples,
        n_classes=args.n_classes,
        n_members=args.n_members,
        n_epochs=args.n_epochs,
        lr = args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        hidden_layers=args.hidden_layers,
        hidden_dim = args.hidden_dim,
        x_dep=args.x_dep,
        deg=args.deg,
    )
    #print(config)
    print(config["LP"])
