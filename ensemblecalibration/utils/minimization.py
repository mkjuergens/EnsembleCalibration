import numpy as np
import torch
from scipy.optimize import minimize
from ensemblecalibration.meta_model import get_optim_lambda_mlp, MLPCalW
from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.utils.helpers import (
    calculate_pbar,
    test_train_val_split,
    data_split,
)


def calculate_min(
    x_inst: torch.Tensor,
    p_probs: torch.Tensor,
    y_labels: torch.Tensor,
    params: dict,
    verbose: bool = False,
    val: bool = True,
    test: bool = True,
    output_p_preds: bool = False,
):
    """
    Calculate the optimal convex combination of predictors using a given miscalibration estimate
    and minimization technique.
    """
    device = params.get("device", "cpu")

    # Determine which splits to create
    split_test = test
    split_val = val

    # Split the data based on the flags
    data_splits = data_split(
        x_inst,
        p_probs,
        y_labels,
        test_size=0.2,
        val_size=0.2,
        split_test=split_test,
        split_val=split_val,
        random_state=42,
    )

    # Select datasets based on flags
    data_train = data_splits["train"]
    data_val = data_splits["val"] if val else data_splits["train"]
    data_test = data_splits["test"] if test else data_val

    n_dims = 2 if params["optim"] == "mlp" else 1

    if params["optim"] == "mlp":
        # Create datasets
        dataset_train = MLPDataset(x_train=data_train[0], P=data_train[2], y=data_train[1])
        dataset_val = MLPDataset(x_train=data_val[0], P=data_val[2], y=data_val[1])
        dataset_test = MLPDataset(x_train=data_test[0], P=data_test[2], y=data_test[1])

        # Initialize model
        model = MLPCalW(
            in_channels=data_train[0].shape[1],
            out_channels=data_train[2].shape[1],
            hidden_dim=params["hidden_dim"],
            hidden_layers=params["hidden_layers"],
            use_relu=True,
        ).to(device)

        # Find optimal weights using the MLP model
        l_weights, loss_train, loss_val = get_optim_lambda_mlp(
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            model=model,
            loss=params["loss"],
            n_epochs=params["n_epochs"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            patience=params["patience"],
            device=device,
            verbose=verbose,
        )
    elif params["optim"] in ["COBYLA", "SLSQP"]:
        l_weights = minimize_const_weights(data_test[2], data_test[1], params, method=params["optim"])
    else:
        raise NotImplementedError("Only 'mlp', 'COBYLA', and 'SLSQP' are implemented.")

    # Compute the convex combination
    p_bar = calculate_pbar(l_weights, data_test[2], reshape=False, n_dims=n_dims)

    # Calculate test statistic
    minstat = params["obj"](p_bar, data_test[1], params)

    if output_p_preds:
        return minstat, l_weights, p_bar, data_test[1], data_test[2]
    else:
        return minstat, l_weights, p_bar, data_test[1]


# def calculate_min(
#     x_inst: np.ndarray,
#     p_probs: np.ndarray,
#     y_labels: np.ndarray,
#     params: dict,
#     verbose: bool = False,
#     val: bool = True,
#     test: bool = True,
#     output_p_preds: bool = False,
# ):
#     """calculate the optimal convex combination of predictors using a given miscalibration estimate
#     and minimization technique.

#     Parameters
#     ----------
#     x_inst : np.ndarray
#         array or tensor of instances
#     p_probs : np.ndarray
#         tensor of shape (n_samples, n_predictors, n_classes)
#     y_labels : np.ndarray
#         array of labels
#     params : dict
#         dictionary of parameters
#     verbose : bool, optional
#         whether to print the results and loss, by default False
#     val: bool, optional
#         whether to use validation set for training, by default True
#     test: bool, optional
#         whether to use test set for the convex combination, by default False
#     output_p_preds: bool, optional

#     Returns
#     -------
#     minstat, l_weights, p_bar, y_labels
#         minstat: minimal statistic
#         l_weights: optimal weights
#         p_bar: convex combination of predictors on test set
#         y_labels: labels of test set

#     Raises
#     ------
#     NotImplementedError
#         if the optimization method is not implemented
#     """
#     # Determine which splits to create
#     split_test = test
#     split_val = val

#     # Split the data based on the flags
#     data_splits = data_split(
#         x_inst,
#         p_probs,
#         y_labels,
#         test_size=0.2,
#         val_size=0.2,
#         split_test=split_test,
#         split_val=split_val,
#         random_state=42,
#     )
#     # Select datasets based on flags
#     data_train = data_splits["train"]
#     data_val = data_splits["val"] if val else data_splits["train"]
#     data_test = data_splits["test"] if test else data_val
#     # n_dims = 2 if params["x_dep"] else 1
#     n_dims = 2 if params["optim"] == "mlp" else 1
#     # data_test, data_train, data_val = test_train_val_split(p_probs, y_labels, x_inst)
#     # print(len(data_train[0]))
#     # print(len(data_val[0]))
#     # print(len(data_test[0]))
#     if params["optim"] == "mlp":
#         # split data into train, validation and test (train and val are used to train the MLP)
#         dataset_train = MLPDataset(
#             x_train=data_train[0], P=data_train[2], y=data_train[1]
#         )
#         if val:
#             dataset_val = MLPDataset(x_train=data_val[0], P=data_val[2], y=data_val[1])
#         else:
#             dataset_val = dataset_train
#         if test:
#             dataset_test = MLPDataset(
#                 x_train=data_test[0], P=data_test[2], y=data_test[1]
#             )
#         else:
#             dataset_test = dataset_val
#         # intialise model
#         model = MLPCalW(
#             in_channels=data_train[0].shape[1],
#             out_channels=data_train[2].shape[1],
#             hidden_dim=params["hidden_dim"],
#             hidden_layers=params["hidden_layers"],
#             use_relu=True,
#         )
#         # the model outputs the optimal weights on the test set
#         l_weights, loss_train, loss_val = get_optim_lambda_mlp(
#             dataset_train=dataset_train,
#             dataset_val=dataset_val,
#             dataset_test=dataset_test,
#             model=model,
#             loss=params["loss"],
#             n_epochs=params["n_epochs"],
#             lr=params["lr"],
#             batch_size=params["batch_size"],
#             patience=params["patience"],
#             device=params["device"],
#             verbose=verbose,
#         )
#     elif params["optim"] == "COBYLA":
#         l_weights = minimize_const_weights(
#             data_test[2], data_test[1], params, method="COBYLA"
#         )
#     elif params["optim"] == "SLSQP":
#         l_weights = minimize_const_weights(
#             data_test[2], data_test[1], params, method="SLSQP"
#         )
#     else:
#         raise NotImplementedError("Only 'mlp', 'COBYLA' and 'SLSQP' are implemented.")
#     p_bar = calculate_pbar(l_weights, data_test[2], reshape=False, n_dims=n_dims)
#     # calculate test statistic
#     minstat = params["obj"](p_bar, data_test[1], params)

#     return (
#         minstat,
#         l_weights,
#         p_bar,
#         data_test[1],
#         data_test[2] if output_p_preds else None,
#     )  # TODO: check output_p_preds


def minimize_const_weights(
    p_probs: np.ndarray,
    y: np.ndarray,
    params: dict,
    enhanced_output: bool = False,
    method: str = "COBYLA",
):
    """returns the vector of weights which results in a convex combination of predictors with
        the minimal calibration error.
        Here, the weights do not depend on the instances, therefore resulting in a one-dimensional array.

    Parameters
    ----------
    p_probs : np.ndarray
        matrix of point predictions for each instance and predcitor
    y : np.ndarray
        labels
    params : dict
        dictionary of test parameters
    enhanced_output : bool, optional
        whether to return the minimal statistic, by default False
    method : str, optional
        optimization method, by default "COBYLA". Options: {"COBYLA", "SLSQP"}
    """

    # inittial guess: equal weights
    l_0 = np.array([1 / p_probs.shape[1]] * p_probs.shape[1])
    bnds = tuple([tuple([0, 1]) for _ in range(p_probs.shape[1])])
    cons = [{"type": "ineq", "fun": c1_constr}, {"type": "ineq", "fun": c2_constr}]
    # bounds must be included as constraints for COBYLA
    for factor in range(len(bnds)):
        lower, upper = bnds[factor]
        lo = {"type": "ineq", "fun": lambda x, lb=lower, i=factor: x[i] - lb}
        up = {"type": "ineq", "fun": lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(lo)
        cons.append(up)
    solution = minimize(
        params["obj_lambda"],
        l_0,
        (p_probs, y, params, False),
        method=method,
        constraints=cons,
    )
    l = np.array(solution.x)
    minstat = params["obj_lambda"](l, p_probs, y, params, False)
    if enhanced_output:
        return l, minstat
    else:
        return l


def c1_constr(x):
    return np.sum(x) - 1.0


def c2_constr(x):
    return -(np.sum(x) - 1.0)
