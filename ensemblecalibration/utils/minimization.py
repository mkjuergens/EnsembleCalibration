import numpy as np
from scipy.optimize import minimize
from ensemblecalibration.meta_model import get_optim_lambda_mlp
from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.utils.helpers import calculate_pbar


def calculate_min(
    x_inst: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray, params: dict
):
    """calculate the optimal convex combination of predictors using a given miscalibration estimate
    and minimization technique.

    Parameters
    ----------
    x_inst : np.ndarray
        array or tensor of instances
    p_probs : np.ndarray
        tensor of shape (n_samples, n_predictors, n_classes)
    y_labels : np.ndarray
        array of labels
    params : dict
        dictionary of parameters

    Returns
    -------
    float, np.ndarray
        minimal statistic and optimal weights

    Raises
    ------
    NotImplementedError
        if the optimization method is not implemented
    """
    #n_dims = 2 if params["x_dep"] else 1
    n_dims = 2 if params["optim"] == "mlp" else 1
    #print(f"n_dims: {n_dims}")
    if params["optim"] == "mlp":
        dataset = MLPDataset(x_train=x_inst, P=p_probs, y=y_labels)
        l_weights, loss = get_optim_lambda_mlp(
            dataset,
            loss=params["loss"],
            n_epochs=params["n_epochs"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            hidden_dim=params["hidden_dim"],
            hidden_layers=params["hidden_layers"],
            patience=params["patience"],
        )
    elif params["optim"] == "COBYLA":
        l_weights = minimize_const_weights(p_probs, y_labels, params)
    else:
        raise NotImplementedError("Only 'mlp' and 'COBYLA' are implemented.")
    p_bar = calculate_pbar(l_weights, p_probs, reshape=False, n_dims=n_dims)
    # calculate test statistic
    minstat = params["obj"](p_bar, y_labels, params)

    return minstat, l_weights

def minimize_const_weights(
    p_probs: np.ndarray, y: np.ndarray, params: dict, enhanced_output: bool = False
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
        params["obj_lambda"], l_0, (p_probs, y, params, False), method="COBYLA", constraints=cons
    )
    l = np.array(solution.x)
    minstat = params["obj_lambda"](l, p_probs, y, params, False)
    if enhanced_output:
        return l, minstat
    else:
        return l
    

def c1_constr(x):
    return np.sum(x)-1.0

def c2_constr(x):
    return -(np.sum(x)-1.0)