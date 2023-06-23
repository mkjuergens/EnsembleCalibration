import numpy as np
import torch
from torch import nn
from scipy.stats import entropy

from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar
from ensemblecalibration.sampling import multinomial_label_sampling

def weights_l_fct(p_probs: np.ndarray, fct_type: str = "entropy"):
    """function which calculates weights of the convex combination of ensemble predictions
    using a predefined function which is dependent of the class probabilities

    Parameters
    ----------
    p_probs : np.ndarray of shape (n_instances, n_members, n_classes)
        probabilistic predictions for each ensemble and each instance
    fct_type : str, optional
        type of the function that is used  , by default "entropy"

    Returns
    -------
    _type_
        _description_
    """

    if fct_type == "entropy":
        weights = np.apply_along_axis(entropy, 2, p_probs)
    else:
        raise NotImplementedError("Function not implemented")
    
    softmax = nn.Softmax(dim=1)
    weights = softmax(torch.from_numpy(weights).float())
    weights = weights.detach().numpy()
    return weights


def _binary_weights_fct(n_instances: int, fct: str = "linear", period: int = 100):
    """function for generating weights for binary classification, using a predefined function.

    Example: fct = "linear", period = 100
     then \lambda(x) = (\lambda_1(x), \lambda_2(x)) = (x // period, 2*x // period)

    Parameters
    ----------
    n_instances : int
        number of instances
    fct : str, optional
        type of function used, by default "linear". Options are "const", "linear", "sin", "quadratic"
    period : int, optional
        modulo value used in the process (see below) , by default 100

    Returns
    -------
    weights : np.ndarray
        vector of weights for each instance, of shape (n_instances, 2)
    """

    weights = np.zeros((n_instances, 2))
    if fct == "const":
        for i in range(n_instances):
            weights[i, 0] = 1/2
            weights[i, 1] = 1/2

    elif fct == "linear":
        for i in range(n_instances):
            weights[i, 0] = i // 100
            weights[i, 1] = (2*i) // 100

    elif fct == "sin":
        for i in range(n_instances):
            weights[i, 0] = np.sin(i / 100)
            weights[i, 1] = np.sin(2*i / 100)

    elif fct == "quadratic":
        for i in range(n_instances):
            weights[i, 0] = (i // 100)**2
            weights[i, 1] = ((2*i) // 100)**2

    else:
        raise NotImplementedError("Function not implemented")

    softmax = nn.Softmax(dim=1)
    weights = softmax(torch.from_numpy(weights).float())
    weights = weights.detach().numpy()

    return weights

def experiment_h0_nn(n_instances: int = 1000, n_classes: int = 2, n_ens: int = 2,
                      fct: str = "entropy", period_cycle: int = 100, uct: float = 0.01):
    
    # sample predictions from ensemble members
    p_probs = np.zeros((n_instances, n_ens, n_classes))
    # sample prior for different ensemble members only once:
    for ens in range(n_ens):
        # sample prior parameter
        p_prior = n_classes*(np.random.dirichlet([1/n_classes]*n_classes, 1)[0, :])/uct
        # sample predictions for each ensemble (form same prior parameters)
        p_probs[:, ens, :] = np.random.dirichlet(p_prior, n_instances)
    # weights
    weights = weights_l_fct(p_probs, fct_type=fct)
    # sample labels from convex combination
    p_bar = calculate_pbar(weights, p_probs, reshape=True, n_dims=2)

    y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_bar)

    return p_probs, y_labels, weights


def binary_experiment_nn(n_instances: int, n_classes: int, fct: str = "linear", period_cycle: int = 100,
                         uct: float = 0.01):
    weights = _binary_weights_fct(n_instances, fct=fct, period=period_cycle)
    p_probs = np.zeros((n_instances, 2, n_classes))
    for ens in range(2):
        # sample prior parameter
        p_prior = n_classes*(np.random.dirichlet([1/n_classes]*n_classes, 1)[0, :])/uct
        # sample predictions for each ensemble (form same prior parameters)
        p_probs[:, ens, :] = np.random.dirichlet(p_prior, n_instances)
    
    p_bar = calculate_pbar(weights, p_probs, reshape=True, n_dims=2)

    y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_bar)

    return p_probs, y_labels, weights



if __name__ == "__main__":
    p_probs = np.random.dirichlet([1/3, 1/3, 1/3], size=(100,10))
    weights = weights_l_fct(p_probs, fct_type="entropy")
    print(weights)
    print(p_probs.shape)
    p_probs_h0, y_labels, weights = experiment_h0_nn(n_instances=1000, n_classes=3, n_ens=10)
    print(p_probs_h0)