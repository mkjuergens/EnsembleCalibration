import torch
import numpy as np

from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar
from ensemblecalibration.sampling import multinomial_label_sampling


def sample_uniform_instances(
    n_samples: int = 1000, x_lower: int = 0, x_upper: int = 100, n_dim: int = 1
):
    """Samples n_samples instances uniformly from the hypercube [x_lower, x_upper]^n_dim.

    Parameters
    ----------
    n_samples : int, optional
        number of instances to sample, by default 1000
    x_lower : int, optional
        lower bound of the hypercube, by default 0
    x_upper : int, optional
        upper bound of the hypercube, by default 100
    n_dim : int, optional
        dimension of the hypercube, by default 1

    Returns
    -------
    X : np.ndarray
        array of shape (n_samples, n_dim) containing the sampled instances
    """
    X = np.random.uniform(x_lower, x_upper, size=(n_samples, n_dim))
    return X


def sample_binary_predictions(
    n_samples: int, n_ens: int, scale_factor: int = 2, lower: int = 0, upper: int = 1
):
    """function which generates binary predictions for a number of instances which are sampled from
    a uniform distribution. The probability of the first class is sampled from a beta distribution
    with parameters a = x and b = x*scale_factor, where x is the sampled instance. The probability

    Parameters
    ----------
    n_samples : int
        number of instances to sample
    n_ens : int
        number of ensemble members
    scale_factor : int, optional
        scaling factor of the second parameter of the beta distribution, by default 2
    lower : int, optional
        lower bound of the hypercube of which the instances are sampled from, by default 0
    upper : int, optional
        upper bound of the hypercube of which the instances are sampled from, by default 1

    Returns
    -------
    x_inst : np.ndarray
        array of shape (n_samples, 1) containing the sampled instances
    p_probs : np.ndarray
        array of shape (n_samples, n_ens, 2) containing the probabilities of the first and second class
    """

    # sample instances
    x_inst = sample_uniform_instances(
        n_samples=n_samples, x_lower=lower, x_upper=upper, n_dim=1
    )

    p_probs = np.zeros((n_samples, n_ens, 2))
    for i in range(n_samples):
        for ens in range(n_ens):
            # sample probabilities
            p_probs[i, ens, 0] = np.random.beta(
                a=x_inst[i, 0], b=x_inst[i, 0] * scale_factor
            )
            # set second column to 1 - first column
            p_probs[i, ens, 1] = 1 - p_probs[i, ens, 0]
    return x_inst, p_probs


def generate_weights_binary(
    x_inst: np.ndarray, p_probs: np.ndarray, deg: int = 1
):
    """function for generating weights for the binary case. The weights are generated based on the
    instance values. The weights are generated as follows: for each instance, the weights of the
    ensemble members are set to the instance value for even indices and to 1 - instance value for
    odd indices. The weights are then normalized to sum to 1.

    Parameters
    ----------
    x_inst : np.ndarray
        array of shape (n_samples, 1) containing the sampled instances
    p_probs : np.ndarray
        array of shape (n_samples, n_ens, 2) containing the probabilities of the first and second class
    fct : str, optional
        function to generate the weights, by default "linear". Options: "linear", "quadratic", "const"
    Returns
    -------
    weights : np.ndarray
        array of shape (n_samples, n_ens) containing the weights
    """
    # check dimension iof p_probs
    assert (
        len(p_probs.shape) == 3
    ), "p_probs should have shape (n_samples, n_ens, n_classes)"

    n_samples, n_ens, n_classes = p_probs.shape
    # initialize weights
    weights = np.zeros((n_samples, n_ens))
    # make weights instance dependent
    for i in range(n_samples):
        for j in range(n_ens):
            # let weights increase linearly with value of instance for even indices, decrease linearly for odd
            # indices
            if j % 2 == 0:
                weights[i, j] = x_inst[i]**deg
            else:
                weights[i, j] = 1 - x_inst[i]**deg



            """
            if j % 2 == 0:
                if fct == "linear":
                    weights[i, j] = x_inst[i]
                elif fct == "quadratic":
                    weights[i, j] = x_inst[i] ** 2
                elif fct == "const":
                    weights[i, j] = j
            else:
                if fct == "linear":
                    weights[i, j] = 1 - x_inst[i]
                elif fct == "quadratic":
                    weights[i, j] = 1 - x_inst[i] ** 2
                elif fct == "const":
                    weights[i, j] = j
           """ 
    # Normalize the weights to sum to 1
    softmax = torch.nn.Softmax(dim=1)
    weights = softmax(torch.tensor(weights)).numpy()

    # check if weights sum to 1
    assert np.sum(weights, axis=1).all() == 1, "weights do not sum to 1"

    return weights


def experiment_binary_nn(
    n_samples: int,
    n_ens: int,
    x_lower: int = 0,
    x_upper: int = 1,
    scale_factor: int = 2,
    deg: int = 1,
):
    """function for running an experiment for the binary case. The experiment consists of sampling
    instances and probabilities, generating weights and calculating the weighted average. The
    weighted average is then used to sample labels from a multinomial distribution.

    Parameters
    ----------
    n_samples : int
        _description_
    n_ens : int
        _description_
    x_lower : int
        _description_
    x_upper : int
        _description_
    scale_factor : int, optional
        factor used , by default 2
    deg: int, optional
        degree of the function used to generate the weights, by default 1

    Returns
    -------
    x_inst : np.ndarray

    p_probs : np.ndarray

    weights : np.ndarray

    p_bar : np.ndarray

    y_labels : np.ndarray

        _description_
    """
    # sample instances and probabilities
    x_inst, p_probs = sample_binary_predictions(
        n_samples=n_samples,
        n_ens=n_ens,
        lower=x_lower,
        upper=x_upper,
        scale_factor=scale_factor,
    )
    # generate weights
    weights = generate_weights_binary(x_inst=x_inst, p_probs=p_probs, deg=deg)
    # calculate weighted average
    p_bar = calculate_pbar(weights, p_probs, reshape=True, n_dims=2)
    # sample labels from categorical distribution
    y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_bar)

    # split into train and test set

    return x_inst, p_probs, weights, p_bar, y_labels


if __name__ == "__main__":
    x_inst, p_probs, weights, p_bar, y_labels = experiment_binary_nn(
        n_samples=100, n_ens=2, x_lower=0, x_upper=1, deg=0
    )
    print(weights)
