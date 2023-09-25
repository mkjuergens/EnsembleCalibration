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


def sample_binary_predictions_beta(
    n_samples: int,
    n_ens: int,
    scale_factor: int = 2,
    lower: int = 0,
    upper: int = 1,
    ens_dep: bool = False,
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
    ens_dep : bool, optional
        whether the parameters of the sampling distributions should depend on the ensemble members, by default False
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
            if ens_dep:
                p_probs[i, ens, 0] = np.random.beta(
                    a=x_inst[i, 0], b=x_inst[i, 0] * (ens + 1)
                )
            else:
                p_probs[i, ens, 0] = np.random.beta(
                    a=x_inst[i, 0], b=x_inst[i, 0] * scale_factor
                )
            # set second column to 1 - first column
            p_probs[i, ens, 1] = 1 - p_probs[i, ens, 0]
    return x_inst, p_probs


def generate_weights_ens_dep(x_inst: np.ndarray, p_probs: np.ndarray, deg: int = 1):
    """"""
    assert (
        p_probs.shape[0] == x_inst.shape[0]
    ), "Number of samples should be equal to number of predictions"

    n_samples, n_ens, n_classes = p_probs.shape
    weights_lambda = np.zeros((n_samples, n_ens))
    for i in range(n_samples):
        for j in range(n_ens):
            weights_lambda[i, j] = ((j + 1) * x_inst[i] ** deg) / n_ens

    # use softmax
    softmax = torch.nn.Softmax(dim=1)
    weights = softmax(torch.tensor(weights_lambda)).numpy()
    return weights


def generate_fct_dep_weights(x_inst: np.ndarray, p_probs: np.ndarray, deg: int = 1):
    """function for generating weights for the binary case. The weights are generated based on the
    instance values. The weights are generated as follows: for each instance, the weights of the
    ensemble members are set to
        ((INSTACE_VALUE)**2)/(N_ENSEMBLE)
    for odd indices and


        for even indices and to 1 - instance value for
    odd indices. The weights are then normalized to sum to 1.

    Parameters
    ----------
    x_inst : np.ndarray
        array of shape (n_samples, 1) containing the sampled instances
    p_probs : np.ndarray
        array of shape (n_samples, n_ens, 2) containing the probabilities of the first and second class
    deg: int, optional
        degree of the function used to generate the weights, by default 1. Has to be a positive integer
    Returns
    -------
    weights : np.ndarray
        array of shape (n_samples, n_ens) containing the weights
    """
    # check dimension of p_probs
    assert (
        len(p_probs.shape) == 3
    ), "p_probs should have shape (n_samples, n_ens, n_classes)"

    assert deg > 0, "degree should be a positive integer"

    n_samples, n_ens, n_classes = p_probs.shape
    # initialize weights
    weights = np.zeros((n_samples, n_ens))
    # make weights instance dependent
    for i in range(n_samples):
        for j in range(n_ens):
            # let weights increase linearly with value of instance for even indices, decrease linearly for odd
            # indices
            if j % 2 == 0:
                # weights[i, j] = (x_inst[i] ** deg) / n_ens
                weights[i, j] = x_inst[i] ** deg
            else:
                # weights[i, j] = (
                #   1 - (np.ceil(n_ens / (2)) * x_inst[i] ** deg) / n_ens
                #  ) / np.floor(n_ens / 2)
                weights[i, j] = 1 - x_inst[i] ** deg
                # use this to make sure that the weights sum to 1
    # Normalize the weights to sum to 1
    softmax = torch.nn.Softmax(dim=1)
    # dont use softmax in the end to keep linear/quadratic/.. relationship ??

    weights = softmax(torch.tensor(weights)).numpy()

    # assert that all weights sum up to 1
    assert np.sum(weights, axis=1).all() == 1, "weights do not sum to 1"
    # assert that values of weights lie in range [0, 1]
    assert (weights >= 0).all(), "weights should be positive"
    assert (weights <= 1).all(), "weights should be smaller or equal to 1"

    return weights


def experiment_binary_nn(
    n_samples: int,
    n_ens: int,
    ens_dep: bool = False,
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
        number of instances
    n_ens : int
        number of ensemble members
    ens_dep : bool, optional
        whether the ensemble predictions are dependent on the ensemble member, by default False
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

    """
    # sample instances and probabilities
    x_inst, p_probs = sample_binary_predictions_beta(
        n_samples=n_samples,
        n_ens=n_ens,
        lower=x_lower,
        upper=x_upper,
        scale_factor=scale_factor,
        ens_dep=ens_dep,
    )
    # generate weights
    weights = generate_fct_dep_weights(x_inst=x_inst, p_probs=p_probs, deg=deg)
    # calculate weighted average
    p_bar = calculate_pbar(weights, p_probs, reshape=True, n_dims=2)
    # sample labels from categorical distribution
    y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_bar)

    # split into train and test set

    return x_inst, p_probs, weights, p_bar, y_labels


def generate_linear_fct_ivl(
    x: np.ndarray, lower: float = 0.0, upper: float = 1.0):
    """generates values of a random linear function evaluated on the respective x-values.
    Its values all lie in the specified interval.
    Parameters
    ----------
    x : np.ndarray
        array of shape (n_instances, n_features) containing the instance values
    lower : float, optional
        lower bound of the interval, by default 0.0
    upper : float, optional
        upper bound of the interval, by default 1.0

    Returns
    -------

    fct_vals:    np.ndarray of shape {n_instances, } or (n_instances, n_preds)
            function values evaluated on the instances
    [slope, intercept]: list of length 2
            list containing the slope and intercept of the linear function
    """
    # sample intercept with y axis ranomly in interval
    intercept = np.random.rand(1) * (upper - lower) + lower
    # sample slope dependet on intercept so that he fct values lie in the interval
    slope = np.random.rand(1) * (upper - lower) - (intercept - lower)

    fct_vals = slope * x[:, 0] + intercept

    return fct_vals, [slope, intercept]


def gen_polynomial_predictions_ivl_binary(
    x_inst: np.ndarray,
    a_lower: float = 0.0,
    b_upper: float = 1.0,
    deg_fct: int = 1,
    n_preds: int = 1,
):
    """generates values of a random polynomial function evaluated on the given instances.
    Its values all lie in the specified interval [a_lower, b_upper].

    Parameters
    ----------
    x_inst : np.ndarray
        vector of instance values of shape (n_instances, n_features)
    a_lower : float, optional
        lower bound of the interval, by default 0.0
    b_upper : float, optional
        upper bound of the interval, by default 1.0
    deg_fct : int, optional
        degree of the polynomial, by default 1
    n_preds: int, optional
        number of predictions (different predictors), ny default 1

    Returns
    -------
    np.ndarray of shape (n_instances,) or (n_instances, n_preds) if n_preds > 1
        vector of the function values
    """

    assert (
        x_inst.ndim == 2
    ), "instance vector should be of shpe (n_instances, n_features)"

    if deg_fct == 1:
        p_prob = generate_linear_fct_ivl(x_inst, a_lower, b_upper, n_preds=n_preds)
    else:
        # define the polynomial fct
        def f(x, coefs):
            return sum(coefs[k] * (x**k) for k in range(deg_fct + 1))

        # generate random coefficients for the polynomial
        coefs = np.random.uniform(-1, 1, deg_fct + 1)

        p_prob = f(x_inst[:, 0], coefs=coefs)
        # find the existing range of values
        min_val, max_val = min(p_prob), max(p_prob)
        # scale, translate
        p_prob = a_lower + (b_upper - a_lower) * (p_prob - min_val) / (
            max_val - min_val
        )
    

    return p_prob


def binary_experiment_cone_h0(
    n_samples: int,
    n_ens: int = 2,
    x_lower: float = 0,
    x_upper: float = 1,
    deg_fct: int = 0,
    w_1: float = 0.5,
):
    """experimental setting for a binary case and two ensemble members, where
    the predictions of the two ensemble members form a cone like shape in dependence of the instance values.
    The true calibrted predictor is a convex combination of the two ensemble members.

    Parameters
    ----------
    n_samples : int
        number of instances
    n_ens : int, optional
        number of ensemble members, by default 2
    x_lower : int, optional
        lower bound of instance values, by default 0
    x_upper : int, optional
        upper bound of instacne values, by default 1
    deg_fct : int, optional
        degree of the polynomial function which defines the weight per instance , by default 1
    w_1 : float, optional
        value of the weight for the first ensemble member (only used in the constant case), by default 0.5

    Returns
    -------
    x_inst : np.ndarray
        array of shape (n_samples, 1) containing the sampled instances
    p_probs : np.ndarray
        array of shape (n_samples, n_ens, 2) containing the probabilities of the first and second class
    weights : np.ndarray
        array of shape (n_samples, n_ens) containing the weights
    p_bar : np.ndarray
        array of shape (n_samples, 2) containing the probabilities of the first and second class of the calibrated predictor
    y_labels : np.ndarray
        array of shape (n_samples,) containing the labels
    """
    x_inst = sample_uniform_instances(
        n_samples=n_samples, x_lower=x_lower, x_upper=x_upper, n_dim=1
    )

    # sample two ensemble predictors whose predictions in dependence of instance values form a cone like shape
    p_1 = np.zeros((n_samples, 2))
    p_2 = np.zeros((n_samples, 2))

    p_1[:, 0] = 0.5 * x_inst[:, 0]
    p_1[:, 1] = 1 - p_1[:, 0]
    p_2[:, 0] = 1 - 0.5 * x_inst[:, 0]
    p_2[:, 1] = 1 - p_2[:, 0]

    # save in one array
    p_probs = np.stack((p_1, p_2), axis=1)

    if deg_fct == 0:
        # use constant weight for all instances
        p_bar = w_1 * p_1 + (1 - w_1) * p_2
        weights = np.zeros((n_samples, n_ens))
        weights[:, 0] = w_1
        weights[:, 1] = 1 - w_1
    else:
        # generate instance dependent weights
        weights = generate_fct_dep_weights(x_inst=x_inst, p_probs=p_probs, deg=deg_fct)
        # calculate convex combination
        p_bar = calculate_pbar(weights, p_probs, reshape=True, n_dims=2)

    # sample labels from categorical distribution induced by p_bar
    y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_bar)

    return x_inst, p_probs, weights, p_bar, y_labels


def binary_experiment_cone_h1(
    n_samples: int,
    n_ens: int = 2,
    x_lower: float = 0.0,
    x_upper: float = 1.0,
    weight_l: float = 0.5,
):
    """toy experiment which uses predictions of two ensemble members which form a cone like shape

    Parameters
    ----------
    n_samples : int
        _description_
    n_ens: int
        number of ensemble members, by default 2
    x_lower : float, optional
        _description_, by default 0.0
    x_upper : float, optional
        _description_, by default 1.0

    Returns
    -------
    _type_
        _description_
    """

    x_inst = sample_uniform_instances(
        n_samples=n_samples, x_lower=x_lower, x_upper=x_upper, n_dim=1
    )

    # sample two ensemble predictors whose predictions in dependence of instance values form a cone like shape
    p_1 = np.zeros((n_samples, 2))
    p_2 = np.zeros((n_samples, 2))

    p_1[:, 0] = 0.5 * x_inst[:, 0]
    p_1[:, 1] = 1 - p_1[:, 0]
    p_2[:, 0] = 1 - p_1[:, 0]
    p_2[:, 1] = 1 - p_2[:, 0]

    # save in one array
    p_probs = np.stack((p_1, p_2), axis=1)

    # set true p_bar to be outside cone
    p_bar = np.zeros((n_samples, 2))
    p_bar = weight_l * p_1[:, 0] - (1 - weight_l) * p_2[:, 0]

    # sampkle labels from induced multinomial distribution
    y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_bar)

    return x_inst, p_probs, weights, p_bar, y_labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_inst, p_probs, weights, p_bar, y_labels = binary_experiment_cone_h0(
        n_samples=1000, n_ens=2, w_1=0.75
    )
    plt.plot(x_inst.squeeze(), p_bar[:, 0])
    plt.plot(x_inst.squeeze(), p_probs[:, 0, 0], label="$\hat{p}^{(1)}$")
    plt.plot(x_inst.squeeze(), p_probs[:, 1, 0], label="$\hat{p}^{(2)}$")
    plt.show()
