from typing import Optional
import numpy as np
from scipy.stats import multinomial

from ensemblecalibration.calibration.iscalibrated import is_calibrated
from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar
from ensemblecalibration.sampling import multinomial_label_sampling


def get_ens_alpha(K: int, u: float, a0: np.ndarray):
    """yields the mean vector of the sampled dataset using a dirichlet distribution

    Parameters
    ----------
    K : int
        number of classes
    u : float
        parameter which describes the "spread" of the generated distribution within the simplex
    a0 :
        _description_

    Returns
    -------
    np.ndarray
        vector of size (K,) defining the parameterof the dirichlet distribution where
          the dataset is sampled from
    """
    p0 = np.random.dirichlet(a0, 1)[0, :]
    return (K * p0) / u


def getBoundary(p_probs, p_mu, p_c):
    """function for getting  the boundary of the convex hull of a given set of predictors.

    Parameters
    ----------
    P : np.ndarray
        tensor containing probabilistic predictions for each instance of each ensemble member
    p_mu : np.ndarray
        mean of the ensemble predictors
    p_c : np.ndarray
        randomly chosen corner

    Returns
    -------
    np.ndarray
        _description_
    """
    l_arr = np.linspace(0, 1, 100)
    # get convex combinations between yc and mu
    L = np.stack([l * p_c + (1 - l) * p_mu for l in l_arr])
    # determine boundary
    bi = 0
    for i in range(len(L)):
        # stop if boundary is reached
        if not is_calibrated(p_probs, L[i, :]):
            bi = i - 1
            break
    yb = L[bi, :]

    return yb


def experiment_h0_feature_dependency(
    N: int, M: int, K: int, u: float, return_optim_weight: bool = False
):
    """yields the predictive value tensor as well as the labels for the experiment in Mortier
    et al, where the null hypothesis that the ensemble model is calibrated is true.

    Parameters
    ----------
    N : int
        _description_
    M : int
        _description_
    K : int
        _description_
    R : int
        _description_
    u : float
        _description_

    return_optim_weight: bool

    Returns
    -------
    _type_
        _description_
    """

    P, y, L = [], [], []
    for n in range(N):
        # sample weight vector
        l = np.random.dirichlet([1 / M] * M, 1)[0, :]
        # repeat sampled weight vector K times, save in matrix of shape (M, K)
        l_n = np.repeat(l.reshape(-1, 1), K, axis=1)
        # sample parameter of dirichlet distribution
        a = get_ens_alpha(K, u, [1 / K] * K)
        while np.any(a <= 0):
            a = get_ens_alpha(K, u, [1 / K] * K)
        # sample probability
        Pm = np.random.dirichlet(a, M)
        Pbar = np.sum(Pm * l_n, axis=0)
        # sample instance
        try:
            # sample labels from the categorical distribution defined over the randomly sampled convex comb
            yl = np.argmax(multinomial(1, Pbar).rvs(size=1), axis=1)[0]
        except ValueError as e:
            yl = np.argmax(Pbar)
        P.append(Pm)
        y.append(yl)
        L.append(l)
    P = np.stack(P)
    y = np.array(y)
    L = np.stack(L)

    if return_optim_weight:
        return P, y, L
    return P, y


def experiment_h1_feature_dependecy(
    N: int, M: int, K: int, u: float, random: bool = False
):
    """returns P tensor and array of labels for the setting in Mortier et al where the null
    hypothesis is false

    Parameters
    ----------
    N : int
        number of instances
    M : int
        number of predictors
    K : int
        number of different classes
    R : int
        _description_
    u : float
        parameter of the dirichlet distribution describing the "spread" or "uncertainty"
    random : bool, optional
        whether the corner is randomly chosen or the closest corner is chosen, by default False_p
    Returns
    -------
    _type_
        _description_
    """

    P, y = [], []
    for n in range(N):
        a = get_ens_alpha(K, u, [1 / K] * K)
        while np.any(a <= 0):
            a = get_ens_alpha(K, u, [1 / K] * K)
        mu = (a * u) / K
        if M == 1:
            Pm = mu.reshape(1, -1)
        else:
            Pm = np.random.dirichlet(a, M)
        # pick class and sample ground-truth outside credal set
        if not random:
            c = np.argmax(mu)
        else:
            c = np.random.randint(0, K, 1)[0]
        yc = np.eye(K)[c, :]
        # get boundary
        if M == 1:
            yb = mu
        else:
            yb = getBoundary(Pm, mu, yc)
        # get random convex combination
        l = np.random.rand(1)[0]
        l = l * yc + (1 - l) * yb
        # sample instance
        try:
            yl = np.argmax(multinomial(1, l).rvs(size=1), axis=1)[0]
        except ValueError as e:
            yl = np.argmax(l)
        P.append(Pm)
        y.append(yl)
    P = np.stack(P)
    y = np.array(y)

    return P, y


def new_experiment_h0(n_instances: int, n_ensembles: int, n_classes: int,
                        uct: float = 0.01, l_prior: Optional[np.ndarray] = None, 
                        feature_dependent: bool = False):
    """new setting for experiment where the null hypothesis is true. Here, an initial prior mean is
    sampled from a Dirichlet distribution for each ensemble member, then the predictions are
    sampled using a Dirichlet distribution defined over the respective prior.

    Parameters
    ----------
    n_instances : int
        number of instances
    n_ensembles : int
        number of ensemble members
    n_classes : int
        number of classes
    uct : float, optional
        parameter used to control the "spread" of the dsitribution of the ensemble prior means, by default 0.01
    l_prior : Optional[np.ndarray], optional
        _description_, by default None
    feature_dependent : bool, optional
        whether the convex combination is feature dependent or not, by default False

    Returns
    -------
    p_probs, y_labels, l_weights
        tensor of probabilistic predictions, array of labels and array of weights
    """


    # sample weight vector for (random convex combination)
    if l_prior is None:
        if feature_dependent:
            l_weights = np.random.dirichlet([1/n_ensembles]* n_ensembles, n_instances)
        else:
            l_weights = np.random.dirichlet([1]*n_ensembles)
    else:
        if feature_dependent:
            l_weights = np.random.dirichlet(l_prior, n_instances)
        else:
            l_weights = np.random.dirichlet(l_prior)


    p_probs = []
    for ens in range(n_ensembles):
        # sample prior mean
        mu = np.random.dirichlet([1/n_classes]* n_classes)
        p_prior = (n_classes*mu)/uct
        p_m = np.random.dirichlet(p_prior, n_instances)
        p_probs.append(p_m)
    
    p_probs = np.stack(p_probs, axis=1)

    # set one convex combination to be the calibrated one
    if feature_dependent:
        p_bar = calculate_pbar(l_weights, p_probs, reshape=True, n_dims=2)
    else:
        p_bar = calculate_pbar(l_weights, p_probs, reshape=False, n_dims=1)
    # sample labels from categorical distribution
    y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_bar)

    return p_probs, y_labels, l_weights


def experiment_h0(
    N: int, M: int, K: int, u: float, l_prior: int = 1, output_weights: bool = False
):
    """yields the predictive value tensor as well as the labels for the experiment in Mortier
    et al, where the null hypothesis that the ensemble model is calibrated is true.

    Parameters
    ----------
    N : int
        _description_
    M : int
        _description_
    K : int
        _description_
    R : int
        _description_
    u : float
        _description_
    l_prior: int
        parameter of the dirichlet distribution
    output_weights: bool
        whether to output the weights of the ensemble model used for the "truly"
        calibrated convex combination

    Returns
    -------
    if output_weights:
        P, y, l
    else:
        P, y
    """

    l = np.random.dirichlet([1 / l_prior] * M, 1)[0, :]  # TODO: change back to 1/M ?
    # repeat sampled weight vector K times, save in matrix of shape (M, K)
    L = np.repeat(l.reshape(-1, 1), K, axis=1)
    P, y = [], []
    for n in range(N):
        # sample parameter of dirichlet distribution
        a = get_ens_alpha(K, u, [1 / K] * K)
        while np.any(a <= 0):
            a = get_ens_alpha
            (K, u, [1 / K] * K)
        # sample probability
        Pm = np.random.dirichlet(a, M)  # of shape (M, K)
        Pbar = np.sum(Pm * L, axis=0)  # of shape (K,)
        # sample instance
        try:
            # sample labels from the categorical distribution defined over the randomly sampled convex comb
            yl = np.argmax(multinomial(1, Pbar).rvs(size=1), axis=1)[0]
        except ValueError as e:
            yl = np.argmax(Pbar)
        P.append(Pm)
        y.append(yl)
    P = np.stack(P)
    y = np.array(y)

    if output_weights:
        return P, y, l
    else:
        return P, y
    

def experiment_h0_const_preds(N: int, K: int):
    """experiment with one (constant) probabilistic predictor who gives equal 
    probability to all classes.

    Parameters
    ----------
    N : int
        number of instances
    K : int
        number of classes

    Returns
    -------
    p_m, y
        matrix of shape (N, K) with the probabilities of the constant predictor
        and the labels of the instances
    """

    y = []
    p_m = np.zeros((N, K)) + 1 / K
    for n in range(N):

        y_n = np.argmax(multinomial(1, p_m[n, :]).rvs(size=1), axis=1)[0]
        y.append(y_n)
    y = np.array(y)

    return p_m, y


def experiment_h1(N: int, M: int, K: int, u: float, random: bool = False):
    """returns P tensor and array of labels for the setting in Mortier et al where the null
    hypothesis is false

    Parameters
    ----------
    N : int
        number of instances
    M : int
        number of predictors
    K : int
        number of different classes
    R : int
        _description_
    u : float
        parameter of the dirichlet distribution describing the "spread" or "uncertainty"
    random : bool, optional
        whether the corner is randomly chosen or the closest corner is chosen, by default False

    Returns
    -------
    _type_
        _description_
    """

    P, y = [], []
    for n in range(N):
        a = get_ens_alpha(K, u, [1 / K] * K)
        while np.any(a <= 0):
            a = get_ens_alpha(K, u, [1 / K] * K)
        mu = (a * u) / K
        if M == 1:
            Pm = mu.reshape(1, -1)
        else:
            Pm = np.random.dirichlet(a, M)
        # pick class and sample ground-truth outside credal set
        if not random:
            c = np.argmax(mu)
        else:
            c = np.random.randint(0, K, 1)[0]
        yc = np.eye(K)[c, :]
        # get boundary
        if M == 1:
            yb = mu
        else:
            yb = getBoundary(Pm, mu, yc)
        # get random convex combination
        l = np.random.rand(1)[0]
        l = l * yc + (1 - l) * yb
        # sample instance
        try:
            yl = np.argmax(multinomial(1, l).rvs(size=1), axis=1)[0]
        except ValueError as e:
            yl = np.argmax(l)
        P.append(Pm)
        y.append(yl)
    P = np.stack(P)
    y = np.array(y)

    return P, y


if __name__ == "__main__":
    P, y = experiment_h0_feature_dependency(100, 10, 3, 0.01)
    
    p_mu = np.random.dirichlet([1/3]*3)
    p_c = np.random.dirichlet([1/3]*3)

    y_b = getBoundary(P, p_mu, p_c)
    print(y_b)

    p_probs, y_labels, l_weights = new_experiment_h0(100, 10, 3, 0.01, feature_dependent=True)
    print(p_probs.shape)

