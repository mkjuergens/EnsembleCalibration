import numpy as np
from scipy.stats import multinomial

from ensemblecalibration.calibration.iscalibrated import is_calibrated


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



def experiment_h0(N: int, M: int, K: int, u: float, l_prior: int = 1):
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

    Returns
    -------
    _type_
        _description_
    """

    l = np.random.dirichlet([1/l_prior] * M, 1)[0, :] # TODO: change back to 1/M ?
    # repeat sampled weight vector K times, save in matrix of shape (M, K)
    L = np.repeat(l.reshape(-1, 1), K, axis=1)
    P, y = [], []
    for n in range(N):
        # sample parameter of dirichlet distribution
        a = get_ens_alpha(K, u, [1 / K] * K)
        while np.any(a <= 0):
            a = get_ens_alpha(K, u, [1 / K] * K)
        # sample probability
        Pm = np.random.dirichlet(a, M) # of shape (M, K)
        Pbar = np.sum(Pm * L, axis=0) # of shape (K,)
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

    return P, y


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
    print(P)
    l_1 = np.random.dirichlet([1/10]*10,1)[0,:]
    print(l_1)
    l_2 = np.random.dirichlet([1]*10,1)[0, :]
    print(l_2)
