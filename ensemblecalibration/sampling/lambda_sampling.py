import numpy as np

from numpy import random
from scipy.stats import multinomial, dirichlet

def uniform_weight_sampling(P: np.ndarray):
    """function for sampling new convex combinations of the predictions using coefficinets
    sampled uniformly from the (K-1) simplex.

    Parameters
    ----------
    P : np.ndarray of shape(n_predictors, n_classes) or
     (n_samples, n_predictors, n_classes)
        matrix containing the ensemble predictors
    size : int
        number of samples to generate

    Returns
    -------
    np.ndarray of shape (size, n_classes)
        array containing samples
    """
    if P.ndim == 2:
        M, K = P.shape 
    elif P.ndim == 3:
        N, M, K = P.shape

    lambdas = random.dirichlet([1]*M, size=1)[0, :]

    preds_new = lambdas @ P

    return preds_new

def multinomial_label_sampling(probs: np.ndarray):
    """draws a sampüle y from the categorical distribution
    defined by a probaibility vector.

    Parameters
    ----------
    probs : np.ndarray
        probability vector that sums up to one

    Returns
    -------
    _type_
        _description_
    """

    draws = multinomial(1, probs).rvs(size=1)[0, :]
    y = np.argmax(draws)

    return y


if __name__ == "__main__":
    P = np.random.dirichlet([1]*10, size=(100, 5))
    p_2 = uniform_weight_sampling(P)
    y = np.apply_along_axis(multinomial_label_sampling, 1, p_2)
    print(y)