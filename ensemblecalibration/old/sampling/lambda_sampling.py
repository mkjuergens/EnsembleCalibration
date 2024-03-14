import time

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
    """draws a sample y from the categorical distribution
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
    try:
        draws = multinomial(1, probs).rvs(size=1)[0, :]
        y = np.argmax(draws)

    except ValueError as e:
        y = np.argmax(probs)

    return y

def sample_m(p):
    try:
        y = np.argmax(multinomial(1,p).rvs(size=1)[0,:])
    except ValueError as e:
        y = np.argmax(p)

    return y

def sample_l(P):
    # take convex combination of ensemble predictions
    l = dirichlet([1]*P.shape[1]).rvs(size=1)[0,:]
    P_bar = np.matmul(np.swapaxes(P,1,2),l)
        
    return P_bar



if __name__ == "__main__":
    P = np.random.dirichlet([1]*3, size=(100, 10))
    print(P.shape)
    t_0 = time.time()
    p_2 = uniform_weight_sampling(P)
    t_1 = time.time()
    print(f'Time for sampling 1000 times: {t_1-t_0}')
    print(p_2.shape)
    y = np.apply_along_axis(multinomial_label_sampling, 1, p_2)
    P_bar_b = sample_l(P)
    print(P_bar_b.shape)
    y_b = np.apply_along_axis(sample_m, 1, P_bar_b)
    print(y_b.shape)

