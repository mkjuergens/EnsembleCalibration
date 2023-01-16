import numpy as np

from numpy.random import dirichlet

def uniform_weight_sampling(P: np.ndarray, size: int):
    """function for sampling new convex combinations of the predictions using coefficinets
    sampled uniformly from the (K-1) simplex.

    Parameters
    ----------
    P : np.ndarray of shape(n_predictors, n_classes)
        matrix containing the ensemble predictors
    size : int
        number of samples to generate

    Returns
    -------
    np.ndarray of shape (size, n_classes)
        array containing samples
    """

    M, K = P.shape 
    lambdas = dirichlet([1]*M, size=size)

    preds_new = lambda @ P

    return preds_new