import numpy as np
from numpy.random import dirichlet

from uncertaintyestimation.calibration.iscalibrated import isCalibrated

def rejectance_sample(P, x_0, v: np.ndarray):
    """yields the next sample in acceptance-rejectance sampling given a matrix P which 
    contains all M predictors as rows.

    Parameters
    ----------
    P : np.ndarray of shape (M, K) with M number of predictive models, K number of different classes
        
    x_0 : np.ndarray of shape (K, 1) or (K,)
        starting point
    v : np.ndarray
        proposed point (within the simplex)

    Returns
    -------
    np.ndarray
        newly sampled point
    """

    if isCalibrated(P, v):
        x_new = v
    else:
        x_new =x_0

    return x_new

def rejectance_sampling(P: np,ndarray, x_0: np.ndarray, n_steps: int):
    """_summary_

    Parameters
    ----------
    P : np
        _description_
    ndarray : _type_
        _description_
    x_0 : np.ndarray
        _description_
    n_steps : int
        _description_

    Returns
    -------
    _type_
        _description_
    """

    M, K = P.shape
    samples = [x_0]
    curr_x = x_0
    for i in range(n_steps):

        v = dirichlet([1]*M, size=1) # sample one weight vector uniformly from simplex as proposal point
        x_new = rejectance_sample(P, curr_x, v)
        if x_new != curr_x: 
            samples.append(x_new)
            x_new = curr_x

    return samples



