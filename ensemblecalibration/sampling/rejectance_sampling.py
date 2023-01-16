import numpy as np
from numpy.random import dirichlet

from ensemblecalibration.calibration.iscalibrated import is_calibrated

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


    if is_calibrated(P, v):
        x_new = v
    else:
        x_new =x_0


    return x_new

def rejectance_sampling(P: np.ndarray, x_0: np.ndarray, n_steps: int):
    """_summary_

    Parameters
    ----------
    P : np.ndarray of shape (M, K)
        matrix containing the vectors who are spanning the polytope. Here, M is the number of (ensemble) predictors, K the number of classes.
    x_0 : np.ndarray of shape (K,)
        initial statrting point for sampling
    n_steps : int
        number of steps in the sampling

    Returns
    -------
    list
        list of all the (accepted) samples
    """

    M, K = P.shape
    samples = [x_0]
    curr_x = x_0
    for i in range(n_steps):

        v = dirichlet([1]* K, size=None) # sample one weight vector uniformly from simplex as proposal point; needs to be of shape (M,)
        x_new = rejectance_sample(P, curr_x, v)
        if not np.array_equal(x_new, curr_x): # check if vectors are the same, if not, append it to the list
            samples.append(x_new)
            x_new = curr_x

    return samples

if __name__ == "__main__":
    # test for K = 3, M = 4
    predicts = [(0,0,1), (0, 1/2, 1/2), (1,0,0), (0, 1, 0)] # point predictions of three predictive models
    # as array
    predicts_array = np.stack([np.asarray(predicts[i]) for i in range(len(predicts))])
    x_0 = np.array([1, 0, 0])
    samples = rejectance_sampling(predicts_array, x_0, n_steps=10)
    print(np.concatenate(samples, axis=1))






