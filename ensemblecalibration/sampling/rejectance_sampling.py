import time

import numpy as np
from numpy.random import dirichlet

from ensemblecalibration.calibration.iscalibrated import is_calibrated
from ensemblecalibration.sampling.mcmc_sampling import find_inner_point

def rejectance_sample(P, x_0, v: np.ndarray):
    """yields the next sample in acceptance-rejectance sampling given a matrix P which 
    contains all M predictors as rows.

    Parameters
    ----------
    P : np.ndarray of shape (M, K) or (N, M, K) with M number of predictive models, K number of different classes
        
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

def rejectance_sampling(P: np.ndarray, n_samples: int):
    """_summary_

    Parameters
    ----------
    P : np.ndarray of shape (M, K)
        matrix containing the vectors who are spanning the polytope. Here, M is the number of (ensemble) predictors, K the number of classes.
    x_0 : np.ndarray of shape (K,)
        initial starting point for sampling
    n_steps : int
        number of samples

    Returns
    -------
    np.ndarray of shape (n_samples, K)
         of all the (accepted) samples
    """
    x_0 = find_inner_point(P)
    if P.ndim == 3:
        N, M, K = P.shape
    elif P.ndim == 2:
        M, K = P.shape
    samples = []
    curr_x = x_0
    count = 0
    while count < n_samples:

        v = dirichlet([1]* K, size=None) # sample one weight vector uniformly from simplex as proposal point; needs to be of shape (M,)
        x_new = rejectance_sample(P, curr_x, v)
        if not np.array_equal(x_new, curr_x): # check if vectors are the same, if not, append it to the list
            samples.append(x_new)
            x_new = curr_x
            count+=1

    x_out = np.stack(samples)
    return x_out

def rejectance_sampling_p(P: np.ndarray):
    """perform rejecatnce sampling for a tensor of shape (N, M, K). The output is

    Parameters
    ----------
    P : np.ndarray of shape (N, M, K)
        array containing M point rpedictions of length K for each sample x

    Returns
    -------
    np.ndarray
        array of samples
    """

    P_hat = np.zeros((P.shape[0], P.shape[2])) 
    for i in range(P.shape[0]):
        # sample one sample from mhar algorithm
        x_sample = rejectance_sampling(P[i], n_samples=1)
        P_hat[i] = x_sample

    return P_hat


if __name__ == "__main__":
    # test for K = 3, M = 4
    predicts = [(0,0,1), (0, 1/2, 1/2), (1,0,0), (0, 1, 0)] # point predictions of three predictive models
    # as array
    predicts_array = np.stack([np.asarray(predicts[i]) for i in range(len(predicts))])
    samples = rejectance_sampling(predicts_array, n_samples=10)
    print(samples.shape)

    t_0 = time.time()
    P = np.random.dirichlet([1]*3, size=(1000, 10))
    P_hat = rejectance_sampling_p(P)
    t_1 = time.time()
    print(np.sum(P_hat, axis=1))
    print(f'Time for one sampling: {t_1-t_0}')







