import time

import numpy as np
from numpy.random import dirichlet

from ensemblecalibration.calibration.iscalibrated import is_calibrated
from ensemblecalibration.sampling.mcmc_sampling import find_inner_point
from ensemblecalibration.calibration.helpers import init_pmatrix

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

def rejectance_sampling(P: np.ndarray, n_samples: int, enhanced_output: bool = False):
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
    (acc_rate: percentage of accepted samples of the whole run)
    """
    x_0 = find_inner_point(P)
    if P.ndim == 3:
        N, M, K = P.shape
    elif P.ndim == 2:
        M, K = P.shape
    samples = []
    curr_x = x_0
    count = 0
    count_all = 0
    while count < n_samples:

        v = dirichlet([1]* K, size=None) # sample one weight vector uniformly from simplex as proposal point; needs to be of shape (M,)
        #x_new = rejectance_sample(P, curr_x, v)
        x_new = rejectance_sample(P, x_0, v )
        if not np.array_equal(x_new, curr_x): # check if vectors are the same, if not, append it to the list
            samples.append(x_new)
            x_new = curr_x
            count+=1
            count_all += 1
           # print("accepted")
        else:
            count_all += 1
            

    acc_rate = count/count_all
    x_out = np.stack(samples)
    if enhanced_output:
        return x_out, acc_rate
    else:
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

    for k in range(3, 8):
        predicts = np.random.dirichlet([1]*k, size=k)
        print(predicts.shape)
        samples, rate = rejectance_sampling(predicts, n_samples = 10, enhanced_output=True)
        print(f'Acceptance rate for {k} classes: {rate}')
    # test for K = 3, M = 4
    predicts = [(.5,.5,0,0), (0,0,.5, .5), (.2, .5, 0, .3), (.3, .5, .2, 0)] # point predictions of three predictive models
    # as array
    predicts_array = np.stack([np.asarray(predicts[i]) for i in range(len(predicts))])
    samples, rate = rejectance_sampling(predicts_array, n_samples=100, enhanced_output=True)
    print(samples.shape)
    print(f'Acceptance rate: {rate}')

    t_0 = time.time()
    P = np.random.dirichlet([1]*3, size=(100, 10))
    P_hat = rejectance_sampling_p(P)
    t_1 = time.time()
    print(f'Time for one sampling 1000 times: {t_1-t_0}')

    P = init_pmatrix(N=1, M=5, K=5, u=1) # P matrix of point predictions as in experiments
    P = P.squeeze()

    samples, rate = rejectance_sampling(P, n_samples=1, enhanced_output=True)
    print(f"Acceptance rate: {rate}")







