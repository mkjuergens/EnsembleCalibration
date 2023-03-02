"""
Functions for calculation different measures of calibration errors used in
Mortier et al, 2022.
In particular, the calibration in teh strong sense introduced in Widman et al, 2019 is used
to measure calibration in the strong sense . Here, two (computationally different exhaustive)
measures are introduced, namely skce_uq and skce_ul.
"""

import numpy as np
from scipy.stats import chi2
from ensemblecalibration.calibration.distances import l2_distance, tv_distance, matrix_kernel

def h_ij(p_i: np.ndarray, p_j: np.ndarray, y_i: np.ndarray, y_j: np.ndarray, dist_fct, 
        sigma: float=2.0):
    """calculates the entries h_ij which are summed over in the expression of the calibration

    Parameters
    ----------
    p_i : np.ndarray

        first point prediction
    p_j : np.ndarray
        second point prediction
    y_i : np.ndarray
        one hot encoding of labels for sample j
    y_j : np.ndarray
        one hot encoding of labels for sample j
    dist_fct : 
        function used as a distance measure in the matrix valued kernel
    sigma : float, optional
        bandwidth, by default 2.0

    Returns
    -------
    np.ndarray

    """
    
    gamma_ij = matrix_kernel(p_i, p_j, dist_fct=dist_fct, sigma=sigma)
    y_ii = y_i - p_i
    y_jj = y_j - p_j

    h_ij = y_ii @ gamma_ij @ y_jj

    return h_ij

def skce_ul_arr(P_bar: np.ndarray, y: np.ndarray, dist_fct, sigma: float = 2.0):
    """calculates the skce_ul calibration error used as a test statistic in Mortier et  al, 2022.

    Parameters
    ----------
    P_bar :  np.ndarray of shape (n_predictors, n_classes)
        matrix containing all ensmeble predictions
    y : np.ndarray
        vector with class labels of shape
    dist_fct : [tv_distance, l2_distance]
        distance function to be used
    sigma : float
        bandwidth used in the matrix valued kernel

    Returns
    -------
    np.ndarray
        _description_
    """

    n = round(P_bar.shape[0]/2)
    # transform y to one-hot encoded labels
    yoh = np.eye(P_bar.shape[1])[y,:]
    stats = np.zeros(n)
    for i in range(0,n):
        stats[i] = h_ij(P_bar[(2*i),:], P_bar[(2*i)+1,:], yoh[(2*i),:], yoh[(2*i)+1,:], dist_fct=dist_fct,
        sigma=sigma)

    return stats

def skce_uq_arr(P_bar: np.ndarray, y: np.ndarray,dist_fct, sigma: float = 2.0):
    """calculates the skcl_uq calibration error used as a test statistic for one sample.

    Parameters
    ----------
    P_bar : np.ndarray of shape (n_predictors, n_classes)
        matrix containing all ensemble predictions
    y : np.ndarray
       vector of labels of shape (M")
    dist_fct : _type_
        dsitance function used in the calibration statistic
    sigma : float, optional
        bandwidth, by default 2.0

    Returns
    -------
    np.ndarray
        _description_
    """

    N, M = P_bar.shape[0], P_bar.shape[1] # p is of shape (n_samples, m_predictors, n_classes)
    # one-hot encoding
    y_one_hot =np.eye(M)[y, :]

    # binomial coefficient n over 2
    stats = np.zeros(int((N*(N-1))/2))
    count=0
    for j in range(1, N):
        for i in range(j):
            stats[count] = h_ij(P_bar[i, :], P_bar[j, :], y_one_hot[i, :], y_one_hot[j,:], dist_fct=dist_fct, sigma=sigma)
            count+=1

    return stats


def hltest(P, y, params):
    """ Hosmer & Lemeshow test for strong classifier calibration

    Arguments
    ---------
        P : ndarray of shape (n_samples, n_classes) containing probs
        y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
        params: parameters of the test
        
    Return
    ------
        stat : test statistic
        pval : p-value
    """
    # calculate test statistic
    stat = 0
    # get idx for complement of reference probs in increasing order of prob
    idx = np.argsort(1-P[:,0])[::-1]
    # split idx array in nbins bins of roughly equal size
    idx_splitted = np.array_split(idx, params["nbins"])
    # run over different cells and calculate stat
    stat = 0
    for k in range(P.shape[1]):
        for bin_bk in idx_splitted:
            o_bk = np.sum((y==k)[bin_bk])
            p_bk = np.sum(P[bin_bk,k])
            dev_bk = ((o_bk-p_bk)**2)/p_bk
            stat += dev_bk
    # and finally calculate righttail P-value
    pval = 1-chi2.cdf(stat,df=(params["nbins"]-2)*(P.shape[1]-1))
    
    return stat, pval













