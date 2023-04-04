import numpy as np
from scipy.stats import chi2

from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar

def hl_obj_new(weights_l: np.ndarray, P: np.ndarray, y: np.ndarray, params: dict):
    """New objective for the Hosmer-Lemeshow test where the weights are now a matrix containing a weight vector for each instance.
         In this case, the weight vector is a flattened version of the matrix containing the weight vectors
         for each row/instance.

     Parameters
     ----------
     weights_l : np.ndarray
         matrix of shape (N*M,). flattened matrix of weight coefficients
    P : np.ndarray
         tensor of shape (N, M, K)
     y : np.ndarray
         vector of shape (N,)
     params : dict
         test parameters

     Returns
     -------
     float
    """

    P_bar = calculate_pbar(weights_l=weights_l, P=P, reshape=True)
    stat, _ = hltest(P_bar, y, params)

    return stat

def hltest(P, y, params, return_p_val: bool = False):
    """ Hosmer & Lemeshow test for strong classifier calibration

    Arguments
    ---------
        P : ndarray of shape (n_samples, n_classes) containing probs
        y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
        params: parameters of the test
        return_p_val: booelean
            enhance outoput with p-value of the test
        
    Return
    ------
        stat : test statistic
        (pval : p-value)
    """
    # calculate test statistic
    stat = 0
    # get idx for complement of reference probs in increasing order of prob
    idx = np.argsort(1-P[:,0])[::-1]
    # split idx array in nbins bins of roughly equal size
    idx_splitted = np.array_split(idx, params["n_bins"])
    # run over different cells and calculate stat
    stat = 0
    for k in range(P.shape[1]):
        for bin_bk in idx_splitted:
            o_bk = np.sum((y==k)[bin_bk])
            p_bk = np.sum(P[bin_bk,k])
            dev_bk = ((o_bk-p_bk)**2)/p_bk
            stat += dev_bk
    # and finally calculate righttail P-value
    pval = 1-chi2.cdf(stat,df=(params["n_bins"]-2)*(P.shape[1]-1))
    
    if return_p_val:
        return stat, return_p_val
    else:
        return stat
    
def hl_obj(p_bar, y, params):
    """objective function of the Hosmer-Lemeshow test

    Parameters
    ----------
    p_bar: np.ndarray of shape (n_samples, n_classes)
            matrix containing probabilistic predictions for each class
    y : np.ndarray
        weigth vector
    params : dict
        test parameters

    Returns
    -------
    float
        realization of the test statistic
    """
    stat = hltest(p_bar, y, params)

    return stat

def hl_obj_lambda(weights_l: np.ndarray,p_probs: np.ndarray, y_labels: np.ndarray, params: dict):

    p_bar = calculate_pbar(weights_l=weights_l, P=p_probs, reshape=False, n_dims=1)
    stat = hl_obj(p_bar=p_bar, y=y_labels, params=params)

    return stat