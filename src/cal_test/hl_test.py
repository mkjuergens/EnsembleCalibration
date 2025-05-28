import numpy as np
from scipy.stats import chi2


def hltest(p_preds: np.ndarray, y_labels: np.ndarray, params: dict, return_p_val: bool = False):
    """ Hosmer & Lemeshow test for strong classifier calibration

    Arguments
    ---------
        p_preds : ndarray of shape (n_samples, n_classes) containing probs
        y_labels : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
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
    idx = np.argsort(1-p_preds[:,0])[::-1]
    # split idx array in nbins bins of roughly equal size
    idx_splitted = np.array_split(idx, params["n_bins"])
    # run over different cells and calculate stat
    stat = 0
    for k in range(p_preds.shape[1]):
        for bin_bk in idx_splitted:
            o_bk = np.sum((y_labels==k)[bin_bk])
            p_bk = np.sum(p_preds[bin_bk,k])
            dev_bk = ((o_bk-p_bk)**2)/p_bk
            stat += dev_bk
    # and finally calculate righttail P-value
    pval = 1-chi2.cdf(stat,df=(params["n_bins"]-2)*(p_preds.shape[1]-1))
    
    if return_p_val:
        return stat, pval
    else:
        return stat