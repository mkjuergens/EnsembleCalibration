import numpy as np

from scipy.stats import norm

from ensemblecalibration.calibration.calibration_estimates.distances import matrix_kernel
from ensemblecalibration.calibration.calibration_estimates.distances import median_heuristic
from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar


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

def skce_ul_arr(p_bar: np.ndarray, y: np.ndarray, dist_fct, sigma: float = 2.0,
                 use_median_sigma: bool = False):
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

    n = round(p_bar.shape[0]/2)
    # transform y to one-hot encoded labels
    yoh = np.eye(p_bar.shape[1])[y,:]
    stats = np.zeros(n)
    if use_median_sigma:
        sigma = median_heuristic(p_hat=p_bar, y_labels=y)
    for i in range(0,n):
        stats[i] = h_ij(p_bar[(2*i),:], p_bar[(2*i)+1,:], yoh[(2*i),:], yoh[(2*i)+1,:], dist_fct=dist_fct,
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
        distance function used in the calibration statistic
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
            stats[count] = h_ij(P_bar[i, :], P_bar[j, :], y_one_hot[i, :], y_one_hot[j,:],
                                 dist_fct=dist_fct, sigma=sigma)
            count+=1

    return stats

def skce_ul_obj(p_bar: np.ndarray, y: np.ndarray, params: dict):
    """calculates the estimator SKCE_ul for the squared kernel calibration error
      proposed in Widman et al., 2019

    Parameters
    ----------
    p_bar : np.ndarray of shape (n_samples, n_classes)
        matric containing probabilistic predictions for each instance
    y : np.ndarray of shape (n_samples,)
        vector containing labels
    params : dictionary of test parameters
        dictionary of test parameters

    Returns
    -------
    float
        value of the estimator
    """
    hat_skce_ul_arr = skce_ul_arr(
        p_bar, y, dist_fct=params["dist"], sigma=params["sigma"]
    )
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)

    return hat_skce_ul_mean

def skce_uq_obj(p_bar: np.ndarray, y: np.ndarray, params: dict):
    """calculates the estimator SKCE_uq for the squared kernel calibration error
      proposed in Widman et al., 2019

    Parameters
    ----------
    p_bar : np.ndarray of shape (n_samples, n_classes)
        matric containing probabilistic predictions for each instance
    y : np.ndarray of shape (n_samples,)
        vector containing labels
    params : dictionary of test parameters
        dictionary of test parameters

    Returns
    -------
    float
        value of the estimator
    """
    # calculate SKCE_uq estimate
    hat_skce_uq_arr = skce_uq_arr(
        p_bar, y, dist_fct=params["dist"], sigma=params["sigma"]
    )
    hat_skce_uq_mean = np.mean(hat_skce_uq_arr)

    return hat_skce_uq_mean

def skce_uq_obj_new(
    weights_l: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    params: dict,
    take_square: bool = True,
):
    """New test objective for the SKCE_uq using a weight matrix containing a weight vector for each instance
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

    hat_skce_uq_arr = skce_uq_arr(
        P_bar=P_bar, y=y, dist_fct=params["dist"], sigma=params["sigma"]
    )
    hat_skce_uq_mean = np.mean(hat_skce_uq_arr)

    if take_square:
        hat_skce_uq_mean = hat_skce_uq_mean**2

    return hat_skce_uq_mean


def skce_ul_obj_new(
    weights_l: np.ndarray,
    P: np.ndarray,
    y: np.ndarray,
    params: dict,
    take_square: bool = True,
):
    """New test objective for the SKCE_ul using a weight matrix containing a weight vector for
    each instance

    Parameters
    ----------
    weights_l : np.ndarray
        weight matrix of shape (N,M)
    P : np.ndarray
        tensor containing point predictions of each ensemble member for each instance (N, M, K)
    y : np.ndarray
        array of shape (N,) containing labels
    params : dict
        dictionary containing test parameters

    Returns
    -------
    _type_
        _description_
    """

    P_bar = calculate_pbar(weights_l, P, reshape=True)

    hat_skce_ul_arr = skce_ul_arr(
        p_bar=P_bar, y=y, dist_fct=params["dist"], sigma=params["sigma"]
    )
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)

    if take_square:
        hat_skce_ul_mean = hat_skce_ul_mean**2

    return hat_skce_ul_mean

def skceul(P, y, params, square_out: bool = False):

    """ SKCE_ul estimator for strong classifier calibration.

    Based on Calibration tests in multi-class classification: A unifying framework by Widmann et al.

    Arguments
    ---------
    P : ndarray of shape (n_samples, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params for kernel and test

    Returns
    ------
    hat_skce_ul_mean : estimation of SKCE_ul
    """
    # calculate SKCE_ul estimate
    hat_skce_ul_arr = skce_ul_arr(P, y, dist_fct=params["dist"], sigma=params["sigma"])
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)

    if square_out:
        hat_skce_ul_mean = hat_skce_ul_mean ** 2

    return hat_skce_ul_mean

def skceuq(P, y, params, square_out: bool = False):

    """ SKCE_uq estimator for strong classifier calibration.

    Based on Calibration tests in multi-class classification: A unifying framework by Widmann et al.

    Arguments
    ---------
    P : ndarray of shape (n_samples, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    dist_fct: distance function used in teh matrix valued kernel
    sigma: bandwidth used in the matrix valued kernel

    Returns
    ------
    hat_skce_uq_mean : estimation of SKCE_uq
    """
    
    # calculate SKCE_ul estimate
    hat_skce_uq_arr = skce_uq_arr(P, y, dist_fct=params["dist"], sigma=params["sigma"])
    hat_skce_uq_mean = np.mean(hat_skce_uq_arr)

    if square_out:
        hat_skce_uq_mean = hat_skce_uq_mean ** 2

    return hat_skce_uq_mean


def skceultest(P, y, params):
    """ SKCE_ul test statistic for strong classifier calibration.

    Based on Calibration tests in multi-class classification: A unifying framework by Widmann et al.

    Arguments
    ---------
        P : ndarray of shape (n_samples, n_classes) containing probs
        y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
        n_bins : number of bins used 

    Returns
    ------
        stat : test statistic
        pval : p-value
    """
    # calculate SKCE_ul estimate
    hat_skce_ul_arr = skce_ul_arr(P, y, dist_fct=params["dist"],
     sigma=params["sigma"])
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)
    hat_skce_ul_std = np.std(hat_skce_ul_arr)
    # calculate test statistic and P-value
    stat = (np.sqrt(len(hat_skce_ul_arr))/hat_skce_ul_std)*hat_skce_ul_mean
    pval = (1-norm.cdf(stat))

    return stat, pval

def skce_ul_obj_lambda(l_weights: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray,
                        params: dict):
    p_bar = calculate_pbar(weights_l=l_weights, P=p_probs, reshape=False, n_dims=1)
    stat = skce_ul_obj(p_bar=p_bar, y=y_labels, params=params)

    return stat

def skce_uq_obj_lambda(l_weights: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray,
                       params: dict):
    p_bar = calculate_pbar(weights_l=l_weights, P=p_probs, reshape=False, n_dims=1)
    stat = skce_uq_obj(p_bar=p_bar, y=y_labels, params=params)
    return stat
