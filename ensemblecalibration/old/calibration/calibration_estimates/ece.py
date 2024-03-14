import numpy as np

from pycalib.metrics import classwise_ECE, conf_ECE
from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar

def confece_obj_new(weights_l: np.ndarray, P, y, params):
    """New test objective for confidence ECE taking into account that lambda is now a matrix of shape (N,M)

    Parameters
    ----------
    weights_l : np.ndarray
        matrix of lambda weights of shape (N,M)
    P : no.ndarray of shape (N,M,K)
        tensor containing point predictions for every instance for every predictor
    y : np.ndarray
        array of shape (N,) containing the label for every instance
    params : dictionary
        contains the number of bins as a key

    Returns
    -------
    np.ndarray
        matrix of shape (N, K)
    """

    P_bar = calculate_pbar(weights_l, P, reshape=True)
    stat = conf_ECE(y, P_bar, params["n_bins"])

    return stat


def classece_obj_new(weights_l, P, y, params):
    """New test objective for the classwise ECE with lambda being a function dependent on the features

    Parameters
    ----------
    weights_l : np.ndarray
        matrix of shape (N, M) containing the weight function evaluated at each instance
    P : np.ndarray
        tensor of shape (N,M,K)
    y : np.ndarray
        array of shape (N,)
    params : dictionary
        _description_

    Returns
    -------

        _description_
    """
    P_bar = calculate_pbar(weights_l, P, reshape=True)
    # transform y to indicator matrix (needed for classwise_ECE)
    yind = np.eye(P.shape[2])[y, :]
    # calculate classwise ECE
    stat = classwise_ECE(yind, P_bar, 1, params["n_bins"])

    return stat


def confece_obj(p_bar: np.ndarray, y: np.ndarray, params: dict):
    """calculate confidence-wise ECE for a given predictor and labels

    Parameters
    ----------
    p_bar :  np.ndarray of shape (n_samples, n_classes)
        matrix containing probabilistics predictions for each class
    y : np.ndarray
        vector containing labels
    params : dict
        dictionary of test parameters

    Returns
    -------
    evaluation of the estimator for the confECE
    """
    # calculate confidence ECE
    stat = conf_ECE(y, p_bar, params["n_bins"])

    return stat


def classece_obj(p_bar: np.ndarray, y: np.ndarray, params: dict):
    """calculate classwise ECE for a given predictor and labels

    Parameters
    ----------
    p_bar : np.ndarray
        matrix of shape (n_samples, n_classes) containing the probabilistic predictions for each class
    y : np.ndarray
        vector of shape (n_samples,) containing labels of each class
    params : dict
        dictionary of test parameters

    Returns
    -------
    stat    
        estimator of classwise ECE evaluated on the data
    """
    # transform y to indicator matrix (needed for classwise_ECE)
    yind = np.eye(p_bar.shape[1])[y, :]
    # calculate classwise ECE
    stat = classwise_ECE(yind, p_bar, 1, params["n_bins"])

    return stat

def classece_obj_lambda(weights_l: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray, 
                        params: dict, x_dependency: bool = False):
    if x_dependency:
        p_bar = calculate_pbar(weights_l=weights_l, P=p_probs, reshape=True, n_dims=2)
    else:
        p_bar = calculate_pbar(weights_l=weights_l, P=p_probs, reshape=False, n_dims=1)
    stat = classece_obj(p_bar=p_bar, y=y_labels, params=params)
    return stat

def confece_obj_lambda(weights_l: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray, 
                        params: dict, x_dependency: bool = False):
    
    if x_dependency:
        p_bar = calculate_pbar(weights_l=weights_l, P=p_probs, reshape=True, n_dims=2)
    else:
        p_bar = calculate_pbar(weights_l=weights_l, P=p_probs, reshape=False, n_dims=1)
    stat = confece_obj(p_bar=p_bar, y=y_labels, params=params)
    return stat


def confece(P, y, params):

    """ confidence ECE estimator for confidence classifier calibration.

    Arguments
    ---------
        P : ndarray of shape (n_samples, n_classes) containing probs
        y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
        n_bins: integer defining the number of bins used for the test statistic

    Returns
    ------
        estimation of confidence ECE
    """
    return conf_ECE(y, P, params["n_bins"])


def classece(P, y, params):
    """ classwise ECE estimator for strong classifier calibration.

    Arguments
    ---------
        P : ndarray of shape (n_samples, n_classes) containing probs
        y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
        params : dict, params

    Returns
    ------
        estimation of classwise ECE
    """
    # convert y to indicator matrix (needed for classwise_ECE)
    yind = np.eye(P.shape[1])[y,:]

    return classwise_ECE(yind, P, 1, params["n_bins"])