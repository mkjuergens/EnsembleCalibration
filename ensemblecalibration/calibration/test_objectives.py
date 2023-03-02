import numpy as np

from pycalib.metrics import conf_ECE, classwise_ECE

from ensemblecalibration.calibration.calibration_measures import skce_ul_arr, skce_uq_arr, hltest
from ensemblecalibration.calibration.distances import tv_distance

def calculate_pbar(weights_l: np.ndarray, P: np.ndarray, reshape: bool = True):
    """calculate convex combination of a weight matrix of shape (N,M) and a tensor of point predictions
        of shape (N, M, K) such that the new matrix contains a point predictions for each instance and is 
        of shape (N, K).
    Parameters
    ----------
    weights_l : np.ndarray
        weight matrix of shape (N, M)
    P : np.ndarray
        tensor of point predcitions for each isntance for each predcitor, shape (N,M,K)

    Returns
    -------
    np.ndarray
        matrix containinng one new prediction for each instance, shape (N, K)
    """

    n_rows = P.shape[0]
    if reshape:
        assert len(weights_l) % n_rows == 0, " weight vector needs to be a multiple of the number of rows"
        weights_l = weights_l.reshape(n_rows, -1)

    assert weights_l.shape[0] == P.shape[0], " numer of samples need to be the same for P and weights_l"
    assert weights_l.shape[1] == P.shape[1], " numer of ensemble members need to be the same for P and weights_l"

    P_bar = (weights_l[:,:,np.newaxis]*P).sum(-2) # sum over second axis to get diagonal elements

    return P_bar


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




def skce_uq_obj_new(weights_l: np.ndarray, P: np.ndarray, y: np.ndarray, params: dict):
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

    hat_skce_uq_arr = skce_uq_arr(P_bar=P_bar, y=y, dist_fct=params["dist"], sigma=params["sigma"])
    hat_skce_uq_mean = np.mean(hat_skce_uq_arr)

    return hat_skce_uq_mean

def skce_ul_obj_new(weights_l: np.ndarray, P: np.ndarray, y: np.ndarray, params: dict):
    """New test objective for the SKCE_ul using a weight matrix containing a weight vector for each instance

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

    hat_skce_ul_arr = skce_ul_arr(P_bar=P_bar, y=y, dist_fct=params["dist"], sigma=params["sigma"])
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)

    return hat_skce_ul_mean



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
    yind = np.eye(P.shape[2])[y,:]
    # calculate classwise ECE
    stat = classwise_ECE(yind, P_bar, 1, params["n_bins"])

    return stat


if __name__ == "__main__":
    K = 3
    M = 10
    P = np.random.dirichlet([1]*K, size=(100,10)).reshape(100,10, 3)
   ## print(P.shape)
    lambda_weight = np.random.dirichlet([1]*M, size=100).reshape(100, 10)
    #print(lambda_weight.shape)
   # print(P.shape)
    P_bar = lambda_weight @ P
    #print(P_bar.shape)
    P_bar = (lambda_weight[:,:,np.newaxis]*P).sum(-2)
    print(P_bar.shape)
   # print(P_bar.shape)
    y = np.random.randint(0, 1, size=100)
   # print(y.shape)

    params = {"n_bins": 10, "dist": tv_distance, "sigma": 2.0}
    #obj = confece_obj(lambda_weight, P, y, params)
    #print(obj)
    obj = confece_obj_new(lambda_weight, P, y, params)
    print(obj)
    obj_2 = skce_ul_obj_new(lambda_weight, P, y, params)
    print(obj_2)
    obj_3 = skce_uq_obj_new(lambda_weight, P, y, params)
    print(obj_3)