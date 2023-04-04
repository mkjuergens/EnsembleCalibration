"""functions for measuring distances"""
import numpy as np
import torch
from sklearn.metrics import pairwise_distances

def tv_distance(p: np.ndarray, q: np.ndarray):
    """total variation distance between two point predictions.

    Parameters
    ----------
    p : np.ndarray
        point estimate of shape (n_classes,)
    q : np.ndarray
        second point estimate of shape (n_classes,)

    Returns
    -------
    float
        variation distance
    """

    return 0.5*np.sum(np.abs(p-q))


def l2_distance(p: np.ndarray, q: np.ndarray):
    """l2 distance between two point predictions.

    Parameters
    ----------
    p : np.ndarray
        point prediction of shape (n_classes,)
    q : np.ndarray
        second point prediction of shape (n_classes, )

    Returns
    -------
    _float
        distance
    """

    return np.sqrt(np.sum((p-q)**2))


def matrix_kernel(p: np.ndarray, q: np.ndarray, dist_fct, sigma: float = 2.0):
    """returns the matrix-valued kernel evaluated at two point predictions 

    Parameters
    ----------
    p : np.ndarray
        first point prediction 
    q : np.ndarray
        second point prediction
    sigma : float
        bandwidth 
    dist_fct : _type_
        distance measure. Options: {tv_distance, l2_distance}

    Returns
    -------
    np.ndarray
        _description_
    """
    p = p.squeeze()
    q = q.squeeze()

    assert len(p) == len(q), "vectors need to be of the same length"
    id_k = np.eye(len(p)) # identity matrix
    return np.exp((-1/sigma)* (dist_fct(p, q)**2)* id_k)


def median_heuristic(p_hat: np.ndarray, y_labels: np.ndarray):
    """calculates the optimal bandwidth of the kernel used in the SKCE using a median heuristic,
    where the pairwise distances of the predicted labels and the real labels are calculated and the 
    median is taken as a reference bandwidth.

    Parameters
    ----------
    p_hat: torch.Tensor
        tensor of predicted probabilities
    y_labels : torch.Tensor
        tensor of real labels

    Returns
    -------
    float
        bandwidth
    """
    # get predictions of the model
    y_pred = np.array(np.argmax(p_hat, axis=1))
    # reshape to two dimensions
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    y_labels = y_labels.reshape(y_labels.shape[0], -1)
    dist = pairwise_distances(y_pred, y_labels)
    sigma_bw = np.median(dist) / 2

    return sigma_bw


    

if __name__ == "__main__":
    p_test = np.array([[1,0,0]])
    q_test = np.array([[0, 0.5, 0.5]])

    print(f'total variation distance: {tv_distance(p_test, q_test)}')
    print(f'l2 distance: {l2_distance(p_test, q_test)}')
    m = matrix_kernel(p_test, q_test, l2_distance)
    print(m)
    p_hat = np.random.dirichlet([1]*10, size=1000)
    y_labels = np.random.randint(0, 5, size= (1000,))
    print(y_labels)
    print(median_heuristic(p_hat=p_hat, y_labels=y_labels))

