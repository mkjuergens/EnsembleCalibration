"""functions for measuring distances"""
import numpy as np
import ot
from sklearn.metrics import pairwise_distances
from math import log
from sklearn.metrics import pairwise_distances

from scipy.spatial.distance import jensenshannon
from scipy.spatial import KDTree
from scipy.stats import entropy


def mmd(p: np.ndarray, q: np.ndarray, use_optim_bw: bool = True, bw: float = 1.0):
    """Maximum Mean discrepancy between two samples of probabilistic predictions.

    Parameters
    ----------
    p : np.ndarray of shape (n_samples, n_classes)
        vector of probabilistic predictions for sample 1
    q : np.ndarray of shape (n_samples, n_classes)
        vector of probabilistic predictions for sample 2
    use_optim_bw : bool, optional
        whether to use the optimal bandwidth, by default True
    bw : float, optional
        bandwidth, by default 1.0

    Returns
    -------
    float
        value of the (empirical) MMD
    """

    n_1, n_classes= p.shape
    n_2, n_classes = q.shape
    assert p.shape == q.shape, "p and q must have the same shape"
    pq = np.concatenate([p, q], axis=0)
    distances = pairwise_distances(pq, metric="euclidean")
    if use_optim_bw:
        bw = optim_bw(p, q)
    k = np.exp(
        -(distances ** 2) / (2 * bw ** 2))  # + epsilon * torch.eye(n_1 + n_2)  # 2. for numerical stability
    k_x = k[:n_1, :n_1]
    k_y = k[n_1:, n_1:]
    k_xy = k[:n_1, n_1:]

    mmd_score = k_x.sum() / (n_1 * (n_1 - 1)) + k_y.sum() / (n_2 * (n_2 - 1)) - 2 * k_xy.sum() / (n_1 * n_2)
    return mmd_score


def optim_bw(p: np.ndarray, q: np.ndarray):
    """Optimal bandwidth for the MMD between two samples of probabilistic predictions.

    Parameters
    ----------
    p : np.ndarray of shape (n_samples, n_classes)
        vector of probabilistic predictions for sample 1
    q : np.ndarray of shape (n_samples, n_classes)
        vector of probabilistic predictions for sample 2

    Returns
    -------
    float
        optimal bandwidth
    """

    n_1, n_classes = p.shape
    n_2, n_classes = q.shape
    assert p.shape == q.shape, "p and q must have the same shape"
    pq = np.concatenate([p, q], axis=0)
    distances = pairwise_distances(pq, metric="euclidean")
    median_dist = np.median(distances)
    bw = median_dist / np.sqrt(2)
    return bw


def kl_divergence(x: np.ndarray, y: np.ndarray):
    """Kullback-Leibler divergence between two (samples of) multivariate point predictions.

    Parameters
    ----------
    p : np.ndarray
        point estimate of shape (n_isntances, n_classes)
    q : np.ndarray
        second point estimate of shape (n_instances, n_classes)

    Returns
    -------
    float
        Kullback-Leibler divergence
    """

        # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)


    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))
    

def jensen_shannon_dist(p: np.ndarray, q: np.ndarray):
    """Jensen-Shannon distance between two point predictions.

    Parameters
    ----------
    p : np.ndarray
        point estimate of shape (n_classes,)
    q : np.ndarray
        second point estimate of shape (n_classes,)

    Returns
    -------
    float
        Jensen-Shannon distance
    """

    assert p.shape == q.shape, "p and q need to have the same shape"
    n_instances, n_classes = p.shape
    dist = 0
    for n in range(n_instances):
        dist += jensenshannon(p[n, :], q[n, :])

    return dist / n_instances


def w1_distance(p: np.ndarray, q: np.ndarray):
    """
    Function to calculate the Wasserstein-1 distance between two point predictions.

    Parameters
    ----------
    p : np.ndarray
        point prediction of shape (n_instances, n_classes)
    q : np.ndarray
        second point prediction of shape (n_instances, n_classes)

    Returns
    -------
    float
        Wasserstein-1 distance
    """

    dist = ot.dist(p, q, metric="euclidean")
    M = ot.emd(np.ones(p.shape[0]) / p.shape[0], np.ones(q.shape[0]) / q.shape[0], dist)

    return np.sum(M * dist)


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
    assert p.shape == q.shape, "p and q need to have the same shape"
    n_classes = p.shape[-1]
    p = p.reshape(-1, n_classes)
    q = q.reshape(-1, n_classes) 

    return (0.5 * np.sum(np.abs(p - q)))/p.shape[0]


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

    return np.sqrt(np.sum((p - q) ** 2))


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
    id_k = np.eye(len(p))  # identity matrix
    return np.exp((-1 / sigma) * (dist_fct(p, q) ** 2) * id_k)


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
    p = np.random.dirichlet([1] * 2, size=10)
    q = np.random.dirichlet([1] * 2, size=10)
    print(w1_distance(p, q))
    print(tv_distance(p, q))
    print(jensen_shannon_dist(p, q))
    print(kl_divergence(p, q))
    print(mmd(p, q))
