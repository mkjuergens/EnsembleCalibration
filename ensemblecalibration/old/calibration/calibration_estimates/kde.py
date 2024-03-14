import numpy as np

from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon


def estimate_density(samples: np.ndarray, kernel: str = "gaussian", bw: float = 0.2):
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(samples)

    return kde


def kl_divergence_estimate(p: np.ndarray, q: np.ndarray, bw: float = 0.2):
    """Kullback-Leibler divergence between two samples of probabilistic predictions.

    Parameters
    ----------
    p : np.ndarray of shape (n_samples, n_classes)
        vector of probabilistic predictions for sample 1
    q : np.ndarray of shape (n_samples, n_classes)
        vector of probabilistic predictions for sample 2
    bw : float, optional
        bandwidth, by default 0.2

    Returns
    -------
    float
        value of the (empirical) KL divergence
    """
    assert p.shape == q.shape, "p and q must have the same shape"

    p_kde = estimate_density(p, bw=bw)
    q_kde = estimate_density(q, bw=bw)

    x_min = np.min(np.concatenate((p, q)))
    x_max = np.max(np.concatenate((p, q)))

    print(x_min, x_max)

    x = np.linspace(x_min, x_max, 1000).reshape(-1, 1)

    p_1 = np.exp(p_kde.score_samples(x))
    p_2 = np.exp(q_kde.score_samples(x))

    #normalize
    p_1 = p_1 / np.sum(p_1)
    p_2 = p_2 / np.sum(p_2)

    kl_div = np.sum(p_1 * np.log(p_1 / p_2))

    return kl_div

if __name__ == "__main__":
    p = np.random.rand(100, 10)
    q = np.random.rand(100, 10)

    kl_div = kl_divergence_estimate(p, q)

    print(kl_div)
