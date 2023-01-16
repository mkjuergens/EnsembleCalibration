from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull
from composition_stats import ilr, ilr_inv, alr, alr_inv


def ilr_transform(x : np.ndarray, noise: bool=True, sigma: float = 1e-6):
    """isometric log-transformation of a given input vector.
        Can be used to transform d-dimensional points from a d-1 dimensional
        simplex to the (d-1) dimensional subspace of fully rank.

    Parameters
    ----------
    x : np.ndarray
        input vector of shape (n_samples, n_classes)
    noise: bool
            whether to add small noise to the data. can be used if the data contains 
            zeros and a log-ratio transform cannot be applied.
    sigma: float
            variance of the white noise

    Returns
    -------
    np.ndarray
        transformed point
    """
    if noise:
        n_samples, n_classes = x.shape
        eps = np.random.normal(0, sigma,size= (n_samples, n_classes)) # random noise
        x = x + np.abs(eps)

    assert np.all((x>0)), "all entries of the input data need to be larger than 0 to perform a log-ratio transformation"

    x_transformed = ilr(x)

    return x_transformed

def ilr_transform_inv(x_trans: np.ndarray):
    """inverse log ratio transform

    Parameters
    ----------
    x_trans : np.ndarray
        log-ratio transformed vector

    Returns
    -------
    np.ndarray
        back-transformed vector
    """

    x = ilr_inv(x_trans)

    return x

def alr_transform(x : np.ndarray, noise: bool=True, sigma: float = 1e-5):
    """additive log-transformation of a given input vector.
        Can be used to transform d-dimensional points from a d-1 dimensional
        simplex to the (d-1) dimensional subspace of fully rank.

    Parameters
    ----------
    x : np.ndarray
        input vector of shape (n_samples, n_classes)
    noise: bool
            whether to add small noise to the data. can be used if the data contains 
            zeros and a log-ratio transform cannot be applied.
    sigma: float
            variance of the white noise

    Returns
    -------
    np.ndarray
        transformed point
    """
    if noise:
        n_samples, n_classes = x.shape
        eps = np.random.normal(0, sigma,size= (n_samples, n_classes)) # random noise
        x = x + np.abs(eps)

    assert np.all((x>0)), "all entries of the input data need to be larger than 0 to perform a log-ratio transformation"

    x_transformed = alr(x)

    return x_transformed

def alr_transform_inv(x_trans: np.ndarray):
    """inverse additive log ratio transform

    Parameters
    ----------
    x_trans : np.ndarray
        log-ratio transformed vector

    Returns
    -------
    np.ndarray
        back-transformed vector
    """

    x = alr_inv(x_trans)

    return x



def get_vertex_pairs_hull(x: np.ndarray, transform: Optional[str] = None):
    """returns an array of vertices for the given K-dimensional points in the (K-1)-simplex.
    Works by first projecting them int the (K-1)-dimensional coordinate plane, calculating the
     vertices of the convex hull, and then back-transforming to K dimensions.

    Parameters
    ----------
    x : np.ndarray of shape (n_points, K)
        initial points
    
    transform: string. Options: {None, 'additive'}
        transformation to be used to map the points on the lower dimensional subspace.
        Default is None, using the isometric log-ratio transform.

    Returns
    -------
    np.ndarray of shape (n_vertices, K)
        vertex points 
    """

    # transform points on lower dimensional subspace
    if transform is None:
        x_trans = ilr_transform(x)
    elif transform =="additive":
        x_trans = alr_transform(x)

    hull = ConvexHull(x_trans)
    v_k = x_trans[hull.vertices]

    # back transformation
    if transform is None:
        vertices = ilr_transform_inv(v_k)
    elif transform == 'additive':
        vertices = alr_transform_inv(v_k)

    return vertices


if __name__ == "__main__":
    x_test = np.array([[1, 0 ,0], [0,0.5,0.5], [0,0,1],  [1, 0, 0]])
    x_transf = ilr_transform(x_test, noise=True)
    x_transf_alr = alr_transform(x_test, noise=True)
    x_back = alr_transform_inv(x_transf_alr)
    print(x_transf_alr)
    print(x_back)
    v = get_vertex_pairs_hull(x_test, transform="additive")
    print(v)


