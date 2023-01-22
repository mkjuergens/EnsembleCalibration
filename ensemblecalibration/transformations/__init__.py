import numpy as np
from scipy.spatial import ConvexHull

from ensemblecalibration.transformations.isometric_logtransform import ilr_transform, ilr_transform_inv, alr_transform, alr_transform_inv
from ensemblecalibration.transformations.projections_2d import project_points2D, planes_to_coordinates3D

def get_vertex_pairs_hull(x: np.ndarray, transform: str = 'sqrt'):
    """returns an array of vertices for the given K-dimensional points in the (K-1)-simplex.
    Works by first projecting them int the (K-1)-dimensional coordinate plane, calculating the
     vertices of the convex hull, and then back-transforming to K dimensions.

    Parameters
    ----------
    x : np.ndarray of shape (n_points, K)
        initial points
    
    transform: string. Options: {None, 'additive', 'sqrt}
        transformation to be used to map the points on the lower dimensional subspace.
        Default is None, using the isometric log-ratio transform.

    Returns
    -------
    np.ndarray of shape (n_vertices, K)
        vertex points 
    """

    # transform points on lower dimensional subspace
    x_trans = transform_points(x, transform=transform)

    hull = ConvexHull(x_trans)
    v_k = x_trans[hull.vertices]

    # inverse transformation
    vertices = inv_transform_points(v_k, transform=transform)

    return vertices

def transform_points(x: np.ndarray, transform: str = 'sqrt'):
    """transforms K- dimensional points from a (K-1) polytope to the (K-1) dimensional 
    space using a predefined transformation.

    Parameters
    ----------
    x : np.ndarray of shape (M, K) or shape (K,)
        input points
    transform : str, optional
        needs to be in ['sqrt', 'isometric', 'additive'], by default 'sqrt'

    Returns
    -------
    np.ndarray
       transformed points of shape (K-1,)
    """
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if transform == 'sqrt':
        assert x.shape[1] == 3, f"""sqrt transformation only suitbale for 3 dimensional data,
                                    but it is {x.shape[1]} dimensional"""
        x_trans = project_points2D(x)
    elif transform == 'additive':
        x_trans = alr_transform(x)
    elif transform == 'isometric':
        x_trans = ilr_transform(x)
    else:
        raise NameError("transform needs to be in [sqrt, additive, isometric]")

    x_trans = x_trans.squeeze()

    return x_trans


def inv_transform_points(x_trans: np.ndarray, transform: str = 'sqrt'):
    """inversely transforms (K-1)- dimensional points fto K dimensions on a (K-1) dimensional polytope

    Parameters
    ----------
    x : np.ndarray of shape (K -1,)
        input points
    transform : str, optional
        inverse of transform is used. needs to be in ['sqrt', 'isometric', 'additive'], by default 'sqrt'

    Returns
    -------
    np.ndarray
       transformed points of shape (K,)
    """
    if x_trans.ndim == 1:
        x_trans = x_trans[np.newaxis, :]

    if transform == 'sqrt':
        assert x_trans.shape[1] == 2, "sqrt transformation only suitbale for 3 dimensional data"
        x = planes_to_coordinates3D(x_trans)
    elif transform == 'additive':
        x = alr_transform_inv(x_trans)
    elif transform == 'isometric':
        x = ilr_transform_inv(x_trans)
    else:
        raise NameError("transform needs to be in [sqrt, additive, isometric]")

    x = x.squeeze()

    return x




if __name__ == "__main__":
    x = np.random.dirichlet([1]*3, 10)
    vertices = get_vertex_pairs_hull(x, transform='isometric')
    print(vertices)

    y = np.array([1, 0, 0])
    print(y.ndim)
    y_trans = transform_points(y)
    print(y_trans.shape)
    y_back = inv_transform_points(y_trans)
    print(y_back.shape)
