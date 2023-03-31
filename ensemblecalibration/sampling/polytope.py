import torch
import numpy as np

from scipy.spatial import ConvexHull

from ensemblecalibration.transformations import transform_points, inv_transform_points


def get_polytope_equations(points: np.ndarray, transform: str = "sqrt"):
    """function exrtacting the hyperplane inequality constraints
    given points in the K-dim space from a (K-1))-simplex

    Parameters
    ----------
    points : np.ndarray of shape (n_points, K)
        points from a (K-1)-simplex
    transform: str, Options: ['isometric', 'additive', 'sqrt]
        transformation to be used to map the points on the lower dimensional
        subspace. Note that 'sqrt' can be only used for the 3 dimensional space.

    Returns
    -------
    tensors A of shape (n_facets, K-1)
            b of shape (n_facets, 1)

        defining the polytope P in a way s.t.

            P = {x in R^(K-1)): Ax <= b}
    """

    x_trans = transform_points(
        points, transform=transform
    )  # transform points to lower dimensional subspace
    hull = ConvexHull(x_trans)  # compute convex hull

    eqs = hull.equations  # hyperplane equations of the facets of shape (n_facets, 3)

    A = eqs[:, :-1]
    b = -eqs[:, -1:]

    A = torch.from_numpy(A).type(torch.FloatTensor)
    b = torch.from_numpy(b).type(torch.FloatTensor)

    return A, b
