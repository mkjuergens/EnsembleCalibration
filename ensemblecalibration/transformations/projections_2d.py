"""
functions for projecting points from a K-simplex on the (K-1) dimensional hyperplane and for ra-transforming back on the (K) dimensional space.
Functions for K=3 are taken from
https://github.com/marcharper/python-ternary
"""

import numpy as np


def permute_point(p: np.ndarray, permutation=None):
    """
    permutes a given point using a predefined permutation. If None, 
    the initial pont is returned.

    Parameters
    ----------
    p : array of shape (3,) or shape (3,1)
        point that is to be projected onto 2 dimensions
    permutation : _type_, optional
        _description_, by default None
    """

    if permutation:
        raise NotImplementedError
    else:
        return p



def project_point2D(p3D: np.ndarray, transform=None):
    """
    Projects a given three dimensional array (point) from a two-dimensional simplex
    onto the resepctive 2-dimensional coordinate plane


    Parameters
    ----------
    p3D : np.ndarray
        _description_
    transform : _type_, optional
        _description_, by default None

    Returns
    -------
    np.array of shape (2,)
        coordinates of the projected point
    """
    permuted = permute_point(p3D, permutation=transform)

    a = permuted.squeeze()[0]
    b = permuted.squeeze()[1]

    # square root of 3 half for scaling # TODO: look at general projections!!
    sqrt3 = np.sqrt(3)/2

    # x and y coordinates
    x = a + b/2
    y = sqrt3*b

    pr_point= np.array([x, y])

    return pr_point

def project_points2D(points3D: np.ndarray, transform=None):
    """project a number of points from a 2-dim simples on the 2-dim plane.


    Parameters
    ----------
    points3D : np.ndrray
       array of shape (n_samples, 3) to be projeted
    transform : _type_, optional
        by default None

    Returns
    -------
    np.ndaray
        projected points
    """

    pr_points = np.apply_along_axis(project_point2D, 1, points3D)

    return pr_points



def plane_to_coordinate3D(p2D: np.ndarray, scale: int = 1):
    """Re-transforms a point form the 2-dimensional plane back to the 3-dimensional space within the simplex.

    Parameters
    ----------
    p2D : np.ndarray
        point to be transformed
    scale: int
        normalized scale of the simlex, i.e. K s.t. (x,y,z) satisfy
            x + y+ z = K
    """
    sqrt3 = np.sqrt(3)/2
    p2D = p2D.squeeze()
    y = p2D[1] / sqrt3
    x = p2D[0] - y/2
    z = scale - x - y

    return np.array([x, y, z])

def planes_to_coordinates3D(points2D: np.ndarray, scale: int = 1):
    """transforms points in the 2-dimensional plane back to 3 dimensions on the 2-simplex.

    Parameters
    ----------
    points2D : np.ndarray of shape (n_points, 2)
        points on 2-dimensional plane
    scale : int, optional
          by default 1

    Returns
    -------
    np.ndarray of shape (n_samples, 3)
        points projected back to dimensions
    """

    points3D = np.apply_along_axis(plane_to_coordinate3D, 1, points2D)

    return points3D


    


class IsometricLogTransform:
    """
    Class for projecting N-dimensional points from the (N-1) dimensional simplex to the (N-1) dimensional Euclidean plane.
    """
    def __init__(self, dim: int, scale: int = 1) -> None:
        """

        Parameters
        ----------
        dim : int
            initial dimension of the points to be transformed
        scale : int, optional
            size of the simplex, by default 1 (unit simplex)
        """
        self.dim = dim
        self.scale = scale


    def project_point(self, p: np.ndarray, transform=None):
        """
        function for projecting a given N-dimensional point in the simplex to the N-1-dimensional Euclidean plane.


        Parameters
        ----------
        p : np.ndarray of shape (self.dim,) or shape (self.dim, 1)
            point to be projected
        transform : _type_, optional
            _description_, by default None # TODO: add transformations here!!
        """
        assert self.dim == len(p), "dimension of input point must match "
        y = []
        for i in range(self.dim -1):
            sqrt_scale = np.sqrt(i*(i+1))
            prod_p = np.prod([p[i] for i in range(self.dim - 1)])
            y_i = (1/sqrt_scale)*np.log(prod_p/(p[self.dim]**2))

            y.append(y_i)

        return y

def inverse_transform(self, p_transformed: np.ndarray, scale: int):
    pass



if __name__== "__main__":
    # test projection
    p = np.array([[0.5,0,0.5]])
    projected_point = project_point2D(p)
    print(projected_point)
    back_transformed_points = plane_to_coordinate3D(projected_point)
    print(back_transformed_points)
    # now for a multiple points
    points = np.array([[.5, .0, 0.5], [1, 0, 0], [0, 1, 0]])
    pr_points = project_points2D(points)
    print(pr_points)








    







