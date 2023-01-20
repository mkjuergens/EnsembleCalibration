import numpy as np
import torch

from numpy.random import dirichlet
from scipy.spatial import ConvexHull
from mhar import walk

from ensemblecalibration.transformations import transform_points, inv_transform_points
from ensemblecalibration.transformations.projections_2d import project_points2D, planes_to_coordinates3D

# goal is to sample predictions P_bar using a matrix P of shape (N, M, K )

def get_polytope_equations(points: np.ndarray, transform: str= 'sqrt'):
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

    x_trans = transform_points(points, transform=transform) #transform points to lower dimensional subspace
    hull = ConvexHull(x_trans) # compute convex hull

    eqs = hull.equations # hyperplane equations of the facets of shape (n_facets, 3)

    A = eqs[:, :-1]
    b = - eqs[:, -1:]

    A = torch.from_numpy(A).type(torch.FloatTensor)
    b = torch.from_numpy(b).type(torch.FloatTensor)

    return A, b


def mhar_sampling(p: np.ndarray, transform: str = 'sqrt', n_samples: int = 100,
                    device: str= 'cpu'):
    """samples uniformly from a polytope by using the Matrix Hit and Run Algoirithm defined in 
    Vazquez et al, 2021.   

    Parameters
    ----------
    p : np.ndarray of shape (M, K) 
        matrix containing predeictions/ arrays whose convex hull 
        defines the respective polytope
    transform : str, optional
        trasnform to be used to map the points on the (K-1) dimensional subspace
        needs to be in [sqrt, isometric, additive], by default 'sqrt'
    n_samples : int, optional
        number of samples to be drawn from the MCMC, by default 100
    device : str, optional
        needs to be in [cpu, cuda], by default 'cpu'

    Returns
    -------
    np.ndarray of shape (n_samples, K)
        array containing generated samples
    """
    # get polytope equations
    A, b = get_polytope_equations(p, transform=transform)
    # use a convex combination to find an inner point of the polytope as a starting point
    x_0 = find_inner_point(p)
    # transform x_0 to (K-1) dimensions using the predefined transformation
    x_0_trans = torch.from_numpy(transform_points(x_0)).view(-1, 1).type(torch.FloatTensor)

    x_sample = walk(z=n_samples, ai=A, bi=b, ae=torch.empty(0), be=torch.empty(0), x_0= x_0_trans,
    T=1, device=device, warm=0, seed=None, thinning=None) # note that x_sample is still in (K-1) dimensions
    # to numpy
    x_sample = x_sample.cpu().numpy()
    # transform it back to K dimensions
    x_out = inv_transform_points(x_sample, transform=transform)

    return x_out

def find_inner_point(P: np.ndarray):
    """function for finding an inner point of a polytope spanned by some opint predictions 

    Parameters
    ----------
    P : np.ndarray
        array of shape (M, K) containing opint estimators which span the polytope

    Returns
    -------
    np.nadrray
        convex combination of the point predictions lying in the polytope
    """

    M, K = P.shape
    l_weight = np.array([1/M]*M)
    new_p = l_weight @ P

    return new_p




if __name__ == "__main__":
    P = np.random.dirichlet([1]*3, size=10)
    x_out = mhar_sampling(P, transform='sqrt')
    print(x_out.shape)