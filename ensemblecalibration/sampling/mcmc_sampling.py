import sys, os
import time

import numpy as np
import torch

from numpy.random import dirichlet
from scipy.spatial import ConvexHull
from mhar import walk

from ensemblecalibration.transformations import transform_points, inv_transform_points
from ensemblecalibration.transformations.projections_2d import project_points2D, planes_to_coordinates3D
from ensemblecalibration.sampling.lambda_sampling import multinomial_label_sampling

# goal is to sample predictions P_bar using a matrix P of shape (N, M, K )

# first we define blocks for disabling/enabling printing for convenience

def block_print():
    """
    function for blocking print statements returned by any function.
    """
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    """
    fucnntion for (re)enabling print statements of functions.
    """
    sys.stdout = sys.__stdout__

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
                    device: str= 'cpu', random_start_point: bool = True):
    """samples uniformly from a polytope by using the Matrix Hit and Run Algoirithm defined in 
    Vazquez et al, 2021.   

    Parameters
    ----------
    p : np.ndarray of shape (M, K) 
        matrix containing predeictions/ arrays whose convex hull 
        defines the respective polytope
    transform : str, optional
        transform to be used to map the points on the (K-1) dimensional subspace
        needs to be in [sqrt, isometric, additive], by default 'sqrt'
    n_samples : int, optional
        number of samples to be drawn from the MCMC, by default 100
    device : str, optional
        needs to be in [cpu, cuda], by default 'cpu'
    random_start_point: boolean, optional
        whether to sample the start point by using a random convex combination

    Returns
    -------
    np.ndarray of shape (n_samples, K)
        array containing generated samples
    """
    # get polytope equations
    A, b = get_polytope_equations(p, transform=transform)
    # use a convex combination to find an inner point of the polytope as a starting point
    if random_start_point:
        x_0 = find_random_inner_point(p) # UPDATE: use random inner point insteas of fixed inner point
    else:
        x_0 = find_inner_point(p)
    # transform x_0 to (K-1) dimensions using the predefined transformation
    x_0_trans = torch.from_numpy(transform_points(x_0, transform=transform)).view(-1, 1).type(torch.FloatTensor)

    # block the print statements of the algorithm
    block_print()
    x_sample = walk(z=n_samples, ai=A, bi=b, ae=torch.empty(0), be=torch.empty(0), x_0= x_0_trans,
                T=1, device=device, warm=0, seed=None, thinning=None, check=False) # note that x_sample is still in (K-1) dimensions

    # enable printing again
    enable_print()
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
    np.ndarray
        convex combination of the point predictions lying in the polytope
    """

    M, K = P.shape
    l_weight = np.array([1/M]*M)
    new_p = l_weight @ P

    return new_p

def find_random_inner_point(P: np.ndarray):
    """function for (randomly) sampling an inner points of the polytope given by a number of points.

    Parameters
    ----------
    P : np.ndarray of shape(n_predictors, n_classes)
        containing point predcitions which span the polytope

    Returns
    -------
    np.ndaray of shape (n_classes,)
        _description_
    """

    M, K = P.shape
    # sample weights from dirichlet distribution
    weights = dirichlet([1]*M, size=1).squeeze()
    new_p = weights @ P

    return new_p


def mhar_sampling_p(P: np.ndarray, transform: str = 'sqrt'):
    """Given an array of shape (N, M, K), this function samples for every n in 1, ..., N

    Parameters
    ----------
    P : np.ndarray of shape (N, M, K)
        array containing the point predictions for every of the M predcitors and every sample
    transform : str, optional
        needs to be in [sqrt, additive ], by default 'sqrt'

    Returns
    -------
    np.ndarray of shape (N, K)
        array of sampled predictions for each sample n in 1, ..., N
        
    """
    P_hat = np.zeros((P.shape[0], P.shape[2])) 
    for i in range(P.shape[0]):
        # sample one sample from mhar algorithm
        x_sample = mhar_sampling(P[i], transform=transform, n_samples=1)
        P_hat[i] = x_sample

    return P_hat

if __name__ == "__main__":
    P = np.random.dirichlet([1]*3, size=10)
    x_out = mhar_sampling(P, transform='sqrt')
    # now for real P
    P = np.random.dirichlet([1]*3, size=(100, 10))
    print(P.shape)
    P_hat = np.zeros((P.shape[0], 3))
    for i in range(P.shape[0]):
        x_sample = mhar_sampling(P[i], transform='sqrt', n_samples=1)
        P_hat[i] = x_sample
    print(P_hat.shape)
    t_0 = time.time()
    P_hat = mhar_sampling_p(P)
    t_1 = time.time()
    print(f'Time for sampling 1000 times: {t_1-t_0}')
    




    