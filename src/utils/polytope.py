from typing import Union

import numpy as np
import torch
from scipy.optimize import linprog


def is_calibrated(P: np.ndarray, p_hat: np.ndarray):
    """function which returns whether given a matrix of ensemble predictions for one instance P,
        a new prediction p_hat lies in the spanned polytope spanned by the predictions.

    Parameters
    ----------
    P : np.ndarray of shape (n_predictors, n_classes)
        ensemble of predictors for a feature, in matrix notation
    p : np.ndarray of shape (K,)
        prediction for one instance that is to be tested whether it lies in the polytope

    Returns
    -------
    boolean
    """
    M = len(P)
    c = np.zeros(M)
    A = np.r_[P.T, np.ones((1, M))]
    b = np.r_[p_hat, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)

    return lp.success


def is_in_convex_hull(P: np.ndarray, p_hat: np.ndarray, tolerance: float = -1e-9):

    M, K = P.shape
    c = np.zeros(M)
    bounds = [(0, None) for _ in range(M)]
    A_eq = np.ones((1, M))
    b_eq = np.array(1)

    res = linprog(
        c, A_ub=P.T, b_ub=p_hat, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="simplex"
    )

    # return res.fun >= tolerance and all(x >= tolerance for x in res.x)
    return res.success


def find_boundary(P: np.ndarray, p_0: np.ndarray, p_c: np.ndarray):
    """function for finding the convex combination between points p_0
    and p_c with the largest weigt coefficient, s.t. the convex combination
    p_b is still in the polytope spanned by the matrix P. Here, p_0 is
    the mean within the polytope and p_c is a randomly chosen corner.

    Parameters
    ----------
    P :  of shape (n_predictors, n_classes)
        matrix defining the polytope of all convex combinations of
         the predictors.
    p_0 : np.ndarray
        _description_
    p_c : np.ndarray
        _description_

    Returns
    -------
    _type_
        _description_
    """

    l_arr = np.linspace(0, 1, 100)
    # get all convex combinations between p_0 and p_c
    L = np.stack([l * p_c + (1 - l) * p_0 for l in l_arr])
    # determine boundary
    b_i = 0
    for i in range(len(L)):
        if not is_calibrated(P, L[i, :]):
            b_i = i - 1
            break

    p_b = L[b_i, :]

    return p_b

# def get_boundary(P: torch.tensor, p_mean: torch.tensor, p_c: torch.tensor):
#     """function for finding the convex combination between points p_mean
#     and p_c with the largest weigt coefficient, s.t. the convex combination
#     p_b is still in the polytope spanned by the matrix P. Here, p_mean is
#     the mean within the polytope and p_c is a randomly chosen corner.

#     Parameters
#     ----------
#     P : torch.tensor of shape (n_predictors, n_classes)
#         matrix defining the polytope of all convex combinations of
#          the predictors.
#     p_mean : torch.tensor
#         mean of the dirichlet distribution from which the predictions are sampled from
#     p_c : torch.tensor
#         corner within the simplex of the predictions

#     Returns
#     -------
#     torch.tensor
#         convex combination of p_mean and p_c that lies on the boundary of the polytope
#     """

#     l_arr = torch.linspace(0, 1, 100)
#     # get all convex combinations between p_mean and p_c
#     L = torch.stack([l * p_c + (1 - l) * p_mean for l in l_arr])
#     # determine boundary
#     b_i = 0
#     for i in range(len(L)):
#         if not is_calibrated(P.numpy(), L[i, :].numpy()):
#             b_i = i - 1
#             break

#     p_b = L[b_i, :]

#     return p_b

def get_boundary(
    P: Union[torch.Tensor, np.ndarray],
    p_0: Union[torch.Tensor, np.ndarray],
    p_c: Union[torch.Tensor, np.ndarray],
):
    """
    Optimized function for finding the convex combination between points p_0
    and p_c with the largest weight coefficient, such that the convex combination
    p_b is still in the polytope spanned by the matrix P.

    Parameters
    ----------
    P : torch.Tensor or np.ndarray
        The matrix defining the polytope of all convex combinations of the predictors.
        It should be of shape (N, M, K) or (M, K).
    p_0 : torch.Tensor or np.ndarray
        The mean point within the polytope.
    p_c : torch.Tensor or np.ndarray
        A randomly chosen corner (one-hot encoded vector of the respective class).

    Returns
    -------
    p_b : torch.Tensor or np.ndarray
        The convex combination on the boundary.
    """
    # Convert inputs to NumPy arrays if necessary
    if isinstance(P, torch.Tensor):
        P = P.cpu().numpy()
    if isinstance(p_0, torch.Tensor):
        p_0 = p_0.cpu().numpy()
    if isinstance(p_c, torch.Tensor):
        p_c = p_c.cpu().numpy()

    l_arr = np.linspace(0, 1, 100)

    if P.ndim == 3:
        # If P is (N, M, K), we process each batch
        N, M, K = P.shape
        L = (
            np.expand_dims(l_arr, axis=(1, 2)) * p_c[:, None, :]
            + (1 - np.expand_dims(l_arr, axis=(1, 2))) * p_0[:, None, :]
        )
        calibrated = np.array(
            [is_calibrated(P[n], L[n, i]) for n in range(N) for i in range(len(L[n]))]
        ).reshape(N, -1)
        b_idx = np.argmax(~calibrated, axis=1) - 1
        b_idx[b_idx < 0] = 0  # Ensure no negative index
        p_b = np.array([L[n, b_idx[n]] for n in range(N)])

    elif P.ndim == 2:
        # If P is (M, K), process a single polytope
        L = np.array([l * p_c + (1 - l) * p_0 for l in l_arr])
        calibrated = np.array([is_calibrated(P, L[i]) for i in range(len(L))])
        b_idx = np.argmax(~calibrated) - 1
        if b_idx < 0:
            b_idx = 0  # Ensure no negative index
        p_b = L[b_idx]

    else:
        raise ValueError("P must be either (N, M, K) or (M, K)")

    # Convert back to torch.Tensor if required
    if isinstance(P, torch.Tensor):
        p_b = torch.tensor(p_b, device=p_0.device)

    return p_b
