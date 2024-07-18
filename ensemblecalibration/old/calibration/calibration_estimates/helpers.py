import sys
sys.path.append("../")
import numpy as np

from calibration_estimates.distances import tv_distance


def calculate_pbar(
    weights_l: np.ndarray, P: np.ndarray, reshape: bool = True, n_dims: int = 2
):
    """calculate convex combination of a weight matrix of shape (N*M,) or (M,) and a tensor of point predictions
        of shape (N, M, K) such that the new matrix contains a point predictions for each instance and is
        of shape (N, K).
    Parameters
    ----------
    weights_l : np.ndarray
        weight matrix of shape (N*M, ) or of shape (M,)
    P : np.ndarray
        tensor of point predcitions for each instance for each predcitor, shape (N,M,K)
    reshape: boolean
        whether vector of weights shall be reshaped to matrix form
    n_dims: int, must be in {1,2}
        number of dimensions of the weight vector/matrix. If 2, we have instance-wise dependency of
        the convex combination P_bar

    Returns
    -------
    np.ndarray
        matrix containinng one new prediction for each instance, shape (N, K)
    """

    if n_dims == 2:
        n_rows = P.shape[0]
        if reshape:
            assert (
                len(weights_l) % n_rows == 0
            ), " weight vector needs to be a multiple of the "
            "number of rows"
            weights_l = weights_l.reshape(n_rows, -1)

        assert (
            weights_l.shape[0] == P.shape[0]
        ), " numer of samples need to be the same for P and weights_l"
        assert (
            weights_l.shape[1] == P.shape[1]
        ), " numer of ensemble members need to be the same for P and weights_l"

        P_bar = (weights_l[:, :, np.newaxis] * P).sum(
            -2
        )  # sum over second axis to get diagonal elements

    elif n_dims == 1:
        P_bar = np.matmul(np.swapaxes(P, 1, 2), weights_l)

    return P_bar
