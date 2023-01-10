"""Module fo blabalaa. """
import numpy as np
from scipy.optimize import linprog


def is_calibrated(P: np.ndarray, p_hat: np.ndarray):
    """function which returns whether given a matrix of ensemble predictions P,
        a new prediction p_hat lies in the spanned polytope of the predci

    Parameters
    ----------
    P : np.ndarray of shape (M, K)
        ensemble of predictors for a feature, in matrix notation
    p : np.ndarray of shape (K,) 
        prediction that is to be tested whether it lies in the polytope

    Returns
    -------
    _type_
        _description_
    """
    M = len(P)
    K = len(p_hat)
    c = np.zeros(M)
    A = np.r_[P.T, np.ones((1, M))]
    b = np.r_[p_hat, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)

    return lp.success

if __name__ == "__main__":
    # test function for some random data
    n_classes = 10
    n_predictions = 10
    P_1 = np.random.random((n_predictions, n_classes))
    p_new = np.random.random((n_classes))

    print(is_calibrated(P_1, p_new))