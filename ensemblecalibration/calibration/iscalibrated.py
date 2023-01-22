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
    boolean 
    """
    M = len(P)
    c = np.zeros(M)
    A = np.r_[P.T, np.ones((1, M))]
    b = np.r_[p_hat, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)

    return lp.success

def find_boundary(P:np.ndarray, p_0: np.ndarray, p_c: np.ndarray):
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
    L = np.stack([l*p_c+(1-l)*p_0 for l in l_arr])
    # determine boundary
    b_i = 0
    for i in range(len(L)):
        if not is_calibrated(P, L[i, :]):
            b_i = i -1
            break

    p_b = L[b_i, :]

    return p_b

if __name__ == "__main__":
    # test function for some random data
    n_classes = 10
    n_predictions = 10
    P_1 = np.random.random((n_predictions, n_classes))
    p_new = np.random.random((n_classes))

    print(is_calibrated(P_1, p_new))

