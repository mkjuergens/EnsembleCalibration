"""
helping functions for the calibration tests
"""
import numpy as np

from scipy.stats import dirichlet, multinomial

def sample_m(p: np.ndarray):
    """function which samples form the categorical distribution definded by the input probability vector.

    Parameters
    ----------
    p : np.ndarray  
        probability vector defining the probabilities for each class

    Returns
    -------
    int
        index of the sampled class/label
    """
    try:
        y = np.argmax(multinomial(1,p).rvs(size=1)[0,:])
    except ValueError as e:
        y = np.argmax(p)

    return y

def sample_l(P: np.ndarray):
    """samples convex combinations 

    Parameters
    ----------
    P : np.ndarray of shape (n_samples, n_predictoes, n_classes)
        tensor containing predictions of each ensemble member for each sample

    Returns
    -------
    np.ndarary
        _description_
    """
    # take convex combination of ensemble predictions
    l = dirichlet([1]*P.shape[1]).rvs(size=1)[0,:]
    P_bar = np.matmul(np.swapaxes(P,1,2),l)
        
    return P_bar
            
def dec_bounds(l, u):
    if np.sign(l)!=np.sign(u):
        return 0.0
    elif np.sign(l)==1:
        return l
    else:
        return u

""" Constraint function for aucbtest """
def constr(x):
    return np.sum(x)-1.0

""" Constraint function 1 for COBYLA """
def c1_constr(x):
    return np.sum(x)-1.0

""" Constraint function 2 for COBYLA """
def c2_constr(x):
    return -(np.sum(x)-1.0)