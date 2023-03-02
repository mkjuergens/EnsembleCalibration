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

def constraint1_new(l: np.ndarray):
    """constraint 1 in the optimization problem ensuring that the coefficients of the weights sum to
    1.

    Parameters
    ----------
    l : np.ndarray
        matrix of the weights of shape (N, M)

    Returns
    -------
    np.ndarray
    """

    # weight vector now consists of point predictions for each sample
    N, M = l.shape 
    e = np.ones(l.shape[1])
    prod = l @ e
    diff = prod - np.ones(l.shape[0])

    return diff

def c1_constr_flat(l_flat: np.ndarray, n_rows: int):
    """constraint 1 in the optimization problem using the flattened array.

    Parameters
    ----------
    l_flat : np.ndarray
        flattened matrix of weights
    n_rows : int
        number of rows (i.e. number of instances) used to reshape the matrix 
    """

    assert len(l_flat) % n_rows == 0, "length of array must be a multiple of the desired rows"
    l = l_flat.reshape(n_rows, -1)
    e = np.ones(l.shape[1])
    prod = l @ e
    diff = prod - np.ones(l.shape[0])

    return diff



def constraint2_new(l:np.ndarray):
    """constraint 2 in the optimization problem ensuring that the weights sum to 1.

    Parameters
    ----------
    l : np.ndarray
       matrix of shape (N, M)

    Returns
    -------
    np.ndarray
    """

    N, M, = l.shape
    e = np.ones(N)
    prod = l @ e
    diff = - (prod - np.ones(l.shape[0]))

    return diff

def c2_constr_flat(l_flat: np.ndarray, n_rows: int):

    assert len(l_flat) % n_rows == 0, "length of array must be a multiple of the desired rows"
    l = l_flat.reshape(n_rows, -1)
    e = np.ones(l.shape[1])
    prod = l @ e
    diff = - (prod - np.ones(l.shape[0]))

    return diff

def init_pmatrix(N: int, M: int, K: int, u: float = 0.01):
    """Initialises the matrix P containing M predictors evaluated on N instances. It is constructed by sampl

    Parameters
    ----------
    N : int
        _description_
    M : int
        _description_
    K : int
        _description_
    u : float, optional
        _description_, by default 0.01

    Returns
    -------
    _type_
        _description_
    """

    P = []
    a0 = [1/K]*K # parameter of the Dirichlet distribution
    for n in range(N):
        p0 = np.random.dirichlet(a0,1)[0,:]
        a = (K*p0)/u # u serves as a parameter for scaling the "spread" of the distribution
        Pm = np.random.dirichlet(a, M)
        P.append(Pm)
    
    P = np.stack(P)

    return P

if __name__ == "__main__":
    P = init_pmatrix(100, 10, 10)




