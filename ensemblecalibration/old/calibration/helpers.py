"""
helping functions for the calibration tests
"""
from typing import Optional
import numpy as np

from scipy.stats import dirichlet, multinomial

def sort_and_reject(p_vals: np.ndarray, alpha: float, method: str = "hochberg"):
    """ffunction for sorting a list of p values and using a given method of corredction to reject 
    all p values below a critical value.

    Parameters
    ----------
    p_vals : np.ndarray
        list of p-values of all hypothesis tests
    alpha : float or list
        significance level
    method: str
        method of correction, by default "hochberg".

    Returns
    -------
    decs
        list of decisions for each hypothesis test
    """
    # sort in descending order
    p_vals_sorted = np.sort(p_vals)
    sort_idx = np.argsort(p_vals)
    if method == "hochberg":
        rej_coef = [(k/len(p_vals_sorted))* alpha for k in range(1, len(p_vals_sorted)+1)]
    elif method is None:
        rej_coef = [alpha]*len(p_vals_sorted)
    else:
        raise NotImplementedError(f"method {method} not implemented")
    rej_coef = np.array(rej_coef)
    # find first index where p_val <= rej_coef
    idx = np.argmax(p_vals_sorted > rej_coef)
    # reject all p_vals at index idx and above
    decs = np.zeros(len(p_vals_sorted))
    decs[:idx] = 1

    return decs

def sort_and_reject_alpha(p_vals: np.ndarray, alpha: list, method: str = "hochberg"):

    decs_alpha = np.zeros((len(alpha), len(p_vals)))
    for i, a in enumerate(alpha):
        decs_alpha[i] = sort_and_reject(p_vals, a, method=method)
    
    return decs_alpha


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
    P : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing predictions of each ensemble member for each sample

    Returns
    -------
    np.ndarray
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

def constr_eq(x):
    """equality constraint for the optimization problem used e.g. in the Nelder-Mead 
    method.
    
    """
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
    l = l_flat.reshape(n_rows, -1) # reshape to matrix of shape (N, M)
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

def constr_pyswarm(l_flat: np.ndarray, n_rows: int, *args):

    assert len(l_flat) % n_rows == 0, "length of array must be a multiple of the desired rows"
    l = l_flat.reshape(n_rows, -1)
    # e = np.ones(l.shape[1])
    prod = np.sum(l, axis=1)
    cons = []
    for i in range(l.shape[0]):
        diff_1 = prod[i] - 1.0
        diff_2 = -(prod[i] - 1.0)
        cons.append(diff_1)
        cons.append(diff_2)

    return cons

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

def emp_dist_min_objective(params_tests, min_fct, experiment, n_iters: int = 100):
    """test for estimating the empirical distribution of the approximated minimal calibration 
    measure using a predefined method and calibration objective 
    Parameters
    ----------
    params_tests : dict
        test parameters
    min_fct : _type_
        function for solving the minimization problem. Takes as input arguments the tensor of 
        point predictions P, the array of labels y the test parameters and an optional boolean 
        whether the minimal value shall also be output
    experiment : function, 
        experiment which yields a tensor P and labels y used for calculating the minima, 
        by default experiment_h0
    n_iters : int, optional
        number of times the distribution is sampled from and the minima are calculated, by default
        100

    Returns
    -------
    dict
        dictionary containing 
    """

    list_stats = {}
    for test in params_tests:
        list_stats[test] = []
    for i in range(n_iters):
        print(f'start {i}-th iteration...')
        P, y = experiment(N=100, M=10, K=10, u=0.01)
        for test in params_tests:
            conf = params_tests[test]
            weights_l, min_stat = min_fct(P, y, conf["params"], enhanced_output=True)
            list_stats[test].append(min_stat)
    return list_stats


if __name__ == "__main__":
    p_vals = np.random.uniform(0,1,100)
    alphas = [0.05, 0.1, 0.15]
    decs = sort_and_reject_alpha(p_vals, alphas)
    print(np.min(decs, axis=1))
