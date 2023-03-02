import numpy as np
from scipy.optimize import minimize
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.single.global_best import GlobalBestPSO

from ensemblecalibration.calibration.test_objectives import confece_obj_new
from ensemblecalibration.calibration.helpers import constraint1_new, constraint2_new, c1_constr_flat, c2_constr_flat



def solve_cobyla(P: np.ndarray, y: np.ndarray, params: dict):
    """_summary_

    Parameters
    ----------
    P : np.ndarray
        _description_
    y : np.ndarray
        _description_
    params : dict
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # intial guess; must be of shape (N, M)
    l_0 = np.zeros((P.shape[0], P.shape[1]))
    l_0[...] = np.array([1/P.shape[1]]*P.shape[1]) # set every element of l to same value in the beginnning
    l_0 = l_0.flatten() # flatten vector for the minimization problem
    bnds = tuple([tuple([0,1]) for _ in range(P.shape[1]*P.shape[0])]) # each of the N*M entries of the matrix
                                                                        # needs to be between 0 and 1
    cons = [{'type': 'ineq', 'fun': lambda x: c1_constr_flat(x, n_rows=P.shape[0])}, {'type': 'ineq', 'fun':
            lambda x: c2_constr_flat(x, n_rows=P.shape[0]) }]
    
    for factor in range(len(bnds)):
        lower, upper = bnds[factor]
        lo = {'type': 'ineq',
                'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        up = {'type': 'ineq',
                'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(lo)
        cons.append(up)

    solution = minimize(params["obj"],l_0,(P, y, params),method='COBYLA',constraints=cons)
    l = np.array(solution.x)

    return l

def solve_pyswarm(P: np.ndarray, y: np.ndarray, params: dict):

    pass

def solve_minimization(obj, l0, P, y):
    
    # initial guess
    l = np.array()
    pass

if __name__ == "__main__":

    P = np.random.dirichlet([1]*3, size=(100,10))
    y = np.random.randint(2, size=100)
    config = {"obj": confece_obj_new, "n_bins":10}
    l = solve_cobyla(P, y, config)

    print(l)


    

