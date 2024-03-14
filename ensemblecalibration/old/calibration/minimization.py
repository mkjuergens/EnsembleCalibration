""" 
This script contains several functions for solving the minimization problem defined in Mortier at al, 2022 used to find the optimal convex
combination of point predictions such that the calibration measure (defined here as objective in the input parameters) is minimized.
 
 Author: Mira Juergens
 Date: March 2023
"""


import numpy as np 
from scipy.optimize import minimize

from pyswarm import pso

from ensemblecalibration.calibration.calibration_estimates import (
    confece_obj_new,
    classece_obj_new,
    skce_ul_obj_lambda
)
from ensemblecalibration.calibration.calibration_estimates.distances import tv_distance
from ensemblecalibration.calibration.helpers import (
    c1_constr,
    c2_constr,
    c1_constr_flat,
    c2_constr_flat,
    constr_pyswarm,
)

def solve_cobyla1D(
    P: np.ndarray, y: np.ndarray, params: dict, enhanced_output: bool = False
):
    """returns the vector of weights which results in a convex combination of predictors with the minimal calibration error.
        Here, the weights do not depend on the instances, therefore resulting in a one-dimensional array.

    Parameters
    ----------
    P : np.ndarray
        matrix of point predictions for each instance and predcitor
    y : np.ndarray
        labels
    params : dict
        dictionary of test parameters
    """

    # inittial gues: equal weights
    l = np.array([1 / P.shape[1]] * P.shape[1])
    bnds = tuple([tuple([0, 1]) for _ in range(P.shape[1])])
    cons = [{"type": "ineq", "fun": c1_constr}, {"type": "ineq", "fun": c2_constr}]
    # bounds must be included as constraints for COBYLA
    for factor in range(len(bnds)):
        lower, upper = bnds[factor]
        lo = {"type": "ineq", "fun": lambda x, lb=lower, i=factor: x[i] - lb}
        up = {"type": "ineq", "fun": lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(lo)
        cons.append(up)
    solution = minimize(
        params["obj_lambda"], l, (P, y, params), method="COBYLA", constraints=cons
    )
    l = np.array(solution.x)
    minstat = params["obj_lambda"](l, P, y, params)
    if enhanced_output:
        return l, minstat
    else:
        return l


def solve_cobyla2D(
    P: np.ndarray, y: np.ndarray, params: dict, enhanced_output: bool = False
):
    """returns the vector of weights which results in a convex combination of predictors with the minimal calibration error.
        Here, the weights do depend on the instances, therefore resulting in a two-dimensional array.

    Parameters
    ----------
    P : np.ndarray of shape (N, M, K)
        matrix of point predictions for each instance and predcitor
    y : np.ndarray
        labels
    params : dict
        dictionary of test parameters

    Returns
    -------
    _type_
        _description_
    """
    # intial guess; must be of shape (N, M)
    l_0 = np.zeros((P.shape[0], P.shape[1]))
    l_0[...] = np.array(
        [1 / P.shape[1]] * P.shape[1]
    )  # set every element of l to same value in the beginnning
    l_0 = l_0.flatten()  # flatten vector for the minimization problem
    bnds = tuple(
        [tuple([0, 1]) for _ in range(P.shape[1] * P.shape[0])]
    )  # each of the N*M entries of the matrix
    # needs to be between 0 and 1
    cons = [
        {"type": "ineq", "fun": lambda x: c1_constr_flat(x, n_rows=P.shape[0])},
        {"type": "ineq", "fun": lambda x: c2_constr_flat(x, n_rows=P.shape[0])},
    ]

    for factor in range(len(bnds)):
        lower, upper = bnds[factor]
        lo = {"type": "ineq", "fun": lambda x, lb=lower, i=factor: x[i] - lb}
        up = {"type": "ineq", "fun": lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(lo)
        cons.append(up)

    # TODO: adjust objectives to x dependecy argument
    solution = minimize(
        params["obj_lambda"], l_0, (P, y, params, True), method="COBYLA", constraints=cons
    )
    l = np.array(solution.x)
    minstat = params["obj_lambda"](l, P, y, params, True)

    if enhanced_output:
        return l, minstat

    else:
        return l


def solve_neldermead1D(
    P: np.ndarray, y: np.ndarray, params: dict, enhanced_output: bool = False
):
    """returns the vector of weights resulting in convex combination of predictors with the minimal calibration error
    using the Nelder Mead method, without x dependency of the weights.

    Parameters
    ----------
    P :  np.ndarray of shape (N, M, K)
        tensor containing the probabilistic point predictions for each instance and each predictor
    y : np.ndarray of shape (N,)
        labels
    params : dict
        dictionary of the test parameters
    enhanced_output : bool, optional
        whether to output only optimal weights or also resulting objective value, by default False

    Returns
    -------
    _type_
        _description_
    """

    # intial guess: equal weights
    l = np.array([1 / P.shape[1]] * P.shape[1])
    # lower and upper bounds
    bnds = tuple([tuple([0, 1]) for _ in range(P.shape[1])])
    # constraints: here, equality constraints are used
    cons = {"type": "eq", "fun": c1_constr}

    solution = minimize(
        params["obj"],
        l,
        (P, y, params),
        method="Nelder-Mead",
        bounds=bnds,
        constraints=cons,
    )

    l = np.array(solution.x)
    minstat = params["obj_lambda"](l, P, y, params)

    if enhanced_output:
        return l, minstat

    else:
        return l


def solve_neldermead2D(
    P: np.ndarray, y: np.ndarray, params: dict, enhanced_output: bool = False
):
    """returns the vector of weights resulting in convex combination of predictors with the minimal calibration error
    using the Nelder Mead method, wit x dependency of the weights.

    Parameters
    ----------
    P :  np.ndarray of shape (N, M, K)
        tensor containing the probabilistic point predictions for each instance and each predictor
    y : np.ndarray of shape (N,)
        labels
    params : dict
        dictionary of the test parameters
    enhanced_output : bool, optional
        whether to output only optimal weights or also resulting objective value, by default False

    Returns
    -------
    _type_
        _description_
    """

    # intial guess; must be of shape (N, M)
    l_0 = np.zeros((P.shape[0], P.shape[1]))
    # set every element of l to same value in the beginnning
    l_0[...] = np.array([1 / P.shape[1]] * P.shape[1])
    # flatten vector for the minimization problem
    l_0 = l_0.flatten()
    # lower and upper bounds: here we have N*M entries of the resulting matrix, hence also this many bounds
    bnds = tuple([tuple([0, 1]) for _ in range(P.shape[1] * P.shape[0])])
    # constraints: equality constraints
    cons = {"type": "eq", "fun": lambda x: c1_constr_flat(x, n_rows=P.shape[0])}

    solution = minimize(
        params["obj"],
        l_0,
        (P, y, params),
        method="Nelder-Mead",
        bounds=bnds,
        constraints=cons,
    )

    l = np.array(solution.x)
    minstat = params["obj_lambda"](l, P, y, params)

    if enhanced_output:
        return l, minstat

    else:
        return l


def solve_pyswarm(
    P: np.ndarray,
    y: np.ndarray,
    params: dict,
    enhanced_output: bool = False,
    swarm_size: int = 1000,
    maxiter: int = 100,
):
    # lower bounds: list of lower bounds for all variables
    lb = np.zeros(P.shape[0] * P.shape[1])
    # upper bounds: list of upper bounds for all variabels
    ub = np.ones(P.shape[0] * P.shape[1])

    constr = lambda x, P, y, params: constr_pyswarm(x, P.shape[0], (P, y, params))

    lopt, fopt = pso(
        params["obj_lambda"],
        lb=lb,
        ub=ub,
        f_ieqcons=constr,
        args=(P, y, params),
        maxiter=maxiter,
        swarmsize=swarm_size,
    )

    if enhanced_output:
        return lopt, fopt

    else:
        return lopt


def solve_minimization(obj, l0, P, y):
    pass


if __name__ == "__main__":
    P = np.random.dirichlet([1] * 3, size=(100, 10))
    y = np.random.randint(2, size=100)
    config = {"obj_lambda": skce_ul_obj_lambda, "dist": tv_distance, "sigma": 0.1,
               "take_square": False}
    l_1 = solve_cobyla2D(P, y, config)
    print(l_1)
