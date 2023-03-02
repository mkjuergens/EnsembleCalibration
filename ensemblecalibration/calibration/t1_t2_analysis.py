import sys, os
import time
import argparse
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.stats import multinomial

sys.path.append('../..')
from ensemblecalibration.calibration.iscalibrated import is_calibrated
from ensemblecalibration.calibration.config import config_tests_reduced, config_tests_new

def get_ens_alpha(K, u, a0):
    p0 = np.random.dirichlet(a0,1)[0,:]
    
    return (K*p0)/u

def getBoundary(P, mu, yc):
    l_arr = np.linspace(0,1,100)
    # get convex combinations between yc and mu
    L = np.stack([l*yc+(1-l)*mu for l in l_arr])
    # determine boundary
    bi = 0
    for i in range(len(L)):
        if not is_calibrated(P, L[i,:]):
            bi = i-1
            break
    yb = L[bi,:]
    
    return yb

def experiment_h0(N: int, M: int, K: int, u: float):
    """yields the predictive value tensor as well as the labels for the experiment in Mortier
    et al, where the null hypothesis that the ensemble model is calibrated is true.

    Parameters
    ----------
    N : int
        _description_
    M : int
        _description_
    K : int
        _description_
    R : int
        _description_
    u : float
        _description_

    Returns
    -------
    _type_
        _description_
    """

    l = np.random.dirichlet([1/M]*M,1)[0,:]
    L = np.repeat(l.reshape(-1,1),K,axis=1)
    P, y = [], []
    for n in range(N):
        a = get_ens_alpha(K, u, [1/K]*K)
        while np.any(a<=0):
            a = get_ens_alpha(K, u, [1/K]*K)
        Pm = np.random.dirichlet(a, M)
        Pbar = np.sum(Pm*L, axis=0)
        # sample instance
        try: 
            yl = np.argmax(multinomial(1,Pbar).rvs(size=1),axis=1)[0]
        except ValueError as e:
            yl = np.argmax(Pbar)
        P.append(Pm)
        y.append(yl)
    P = np.stack(P)
    y = np.array(y)

    return P, y

def experiment_h1(N: int, M: int, K: int, u: float, random: bool = False):
    """returns P tensor and array of labels for the setting in Mortier et al where the null
    hypothesis is false

    Parameters
    ----------
    N : int
        number of instances
    M : int
        number of predictors
    K : int
        number of different classes
    R : int
        _description_
    u : float
        parameter of the dirichlet distribution describing the "spread" or "uncertainty"
    random : bool, optional
        whether the corner is randomly chosen or the closest corner is chosen, by default False

    Returns
    -------
    _type_
        _description_
    """

    P, y = [], []
    for n in range(N):
        a = get_ens_alpha(K, u, [1/K]*K)
        while np.any(a<=0):
            a = get_ens_alpha(K, u, [1/K]*K)
        mu = (a*u)/K
        if M==1:
            Pm = mu.reshape(1,-1)
        else:
            Pm = np.random.dirichlet(a, M)
        # pick class and sample ground-truth outside credal set 
        if not random:
            c = np.argmax(mu)
        else:
            c = np.random.randint(0, K, 1)[0]
        yc = np.eye(K)[c,:]
        # get boundary
        if M==1:
            yb = mu
        else:
            yb = getBoundary(Pm, mu, yc)
        # get random convex combination
        l = np.random.rand(1)[0]
        l = l*yc+(1-l)*yb
        # sample instance
        try: 
            yl = np.argmax(multinomial(1,l).rvs(size=1),axis=1)[0]
        except ValueError as e:
            yl = np.argmax(l)
        P.append(Pm)
        y.append(yl)
    P = np.stack(P)
    y = np.array(y)

    return P, y



def _simulation_h0(tests, N: int, M: int, K: int, R: int, u: float, alpha: float):
    """Simulation of the test if the Null Hypothesis is true.

    Parameters
    ----------
    tests : dictionary
        dictionary with tests and parameters
    N : int
        nuber of features
    M : int
        number of point predictors
    K : int
        number of different classes to predcit the probability for
    R : int
        number of resamplings/different datasets
    u : float
        parameter which controls the uncertainty/spread in the datasets
    alpha : float
        confidence level of the test

    Returns
    -------
    dictionary
        dictionary containing results
    """

    results = {}
    for test in tests:
        results[test] = np.zeros(len(alpha))
    times_tests = np.zeros((R, N, len(tests)))
    for r in tqdm(range(R)):
        P, y = experiment_h0(N, M, K, u)
        for count, test in enumerate(tests):
            results[test] += np.array(tests[test]["test"](P, y, alpha, tests[test]["params"]))
    

    for test in tests:
        # calculate mean
        results[test] = results[test]/R 
        
    return results


def _simulation_ha(tests, N: int, M: int, K: int, R: int, u: float, alpha: float, 
                    random: bool = False):
    """Simulation of the test in a setting where the alternative hypothesis is true.

    Parameters
    ----------
    tests : dictionary
        _description_
    N : int
        number of samples for each dataset
    M : int
        number of point predictors
    K : int
        number of (different) classes
    R : int
        number of resamplings/different datasets
    u : float
        parameter which controls the uncertainty/ spread in the sampled datasets
    alpha : float
        confidence level of the tests
    random: bool
        whether to randomly chose the corner the outside distribution is sampled from

    Returns
    -------
    dictionary
        results of the test
    """
    results = {}
    for test in tests:
        results[test] = np.zeros(len(alpha))
    
    for r in tqdm(range(R)):
        P, y = experiment_h1(N, M, K, u, random=random)
        for test in tests:
            results[test] += np.array(tests[test]["test"](P, y, alpha, tests[test]["params"]))

    
    for test in tests:
        results[test] = results[test]/R
        
    return results

def main_t1_t2(args, config=config_tests_new, test_h1: bool = True, results_dir: str = 'results'):

    tests = config
    results = []
    alpha = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78, 0.87, 0.95]
    N = args.N
    M = args.M
    K = args.K
    u = args.u
    R = args.R
    sampling_method = args.sampling

    # change sampling method in configuration
    #for i in range(len(list(tests.keys()))):
     #   tests[list(tests.keys())[i]]["params"]["sampling"] = sampling_method

    os.makedirs(results_dir, exist_ok=True)
    file_name = "final_results_experiments_t1t2_alpha_{}_{}_{}_{}_{}_{}.csv".format(N,M,K,R,u, sampling_method)
    save_dir = os.path.join(results_dir, file_name)

    print("Start H0 simulation")
    res_h0 = _simulation_h0(tests, N, M, K, R, u, alpha)
    res = []
    for r in res_h0:
        res.append(list(res_h0[r]))
    results.append(res)

    # tests for when h1 hypothesis is true
    if test_h1:
        print("Start Ha simulation")
        res_h11 = _simulation_ha(tests, N, M, K, R, u, alpha)
        res = []
        for r in res_h0:
            res.append(list(res_h11[r]))
        results.append(res)
        print("Start second Ha simulation")
        res_h12 = _simulation_ha(tests, N, M, K, R, u, alpha, random=True)
        res = []
        for r in res_h0:
            res.append(list(res_h12[r]))
        results.append(res)


    results_df = pd.DataFrame(results)
    colnames = [t for t in tests]
    results_df.columns = colnames
    results_df.to_csv(save_dir, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments for type I and type II error in function of alpha")
    # data args
    parser.add_argument("-N", dest="N", type=int, default=100)
    parser.add_argument("-M", dest="M", type=int, default=10)
    parser.add_argument("-K", dest="K", type=int, default=3)
    parser.add_argument("-u", dest="u", type=float, default=0.01)
    parser.add_argument("-R", dest="R", type=int, default=1000)
    parser.add_argument("-sampling", dest="sampling", type=str, default='lambda')
    args = parser.parse_args()
    main_t1_t2(args, config=config_tests_reduced, test_h1=True)


            



