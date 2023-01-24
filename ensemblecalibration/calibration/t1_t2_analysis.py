import sys

import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.stats import multinomial
from scipy.optimize import linprog

sys.path.append('../..')
from ensemblecalibration.calibration.iscalibrated import is_calibrated
from ensemblecalibration.calibration.config import config_tests

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

def _simulation_h0(tests, N: int, M: int, K: int, R: int, u: float, alpha: float):

    results = {}
    for test in tests:
        results[test] = np.zeros(len(alpha))
    for _ in tqdm(range(R)):
        l = np.random.dirichlet([1/M]*M,1)[0,:]
        L = np.repeat(l.reshape(-1,1),K,axis=1)
        P, y = [], []
        for _ in range(N):
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
        for test in tests:
            results[test] += np.array(tests[test]["test"](P, y, alpha, tests[test]["params"]))
    for test in tests:
        # calculate mean
        results[test] = results[test]/R 
        
    return results


def _simulation_ha(tests, N: int, M: int, K: int, R: int, u: float, alpha: float):
    results = {}
    for test in tests:
        results[test] = np.zeros(len(alpha))
    for _ in tqdm(range(R)):
        P, y = [], []
        for _ in range(N):
            a = get_ens_alpha(K, u, [1/K]*K)
            while np.any(a<=0):
                a = get_ens_alpha(K, u, [1/K]*K)
            mu = (a*u)/K
            if M==1:
                Pm = mu.reshape(1,-1)
            else:
                Pm = np.random.dirichlet(a, M)
            # pick class and sample ground-truth outside credal set 
            c = np.argmax(mu)
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
        for test in tests:
            results[test] += (1-np.array(tests[test]["test"](P, y, alpha, tests[test]["params"])))
    for test in tests:
        results[test] = results[test]/R
        
    return results

def main_t1_t2(args, config=config_tests, test_h1: bool = True):
    tests = config_tests
    results = []
    alpha = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.79, 0.87, 0.95]
    N = args.N
    M = args.M
    K = args.K
    u = args.u
    R = 1000

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
        res_h12 = _simulation_ha(tests, N, M, K, R, u, alpha)
        res = []
        for r in res_h0:
            res.append(list(res_h12[r]))
        results.append(res)

    sampling = list(tests.keys())["params"]["sampling"]

    results_df = pd.DataFrame(results)
    colnames = [t for t in tests]
    results_df.columns = colnames
    results_df.to_csv("./final_results_experiments_t1t2_alpha_{}_{}_{}_{}_{}.csv".format(N,M,K,u, sampling), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments for type I and type II error in function of alpha")
    # data args
    parser.add_argument("-N", dest="N", type=int, default=100)
    parser.add_argument("-M", dest="M", type=int, default=10)
    parser.add_argument("-K", dest="K", type=int, default=3)
    parser.add_argument("-u", dest="u", type=float, default=0.01)
    args = parser.parse_args()
    main_t1_t2(args, test_h1=False)


            



