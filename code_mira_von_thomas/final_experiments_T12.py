""" 
Final experiments for FAST Type 1 and Type 2 error analysis in function of alpha

Author: Thomas Mortier
Date: May 2022
"""
from os import wait
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import time
import ternary
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
sys.path.insert(0, "/home/data/tfmortier/Research/Calibration/calibration/src/main/py")
sys.path.insert(0, "/usr/local/lib/python3.7/site-packages")
from caltest import tvdistance # distance functions
from caltest import aucb_obj, hl_obj, skce_ul_obj, skce_uq_obj, confece_obj, classece_obj # objectives 
from caltest import _npbetest, _npbetest_alpha, _aucbtest # tests
from caltest import hl, skceul, skceuq, confece, classece # estimators
from scipy.stats import halfnorm, dirichlet, multinomial, multivariate_normal
from scipy.optimize import linprog
from tqdm import tqdm

def get_ens_alpha(K, u, a0):
    p0 = np.random.dirichlet(a0,1)[0,:]
    
    return (K*p0)/u

def isCalibrated(P, p):
    M = len(P)
    K = len(p)
    c = np.zeros(M)
    A = np.r_[P.T,np.ones((1,M))]
    b = np.r_[p, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    
    return lp.success

def getBoundary(P, mu, yc):
    l_arr = np.linspace(0,1,100)
    # get convex combinations between yc and mu
    L = np.stack([l*yc+(1-l)*mu for l in l_arr])
    # determine boundary
    bi = 0
    for i in range(len(L)):
        if not isCalibrated(P, L[i,:]):
            bi = i-1
            break
    yb = L[bi,:]
    
    return yb

def _simulation_h0(tests, N, M, K, R, u, alpha):
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
            # sample in stance
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

def _simulation_ha(tests, N, M, K, R, u, alpha, l=None, random=False):
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
            if not random:
                c = np.argmax(mu)
            else:
                c = np.random.randint(0,K,1)[0]
            yc = np.eye(K)[c,:]
            # get boundary
            if M==1:
                yb = mu
            else:
                yb = getBoundary(Pm, mu, yc)
            # get random convex combination
            if l is None:
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

def settings_parser(settings):
    r_list = []
    for setting in settings:
        for setting_val in settings[setting]:
            ret_settings = []
            for other_setting in settings:
                if other_setting == setting:
                    ret_settings.append(setting_val)
                else:
                    ret_settings.append(settings[other_setting][0])
            r_list.append(ret_settings)
    # filter out duplicates
    ret_list = []
    for s in r_list:
        if s not in ret_list:
            ret_list.append(s)
            
    return ret_list
            
def main_final_t1t2(args):
    tests = {
        "NPBE_SKCEul": {
            "test": _npbetest_alpha,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "dist": tvdistance, 
                "h": 2, 
                "n_resamples": 100, 
                "test": skceul, 
                "obj": skce_ul_obj}
        },
        #"NPBE_SKCEuq": {
        #    "test": _npbetest_alpha,
        #    "params": {
        #        "l_prior": 1,
        #        "optim": "cobyla", 
        #        "dist": tvdistance, 
        #        "h": 2, 
        #        "n_resamples": 100, 
        #        "test": skceuq, 
        #        "obj": skce_uq_obj}
        #},
        "NPBE_HL5": {
            "test": _npbetest_alpha,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 5,
                "test": hl, 
                "obj": hl_obj},
        },
        "NPBE_HL10": {
            "test": _npbetest_alpha,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 10,
                "test": hl, 
                "obj": hl_obj},
        },
        "NPBE_CONFECE5": {
            "test": _npbetest_alpha,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 5,
                "test": confece, 
                "obj": confece_obj}
        },
        "NPBE_CONFECE10": {
            "test": _npbetest_alpha,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 10,
                "test": confece, 
                "obj": confece_obj}
        },
        "NPBE_CLASSECE5": {
            "test": _npbetest_alpha,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 5,
                "test": classece, 
                "obj": classece_obj}   
        },
        "NPBE_CLASSECE10": {
            "test": _npbetest_alpha,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 10,
                "test": classece, 
                "obj": classece_obj}   
        }
    }
    N = args.N
    M = args.M
    K = args.K
    u = args.u
    R = 1000
    alpha = [0.05, 0.13, 0.21, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.79, 0.87, 0.95]
    results = []
    print("Start H0 simulation")
    res_h0 = _simulation_h0(tests, N, M, K, R, u, alpha)
    res = []
    for r in res_h0:
        res.append(list(res_h0[r]))
    results.append(res)
    print("Start Ha simulation")
    res_h11 = _simulation_ha(tests, N, M, K, R, u, alpha, None, False)
    res = []
    for r in res_h0:
        res.append(list(res_h11[r]))
    results.append(res)
    print("Start Ha simulation")
    res_h12 = _simulation_ha(tests, N, M, K, R, u, alpha, None, True)
    res = []
    for r in res_h0:
        res.append(list(res_h12[r]))
    results.append(res)
    results_df = pd.DataFrame(results)
    colnames = [t for t in tests]
    results_df.columns = colnames
    results_df.to_csv("./final_results_experiments_t1t2_alpha_{}_{}_{}_{}.csv".format(N,M,K,u), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments for type I and type II error in function of alpha")
    # data args
    parser.add_argument("-N", dest="N", type=int, default=100)
    parser.add_argument("-M", dest="M", type=int, default=10)
    parser.add_argument("-K", dest="K", type=int, default=10)
    parser.add_argument("-u", dest="u", type=float, default=0.01)
    args = parser.parse_args()
    main_final_t1t2(args)
