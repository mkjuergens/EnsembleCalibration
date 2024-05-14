""" 
Final experiments for optimizer

Author: Thomas Mortier
Date: May 2022
"""
import sys
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
import nlopt
from caltest import tvdistance # distance functions
from caltest import aucb_obj, hl_obj, skce_ul_obj, skce_uq_obj, confece_obj, classece_obj # objectives 
from caltest import _npbetest, _aucbtest, __npbetest, __aucbtest # tests
from caltest import hl, skceul, skceuq, confece, classece # estimators
from scipy.stats import halfnorm, dirichlet, multinomial, multivariate_normal
from scipy.optimize import linprog
from tqdm import tqdm

def get_ens_alpha(K, u, a0):
    p0 = np.random.dirichlet(a0,1)[0,:]
    
    return (K*p0)/u

def _simulation(tests, N, M, K, R, u, alpha):
    lambdas = [] 
    results = {}
    for test in tests:
        results[test] = {"outcome": [], "lambdah": []}
    for _ in range(R):
        l = np.random.dirichlet([1/M]*M,1)[0,:]
        lambdas.append(l)
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
            dec, l = tests[test]["test"](P, y, alpha, tests[test]["params"])
            results[test]["outcome"].append(dec)
            results[test]["lambdah"].append(l)

    return results, lambdas

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
            
def main_final_optim():
    tests = {
        "AUCB": {
            "test": __aucbtest,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "obj": aucb_obj,
                "n_resamples": 100, 
                "dist": tvdistance, 
                "h": 2},
        },
        "NPBE_SKCEul": {
            "test": __npbetest,
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
        #    "test": __npbetest,
        #    "params": {
        #        "optim": "other", 
        #        "dist": tvdistance, 
        #        "h": 2, 
        #        "n_resamples": 100, 
        #        "test": skceuq, 
        #        "obj": skce_uq_obj}
        #},
        "NPBE_HL5": {
            "test": __npbetest,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 5,
                "test": hl, 
                "obj": hl_obj},
        },
        "NPBE_HL15": {
            "test": __npbetest,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 15,
                "test": hl, 
                "obj": hl_obj},
        },
        "NPBE_CONFECE5": {
            "test": __npbetest,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 5,
                "test": confece, 
                "obj": confece_obj}
        },
        "NPBE_CONFECE15": {
            "test": __npbetest,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 15,
                "test": confece, 
                "obj": confece_obj}
        },
        "NPBE_CLASSECE5": {
            "test": __npbetest,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 5,
                "test": classece, 
                "obj": classece_obj}   
        },
        "NPBE_CLASSECE15": {
            "test": __npbetest,
            "params": {
                "l_prior": 1,
                "optim": "cobyla", 
                "n_resamples": 100, 
                "nbins": 15,
                "test": classece, 
                "obj": classece_obj}   
        }
    }
    settings = {
        "N": [100, 500, 1000],
        "M": [10, 50, 100, 1],
        "K": [10, 50, 100],
        "R": [50],
        "u": [0.01, 0.1, 0.5],
        "alpha": [0.05, 0.20, 0.50]
    }
    results = []
    for s in tqdm(settings_parser(settings)):
        r, l = _simulation(tests, *s)
        res = []
        res.append(l)
        res.extend(s)
        for ri in r:
            res.append(r[ri])
        results.append(res)
    results_df = pd.DataFrame(results)
    colnames = ["L"]+[s for s in settings]+[t for t in tests]
    results_df.columns = colnames
    results_df.to_csv("./final_results_experiments_optim_cobyla.csv", index=False)

if __name__ == "__main__":
    main_final_optim()
