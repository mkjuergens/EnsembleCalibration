import random
import numpy as np
import pandas as pd

from scipy.stats import multinomial
from scipy.optimize import linprog
from tqdm import tqdm

from ensemblecalibration.calibration.iscalibrated import is_calibrated
from ensemblecalibration.calibration.config import config_tests, settings

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
        results[test] = []
    for _ in range(R):
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
            results[test].append(tests[test]["test"](P, y, alpha, tests[test]["params"]))
    for test in tests:
        results[test] = np.array(results[test])
        
    return results


def _simulation_ha(tests, N: int, M: int, K: int, R: int, u: float, alpha: float):
    results = {}
    for test in tests:
        results[test] = []
    for _ in range(R):
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
            results[test].append(tests[test]["test"](P, y, alpha, tests[test]["params"]))
    for test in tests:
        results[test] = 1-np.array(results[test])
        
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

def main_t1_t2():
    tests = config_tests
    results = []
    for s in tqdm(settings_parser(settings)):
        print(f'Setting: {s}')
        res_h0 = _simulation_h0(tests, *s)
        print(f'Results H0: {res_h0}')
        res = []
        res.extend(s)
        res.append("M1")
        for r in res_h0:
            res.append(list(res_h0[r]))
        results.append(res)
        res_h11 = _simulation_ha(tests, *s, None, False)
        res = []
        res.extend(s)
        res.append("M2")
        for r in res_h0:
            res.append(list(res_h11[r]))
        results.append(res)
        res_h12 = _simulation_ha(tests, *s, None, True)
        res = []
        res.extend(s)
        res.append("M3")
        for r in res_h0:
            res.append(list(res_h12[r]))
        results.append(res)

    results_df = pd.DataFrame(results)
    colnames = [s for s in settings]+["H"]+[t for t in tests]
    results_df.columns = colnames
    results_df.to_csv("./final_results_experiments_t1t2_cobyla_big.csv", index=False)


if __name__ == "__main__":
    main_t1_t2()


            



