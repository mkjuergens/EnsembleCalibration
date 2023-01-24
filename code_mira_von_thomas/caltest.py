""" Code for calibration tests.

Author: Thomas Mortier
Date: April 2022
"""
import sys
from scipy.stats import halfnorm, norm, chi2, dirichlet, multinomial
from pycalib.metrics import conf_ECE, classwise_ECE
from scipy.optimize import minimize
import random
import numpy as np

_MAX_RNORM = np.max(halfnorm.rvs(size=1000*10000).reshape(-1,10000),axis=0)

"""
OTHER FUNCTIONS
"""

""" Total variation distance function """
def tvdistance(p, q):
    return 0.5*np.sum(np.abs(p-q))

""" L2 distance function """
def l2distance(p, q):
    return np.sqrt(np.sum((p-q)**2))

""" Vector-valued kernel function """
def KM(p, q, params):
    return np.exp((-1/params["h"])*(params["dist"](p, q)**2))*np.eye(len(p))

""" hij for SKCE_ul and SKCE_uq estimators """
def h(p, q, yp, yq, params):
    K = KM(p, q, params)
    ypp = (yp-p)
    yqq = (yq-q)
    hpq1 = np.matmul(ypp,K)
    hpq = np.matmul(hpq1,yqq)
    
    return hpq

def skce_ul_arr(P_bar, y, params):
    n = round(P_bar.shape[0]/2)
    # transform y to one-hot encoded labels
    yoh = np.eye(P_bar.shape[1])[y,:]
    stats = np.zeros(n)
    for i in range(0,n):
        stats[i] = h(P_bar[(2*i),:], P_bar[(2*i)+1,:], yoh[(2*i),:], yoh[(2*i)+1,:], params)

    return stats

def skce_uq_arr(P_bar, y, params):
    n = P_bar.shape[0]
    # transform y to one-hot encoded labels
    yoh = np.eye(P_bar.shape[1])[y,:]
    stats = np.zeros(int((n*(n-1))/2))
    cntr = 0
    for j in range(1,n):
        for i in range(j):
            stats[cntr] = h(P_bar[i,:], P_bar[j,:], yoh[i,:], yoh[j,:], params)
            cntr+=1

    return stats

""" Function which samples label from categorical distribution defined by p. """
def sample_m(p):
    try:
        y = np.argmax(multinomial(1,p).rvs(size=1)[0,:])
    except ValueError as e:
        y = np.argmax(p)

    return y

""" Function which samples convex combinations given ensembles 

    Arguments
    ---------
        P : NxMxK 
        params : params for sampling procedure

    Returns
    -------
        P_bar_b : NxK

"""
def sample_l(P, params):
    # take convex combination of ensemble predictions
    if "l_prior" not in params:
        l = dirichlet([1/P.shape[1]]*P.shape[1]).rvs(size=1)[0,:]
    else:
        l = dirichlet([params["l_prior"]]*P.shape[1]).rvs(size=1)[0,:]
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

""" 
OBJECTIVES FOR NPBE 
"""

""" Objective function for aucbtest """
def aucb_obj(x, P, y, params):
    # take convex combinations of ensemble predictions
    P_bar = np.matmul(np.swapaxes(P,1,2),x)
    # calculate SKCE_ul estimate
    hat_skce_ul_arr = skce_ul_arr(P_bar, y, params)
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)    
    hat_skce_ul_std = np.std(hat_skce_ul_arr)
    # quantile for uniform confidence bounds
    c_alpha = np.quantile(_MAX_RNORM, 1-params["alpha"])
    # calculate uniform confidence bounds and check wheter 0 is included
    dec = dec_bounds(hat_skce_ul_mean-(c_alpha/np.sqrt(len(hat_skce_ul_arr)))*hat_skce_ul_std, hat_skce_ul_mean+(c_alpha/np.sqrt(len(hat_skce_ul_arr)))*hat_skce_ul_std)

    return dec

""" Objective function for hl """
def hl_obj(x, P, y, params):
    # take convex combinations of ensemble predictions
    P_bar = np.matmul(np.swapaxes(P,1,2),x)
    stat, _ = hltest(P_bar, y, params)

    return stat

""" Objective function for skce_ul """
def skce_ul_obj(x, P, y, params):
    # take convex combinations of ensemble predictions
    P_bar = np.matmul(np.swapaxes(P,1,2),x)
    # calculate SKCE_ul estimate
    hat_skce_ul_arr = skce_ul_arr(P_bar, y, params)
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)    

    return hat_skce_ul_mean

""" Objective function for skce_uq """
def skce_uq_obj(x, P, y, params):
    # take convex combinations of ensemble predictions
    P_bar = np.matmul(np.swapaxes(P,1,2),x)
    # calculate SKCE_uq estimate
    hat_skce_uq_arr = skce_uq_arr(P_bar, y, params)
    hat_skce_uq_mean = np.mean(hat_skce_uq_arr)    

    return hat_skce_uq_mean

""" Objective function for confidence ECE """
def confece_obj(x, P, y, params):
    # take convex combinations of ensemble predictions
    P_bar = np.matmul(np.swapaxes(P,1,2),x)
    # calculate confidence ECE
    stat = conf_ECE(y, P_bar, params["nbins"])
    #print("[info optim] {0} gives {1}".format(x,stat))

    return stat

""" Objective function for classwise ECE """
def classece_obj(x, P, y, params):
    # take convex combinations of ensemble predictions
    P_bar = np.matmul(np.swapaxes(P,1,2),x)
    # transform y to indicator matrix (needed for classwise_ECE)
    yind = np.eye(P.shape[2])[y,:]
    # calculate classwise ECE
    stat = classwise_ECE(yind, P_bar, 1, params["nbins"])

    return stat

""" 
CALIBRATION TESTS
"""

""" Hosmer & Lemeshow test for strong classifier calibration

Arguments
---------
    P : ndarray of shape (n_samples, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params for test
    
Return
------
    stat : test statistic
    pval : p-value
"""
def hltest(P, y, params):
    # calculate test statistic
    stat = 0
    # get idx for complement of reference probs in increasing order of prob
    idx = np.argsort(1-P[:,0])[::-1]
    # split idx array in nbins bins of roughly equal size
    idx_splitted = np.array_split(idx, params["nbins"])
    # run over different cells and calculate stat
    stat = 0
    for k in range(P.shape[1]):
        for bin_bk in idx_splitted:
            o_bk = np.sum((y==k)[bin_bk])
            p_bk = np.sum(P[bin_bk,k])
            dev_bk = ((o_bk-p_bk)**2)/p_bk
            stat += dev_bk
    # and finally calculate righttail P-value
    pval = 1-chi2.cdf(stat,df=(params["nbins"]-2)*(P.shape[1]-1))
    
    return stat, pval

""" Hosmer & Lemeshow test statistic for strong classifier calibration

Arguments
---------
    P : ndarray of shape (n_samples, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params for test
    
Return
------
    stat : estimated Hosmer & Lemeshow test statistic
"""
def hl(P, y, params):
    # calculate test statistic
    stat = 0
    # get idx for complement of reference probs in increasing order of prob
    idx = np.argsort(1-P[:,0])[::-1]
    # split idx array in nbins bins of roughly equal size
    idx_splitted = np.array_split(idx, params["nbins"])
    # run over different cells and calculate stat
    stat = 0
    for k in range(P.shape[1]):
        for bin_bk in idx_splitted:
            o_bk = np.sum((y==k)[bin_bk])
            p_bk = np.sum(P[bin_bk,k])
            dev_bk = ((o_bk-p_bk)**2)/p_bk
            stat += dev_bk

    return stat
    
""" SKCE_ul test for strong classifier calibration.

Based on Calibration tests in multi-class classification: A unifying framework by Widmann et al.

Arguments
---------
    P : ndarray of shape (n_samples, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params for kernel and test

Returns
------
    stat : test statistic
    pval : p-value
"""
def skceultest(P, y, params):
    # calculate SKCE_ul estimate
    hat_skce_ul_arr = skce_ul_arr(P, y, params)
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)
    hat_skce_ul_std = np.std(hat_skce_ul_arr)
    # calculate test statistic and P-value
    stat = (np.sqrt(len(hat_skce_ul_arr))/hat_skce_ul_std)*hat_skce_ul_mean
    pval = (1-norm.cdf(stat))

    return stat, pval

""" SKCE_ul estimator for strong classifier calibration.

Based on Calibration tests in multi-class classification: A unifying framework by Widmann et al.

Arguments
---------
    P : ndarray of shape (n_samples, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params for kernel and test

Returns
------
    hat_skce_ul_mean : estimation of SKCE_ul
"""
def skceul(P, y, params):
    # calculate SKCE_ul estimate
    hat_skce_ul_arr = skce_ul_arr(P, y, params)
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)

    return hat_skce_ul_mean

""" SKCE_uq estimator for strong classifier calibration.

Based on Calibration tests in multi-class classification: A unifying framework by Widmann et al.

Arguments
---------
    P : ndarray of shape (n_samples, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params for kernel and test

Returns
------
    hat_skce_uq_mean : estimation of SKCE_uq
"""
def skceuq(P, y, params):
    # calculate SKCE_ul estimate
    hat_skce_uq_arr = skce_uq_arr(P, y, params)
    hat_skce_uq_mean = np.mean(hat_skce_uq_arr)

    return hat_skce_uq_mean

""" confidence ECE estimator for confidence classifier calibration.

Arguments
---------
    P : ndarray of shape (n_samples, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params

Returns
------
    estimation of confidence ECE
"""
def confece(P, y, params):
    return conf_ECE(y, P, params["nbins"])

""" classwise ECE estimator for strong classifier calibration.

Arguments
---------
    P : ndarray of shape (n_samples, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params

Returns
------
    estimation of classwise ECE
"""
def classece(P, y, params):
    # convert y to indicator matrix (needed for classwise_ECE)
    yind = np.eye(P.shape[1])[y,:]

    return classwise_ECE(yind, P, 1, params["nbins"])

""" Asymptotic test with uniform confidence bands for strong ensemble calibration.

Based on Calibration tests in multi-class classification: A unifying framework by Widmann et al.

Arguments
---------
    P : ndarray of shape (n_samples, ensemble_size, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params for kernel and test

Returns
-------
    decision : int, 0 if H0 (i.e. ensemble calibrated) not rejected and 1 otherwise
    l : ndarray of shape (ensemble_size,), combination that gives decision
"""
def aucbtest(P, y, params):
    if params["optim"] == "neldermead":
        l = np.array([1/P.shape[1]]*P.shape[1])
        bnds = tuple([tuple([0,1]) for _ in range(P.shape[1])])
        cons = ({'type': 'eq', 'fun': constr})
        solution = minimize(aucb_obj,l,(P, y, params),method='Nelder-Mead',bounds=bnds,constraints=cons)
        l = np.array(solution.x)
        decision = int(aucb_obj(l, P, y, params)!=0.0)
    elif params["optim"] == "cobyla":
        l = np.array([1/P.shape[1]]*P.shape[1])
        bnds = tuple([tuple([0,1]) for _ in range(P.shape[1])])
        cons = [{'type': 'ineq', 'fun': c1_constr}, {'type': 'ineq', 'fun': c2_constr}]
        # bounds must be included as constraints for COBYLA
        for factor in range(len(bnds)):
            lower, upper = bnds[factor]
            lo = {'type': 'ineq',
                 'fun': lambda x, lb=lower, i=factor: x[i] - lb}
            up = {'type': 'ineq',
                 'fun': lambda x, ub=upper, i=factor: ub - x[i]}
            cons.append(lo)
            cons.append(up)
        solution = minimize(params["obj"],l,(P, y, params),method='COBYLA',constraints=cons)
        l = np.array(solution.x)   
        decision = int(aucb_obj(l, P, y, params)!=0.0)
    else:
        if "l_prior" not in params:
            L = np.random.dirichlet([1/P.shape[1]]*P.shape[1], params["n_resamples"])
        else:
            L = np.random.dirichlet([params["l_prior"]]*P.shape[1], params["n_resamples"])
        min_l = np.array([1/P.shape[1]]*P.shape[1])
        min_l_stat = sys.maxsize
        for li in L:
            li_ev = aucb_obj(li, P, y, params)
            if li_ev <= min_l_stat:
                min_l = li
                min_l_stat = li_ev
        l = min_l
        decision = int(min_l_stat!=0.0)

    return decision, l

def _aucbtest(P, y, alpha, params):
    params["alpha"] = alpha
    dec, _ = aucbtest(P, y, params)

    return dec

def __aucbtest(P, y, alpha, params):
    params["alpha"] = alpha
    dec, l = aucbtest(P, y, params)

    return dec, l

""" Nonparametric bootstrapping test for general ensemble calibration 
    
Arguments
---------
    P : ndarray of shape (n_samples, ensemble_size, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params test

Returns
-------
    decision : int, 0 if H0 (i.e. ensemble calibrated) not rejected and 1 otherwise
    l : ndarray of shape (ensemble_size,), combination that gives decision

"""
def npbetest(P, y, params):
    stats = np.zeros(params["n_resamples"])
    for b in range(params["n_resamples"]):
        # extract bootstrap sample
        P_b = random.sample(P.tolist(), P.shape[0])
        P_b = np.stack(P_b)
        # take convex combination of ensemble predictions
        P_bar_b = sample_l(P_b, params)
        # randomly sample labels from P_bar
        y_b = np.apply_along_axis(sample_m, 1, P_bar_b)
        # perform test
        stats[b] = params["test"](P_bar_b, y_b, params)
    # calculate 1-alpha quantile from sampling distribution
    q_alpha = np.quantile(stats,1-(params["alpha"]))
    if params["optim"] == "neldermead":
        l = np.array([1/P.shape[1]]*P.shape[1])
        bnds = tuple([tuple([0,1]) for _ in range(P.shape[1])])
        cons = ({'type': 'eq', 'fun': constr})
        solution = minimize(params["obj"],l,(P, y, params),method='Nelder-Mead',bounds=bnds,constraints=cons)
        l = np.array(solution.x)
    elif params["optim"] == "cobyla":
        l = np.array([1/P.shape[1]]*P.shape[1])
        bnds = tuple([tuple([0,1]) for _ in range(P.shape[1])])
        cons = [{'type': 'ineq', 'fun': c1_constr}, {'type': 'ineq', 'fun': c2_constr}]
        # bounds must be included as constraints for COBYLA
        for factor in range(len(bnds)):
            lower, upper = bnds[factor]
            lo = {'type': 'ineq',
                 'fun': lambda x, lb=lower, i=factor: x[i] - lb}
            up = {'type': 'ineq',
                 'fun': lambda x, ub=upper, i=factor: ub - x[i]}
            cons.append(lo)
            cons.append(up)
        solution = minimize(params["obj"],l,(P, y, params),method='COBYLA',constraints=cons)
        l = np.array(solution.x)   
    else:
        if "l_prior" not in params:
            L = np.random.dirichlet([1/P.shape[1]]*P.shape[1], params["n_resamples"])
        else:
            L = np.random.dirichlet([params["l_prior"]]*P.shape[1], params["n_resamples"])
        min_l = np.array([1/P.shape[1]]*P.shape[1])
        min_l_stat = sys.maxsize
        for li in L:
            li_ev = params["obj"](li, P, y, params)
            if li_ev <= min_l_stat:
                min_l = li
                min_l_stat = li_ev
        l = min_l
    minstat = params["obj"](l, P, y, params)
    decision = int(np.abs(minstat)>q_alpha)

    return decision, l

def npbetest_alpha(P, y, params):
    stats = np.zeros(params["n_resamples"])
    for b in range(params["n_resamples"]):
        # extract bootstrap sample
        P_b = random.sample(P.tolist(), P.shape[0])
        P_b = np.stack(P_b)
        # take convex combination of ensemble predictions
        P_bar_b = sample_l(P_b, params)
        # randomly sample labels from P_bar
        y_b = np.apply_along_axis(sample_m, 1, P_bar_b)
        # perform test
        stats[b] = params["test"](P_bar_b, y_b, params)
    # calculate 1-alpha quantile from sampling distribution
    q_alpha = np.quantile(stats,1-(params["alpha"]))
    if params["optim"] == "neldermead":
        l = np.array([1/P.shape[1]]*P.shape[1])
        bnds = tuple([tuple([0,1]) for _ in range(P.shape[1])])
        cons = ({'type': 'eq', 'fun': constr})
        solution = minimize(params["obj"],l,(P, y, params),method='Nelder-Mead',bounds=bnds,constraints=cons)
        l = np.array(solution.x)
    elif params["optim"] == "cobyla":
        l = np.array([1/P.shape[1]]*P.shape[1])
        bnds = tuple([tuple([0,1]) for _ in range(P.shape[1])])
        cons = [{'type': 'ineq', 'fun': c1_constr}, {'type': 'ineq', 'fun': c2_constr}]
        # bounds must be included as constraints for COBYLA
        for factor in range(len(bnds)):
            lower, upper = bnds[factor]
            lo = {'type': 'ineq',
                 'fun': lambda x, lb=lower, i=factor: x[i] - lb}
            up = {'type': 'ineq',
                 'fun': lambda x, ub=upper, i=factor: ub - x[i]}
            cons.append(lo)
            cons.append(up)
        solution = minimize(params["obj"],l,(P, y, params),method='COBYLA',constraints=cons)
        l = np.array(solution.x)   
    else:
        if "l_prior" not in params:
            L = np.random.dirichlet([1/P.shape[1]]*P.shape[1], params["n_resamples"])
        else:
            L = np.random.dirichlet([params["l_prior"]]*P.shape[1], params["n_resamples"])
        min_l = np.array([1/P.shape[1]]*P.shape[1])
        min_l_stat = sys.maxsize
        for li in L:
            li_ev = params["obj"](li, P, y, params)
            if li_ev <= min_l_stat:
                min_l = li
                min_l_stat = li_ev
        l = min_l
    minstat = params["obj"](l, P, y, params)
    decision = list(map(int,np.abs(minstat)>q_alpha))

    return decision, l

""" Nonparametric boostrapping test for general ensemble calibration 

TODO: delete
    
Arguments
---------
    P : ndarray of shape (n_samples, ensemble_size, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    params : dict, params test

Returns
-------
    decision : int, 0 if H0 (i.e. ensemble calibrated) not rejected and 1 otherwise
    l : ndarray of shape (ensemble_size,), combination that gives decision

"""
def npbetest_debug(P, y, params):
    stats = np.zeros(params["n_resamples"])
    for b in range(params["n_resamples"]):
        # extract bootstrap sample
        P_b = random.sample(P.tolist(), P.shape[0])
        P_b = np.stack(P_b)
        # take convex combination of ensemble predictions
        P_bar_b = sample_l(P_b, params)
        # randomly sample labels from P_bar
        y_b = np.apply_along_axis(sample_m, 1, P_bar_b)
        # perform test
        stats[b] = params["test"](P_bar_b, y_b, params)
    # calculate 1-alpha quantile from sampling distribution
    q_alpha = np.quantile(stats,1-(params["alpha"]))
    if params["optim"] == "neldermead":
        l = np.array([1/P.shape[1]]*P.shape[1])
        bnds = tuple([tuple([0,1]) for _ in range(P.shape[1])])
        cons = ({'type': 'eq', 'fun': constr})
        solution = minimize(params["obj"],l,(P, y, params), method='Nelder-Mead', bounds=bnds, constraints=cons)
        l = np.array(solution.x)
    else:
        if "l_prior" not in params:
            L = np.random.dirichlet([1/P.shape[1]]*P.shape[1], params["n_resamples"])
        else:
            L = np.random.dirichlet([params["l_prior"]]*P.shape[1], params["n_resamples"])
        min_l = np.array([1/P.shape[1]]*P.shape[1])
        min_l_stat = sys.maxsize
        for li in L:
            li_ev = params["obj"](li, P, y, params)
            if li_ev <= min_l_stat:
                min_l = li
                min_l_stat = li_ev
        l = min_l
    minstat = params["obj"](l, P, y, params)
    decision = int(np.abs(minstat)>q_alpha)

    return decision, l, stats

def _npbetest(P, y, alpha, params):
    params["alpha"] = alpha
    dec, l = npbetest(P, y, params)

    return dec

def _npbetest_alpha(P, y, alpha, params):
    params["alpha"] = alpha
    dec, l = npbetest_alpha(P, y, params)

    return dec

def __npbetest(P, y, alpha, params):
    params["alpha"] = alpha
    dec, l = npbetest(P, y, params)

    return dec, l
