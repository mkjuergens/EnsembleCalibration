import random
import numpy as np

from scipy.optimize import minimize
from scipy.stats import halfnorm, norm, chi2
from pycalib.metrics import conf_ECE, classwise_ECE

from ensemblecalibration.calibration.calibration_measures import skce_ul_arr, skce_uq_arr
from ensemblecalibration.calibration.helpers import sample_l, sample_m, dec_bounds, constr, c1_constr, c2_constr
from ensemblecalibration.sampling import mhar_sampling_p, multinomial_label_sampling, uniform_weight_sampling, rejectance_sampling_p
from ensemblecalibration.calibration.test_objectives import hl_obj, skce_ul_obj, skce_uq_obj, confece_obj, classece_obj

_MAX_RNORM = np.max(halfnorm.rvs(size=1000*10000).reshape(-1,10000),axis=0)


""" 
CALIBRATION TESTS
"""

def hltest(P, y, params):
    """ Hosmer & Lemeshow test for strong classifier calibration

    Arguments
    ---------
        P : ndarray of shape (n_samples, n_classes) containing probs
        y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
        params: parameters of the test
        
    Return
    ------
        stat : test statistic
        pval : p-value
    """
    # calculate test statistic
    stat = 0
    # get idx for complement of reference probs in increasing order of prob
    idx = np.argsort(1-P[:,0])[::-1]
    # split idx array in nbins bins of roughly equal size
    idx_splitted = np.array_split(idx, params["n_bins"])
    # run over different cells and calculate stat
    stat = 0
    for k in range(P.shape[1]):
        for bin_bk in idx_splitted:
            o_bk = np.sum((y==k)[bin_bk])
            p_bk = np.sum(P[bin_bk,k])
            dev_bk = ((o_bk-p_bk)**2)/p_bk
            stat += dev_bk
    # and finally calculate righttail P-value
    pval = 1-chi2.cdf(stat,df=(params["n_bins"]-2)*(P.shape[1]-1))
    
    return stat, pval

def hl(P, y, params):
    # calculate test statistic
    stat = 0
    # get idx for complement of reference probs in increasing order of prob
    idx = np.argsort(1-P[:,0])[::-1]
    # split idx array in nbins bins of roughly equal size
    idx_splitted = np.array_split(idx, params["n_bins"])
    # run over different cells and calculate stat
    stat = 0
    for k in range(P.shape[1]): # P is of shape (N, M, K) 
        for bin_bk in idx_splitted:
            o_bk = np.sum((y==k)[bin_bk])
            p_bk = np.sum(P[bin_bk,k])
            dev_bk = ((o_bk-p_bk)**2)/p_bk
            stat += dev_bk

    return stat
    
def skceultest(P, y, params):
    """ SKCE_ul test statistic for strong classifier calibration.

    Based on Calibration tests in multi-class classification: A unifying framework by Widmann et al.

    Arguments
    ---------
        P : ndarray of shape (n_samples, n_classes) containing probs
        y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
        n_bins : number of bins used 

    Returns
    ------
        stat : test statistic
        pval : p-value
    """
    # calculate SKCE_ul estimate
    hat_skce_ul_arr = skce_ul_arr(P, y, dist_fct=params["dist"],
     sigma=params["sigma"])
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)
    hat_skce_ul_std = np.std(hat_skce_ul_arr)
    # calculate test statistic and P-value
    stat = (np.sqrt(len(hat_skce_ul_arr))/hat_skce_ul_std)*hat_skce_ul_mean
    pval = (1-norm.cdf(stat))

    return stat, pval


def skceul(P, y, params):

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
    # calculate SKCE_ul estimate
    hat_skce_ul_arr = skce_ul_arr(P, y, dist_fct=params["dist"], sigma=params["sigma"])
    hat_skce_ul_mean = np.mean(hat_skce_ul_arr)

    return hat_skce_ul_mean

def skceuq(P, y, params):

    """ SKCE_uq estimator for strong classifier calibration.

    Based on Calibration tests in multi-class classification: A unifying framework by Widmann et al.

    Arguments
    ---------
    P : ndarray of shape (n_samples, n_classes) containing probs
    y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
    dist_fct: distance function used in teh matrix valued kernel
    sigma: bandwidth used in the matrix valued kernel

    Returns
    ------
    hat_skce_uq_mean : estimation of SKCE_uq
    """
    
    # calculate SKCE_ul estimate
    hat_skce_uq_arr = skce_uq_arr(P, y, dist_fct=params["dist"], sigma=params["sigma"])
    hat_skce_uq_mean = np.mean(hat_skce_uq_arr)

    return hat_skce_uq_mean

def confece(P, y, params):

    """ confidence ECE estimator for confidence classifier calibration.

    Arguments
    ---------
        P : ndarray of shape (n_samples, n_classes) containing probs
        y : ndarray of shape (n_samples,) containing labels in {0,...,K-1}
        n_bins: integer defining the number of bins used for the test statistic

    Returns
    ------
        estimation of confidence ECE
    """
    return conf_ECE(y, P, params["n_bins"])


def classece(P, y, params):
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
    # convert y to indicator matrix (needed for classwise_ECE)
    yind = np.eye(P.shape[1])[y,:]

    return classwise_ECE(yind, P, 1, params["n_bins"])


### NON PARAMETRIC BOOTSTRAPPING TEST FOR TESTING ENSEMBLE CALIBRATION

def npbe_test(P: np.ndarray, y: np.ndarray, params: dict):
    """non-parametric bootstrapping test for general ensemble calibration

    Parameters
    ----------
    P : np.ndarray of shape (n_samples, ensemble_size, n_classes)
        array containing predictions for each sample and each ensemble member
    y : np.ndarray of shape (n_samples, ) 
        array containing labels in {0, ..., K-1}
    params : dict
        parameters for the test

    Returns
    -------
    decision: int, 0 if H0 (i.e. ensemble is calibrated) is not rejected, 1 otherwise
        l: ndarray of shape (ensemble_size,), combination that yields the decision
    """

    stats = np.zeros(params["n_resamples"]) 
    for b in range(params["n_resamples"]):
        # extract bootstrap samples
        P_b = random.sample(P.tolist(), P.shape[0])
        P_b = np.stack(P_b)
        # sample predictions using a predefined sampling method
        if params["sampling"] == "lambda":
            # sample convex combinations of ensemle predictions
            P_bar_b = uniform_weight_sampling(P_b) # of shape (N, M)
        elif params["sampling"] == "mcmc":
            P_bar_b = mhar_sampling_p(P_b, transform=params["transform"])
        elif params["sampling"] == "rejectance":
            P_bar_b = rejectance_sampling_p(P_b)
        else:
            raise NameError("check sampling method in configuration dictionary")
        # round to 5 decimal digits for numerical stability
        P_bar_b = np.trunc(P_bar_b*10**3)/(10**3)
        P_bar_b = np.clip(P_bar_b, 0, 1)
        P_bar_b = P_bar_b[~np.isnan(P_bar_b).any(axis=1)]
        # sample the labels from teh categorical distribution defined by teh predictions
        y_b = np.apply_along_axis(multinomial_label_sampling, 1, P_bar_b)
        # apply test for the bootstrap samples
        stats[b] = params["test"](P_bar_b, y_b, params)
    
    # calculate 1 - alpha quantile from the empirical distribution
    q_alpha = np.quantile(stats, 1 - params["alpha"])

    # use cobyla method for optimization
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

    minstat = params["obj"](l, P, y, params)
    decision = int(np.abs(minstat)> q_alpha)

    return decision, l

def npbetest_alpha(P, y, params):
    """
    updated, faster version of the npbe test:  calculate sampling distribution on,ly ocne, tehn use it for every alpha
    """
    stats = np.zeros(params["n_resamples"])
    for b in range(params["n_resamples"]):
        # extract bootstrap sample
        P_b = random.sample(P.tolist(), P.shape[0])
        P_b = np.stack(P_b)
        # sample predictions using a predefined sampling method
        if params["sampling"] == "lambda":
            # sample convex combinations of ensemle predictions
            P_bar_b = uniform_weight_sampling(P_b) # of shape (N, M)
        elif params["sampling"] == "mcmc":
            P_bar_b = mhar_sampling_p(P_b, transform=params["transform"])
        elif params["sampling"] == "rejectance":
            P_bar_b = rejectance_sampling_p(P_b)
        else:
            raise NameError("check sampling method in configuration dictionary")
        # round to 5 decimal digits for numerical stability
        P_bar_b = np.trunc(P_bar_b*10**3)/(10**3)
        P_bar_b = np.clip(P_bar_b, 0, 1)
        # P_bar_b = P_bar_b[~np.isnan(P_bar_b).any(axis=1)]
        # randomly sample labels from P_bar
       # print(P_bar_b.shape)
        y_b = np.apply_along_axis(sample_m, 1, P_bar_b)
        # perform test
        stats[b] = params["test"](P_bar_b, y_b, params)
    # calculate 1-alpha quantile from sampling distribution
    q_alpha = np.quantile(stats,1-(np.array(params["alpha"])))
    if params["optim"] == "neldermead":
        l = np.array([1/P.shape[1]]*P.shape[1])
        bnds = tuple([tuple([0,1]) for _ in range(P.shape[1])])
        cons = ({'type': 'eq', 'fun': constr})
        solution = minimize(params["obj"],l,(P, y, params),method='Nelder-Mead',bounds=bnds,constraints=cons)
        l = np.array(solution.x)
    elif params["optim"] == "cobyla":
        l = np.array([1/P.shape[1]]*P.shape[1]) # initial guess 
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

def _npbe_test(P, y, alpha, params):
    params["alpha"] = alpha
    dec, l = npbe_test(P, y, params)

    return dec

def _npbetest_alpha(P, y, alpha, params):
    params["alpha"] = alpha
    dec, l = npbetest_alpha(P, y, params)

    return dec



        





