import random
import numpy as np

from scipy.optimize import minimize
from scipy.stats import halfnorm, norm, chi2
from pycalib.metrics import conf_ECE, classwise_ECE

from ensemblecalibration.calibration.calibration_measures import skce_ul_arr, skce_uq_arr
from ensemblecalibration.calibration.helpers import sample_l, sample_m, dec_bounds, constr_eq, c1_constr, c2_constr
from ensemblecalibration.sampling import mhar_sampling_p, multinomial_label_sampling, uniform_weight_sampling, rejectance_sampling_p
from ensemblecalibration.calibration.test_objectives import hl_obj, skce_ul_obj, skce_uq_obj, confece_obj, classece_obj
from ensemblecalibration.calibration.minimization import solve_cobyla1D, solve_cobyla2D, solve_neldermead1D, solve_neldermead2D



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


def skceul(P, y, params, square_out: bool = True):

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

    if square_out:
        hat_skce_ul_mean = hat_skce_ul_mean ** 2

    return hat_skce_ul_mean

def skceuq(P, y, params, square_out: bool = True):

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

    if square_out:
        hat_skce_uq_mean = hat_skce_uq_mean ** 2

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

def npbetest_alpha(P, y, params):
    """
    updated, faster version of the npbe test:  calculate sampling distribution on,ly ocne,
    then use it for every alpha
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
        y_b = np.apply_along_axis(sample_m, 1, P_bar_b)
        # perform test
        stats[b] = params["test"](P_bar_b, y_b, params)

    # calculate 1-alpha quantile from sampling distribution
    q_alpha = np.quantile(stats,1-(np.array(params["alpha"])))

    # solve minimization problem
    if params["optim"] == "neldermead":
        if params["x_dependency"]:
            l = solve_neldermead2D(P, y, params)
        else:
            l = solve_neldermead1D(P, y, params)
    elif params["optim"] == "cobyla":
        if params["x_dependency"]:
            l = solve_cobyla2D(P, y, params)
        else:
            l = solve_cobyla1D(P, y, params)

    else:
        raise NotImplementedError

    minstat = params["obj"](l, P, y, params)
    decision = list(map(int,np.abs(minstat)>q_alpha))

    return decision, l

def _npbetest_alpha(P, y, alpha, params):
    params["alpha"] = alpha
    dec, l = npbetest_alpha(P, y, params)

    return dec



        





