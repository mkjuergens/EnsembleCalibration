import numpy as np
import random

from ensemblecalibration.calibration.calibration_estimates.helpers import calculate_pbar
from ensemblecalibration.sampling import multinomial_label_sampling
from ensemblecalibration.nn_training.sampling import sample_uniform_instances, generate_weights_ens_dep


def generate_cone_predictors(x_inst: np.ndarray):

    n_samples, n_features = x_inst.shape
    
    # generate function values of the two ensemble predictors
    p_probs = np.zeros((n_samples, 2, 2))
    p_probs[:, 0, 0] = 0.5 * x_inst[:, 0]
    p_probs[:, 0, 1] = 1 - 0.5 * x_inst[:, 0]
    p_probs[:, 1, 0] = 1 - 0.5 * x_inst[:, 0]
    p_probs[:, 1, 1] = 0.5 * x_inst[:, 0]

    return p_probs

def cone_experiment_h0(n_samples: int, deg_fct: int, x_lower: float = 0.0, x_upper: float = 1.0,
                       n_dim: int = 1, w_1: float = 0.5):
    """binary ensemble prediction experiment where the null hypothesis is true, i.e. there exists a
    calibrated convex combination.

    Parameters
    ----------
    n_samples : int
        _description_
    deg_fct : int
        _description_
    x_lower : float, optional
        _description_, by default 0.0
    x_upper : float, optional
        _description_, by default 1.0
    n_dim : int, optional
        _description_, by default 1
    w_1 : float, optional
        _description_, by default 0.5

    Returns
    -------
    _type_
        _description_
    """

    x_inst = sample_uniform_instances(n_samples=n_samples, x_lower=x_lower, x_upper=x_upper, 
                                      n_dim=n_dim)
    
    n_samples, n_features = x_inst.shape
    # generate function values of the two ensemble predictors
    p_probs = generate_cone_predictors(x_inst=x_inst)

    # convex combination
    if deg_fct == 0:
        w_2 = 1 - w_1
        weights = np.zeros((n_samples, 2))
        weights[:, 0] = w_1
        weights[:, 1] = w_2
        p_bar = w_1 * p_probs[:, 0, :] + w_2 * p_probs[:, 1, :]
    else:
        weights = generate_weights_ens_dep(x_inst=x_inst, p_probs=p_probs, deg=deg_fct)
        p_bar = calculate_pbar(weights_l=weights, P=p_probs, reshape=True, n_dims=2)

    # sample labels accoridng to multinomial distribution
    y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_bar)

    return x_inst, p_probs, y_labels, p_bar, weights

def cone_experiment_h1(n_samples: int, x_lower: float = 0.0, x_upper : float = 1.0, n_dim: int = 1,
                       frac_in: float = 0.5):
    """binary ensemble prediction experiment where the alternative hypothesis is true, i.e. 
    there exists no calibrated convex combination.

    Parameters
    ----------
    n_samples : int
        number of samples
    x_lower : float, optional
        lower bound of the opssible values of the instances, by default 0.0
    x_upper : float, optional
        upper bound of the value of instances, by default 1.0
    n_dim : int, optional
        dimensionality of the instance space, by default 1
    frac_in : float, optional
        fraction of the calibrated predictor that lies within the cone spanned by the predcitions 
        of the ensemble members. by default 0.5. if it is 0.0, the prediction lies completely outside.

    Returns
    -------
    _type_
        _description_
    """
    x_inst = sample_uniform_instances(n_samples=n_samples, x_lower=x_lower, x_upper=x_upper,
                                      n_dim=n_dim)
    n_samples, n_features = x_inst.shape

    p_probs = generate_cone_predictors(x_inst=x_inst)

    # predictions for first and second class are dependent on the fraction that lies within the cone
    x_1 = 0.5 * frac_in
    x_2 = 1 - x_1

    p_bar = np.zeros((n_samples, 2))
    p_bar[:, 0] = x_1
    p_bar[:, 1] = x_2

    # sample labels
    y_labels = np.apply_along_axis(multinomial_label_sampling, 1, p_bar)

    return x_inst, p_probs, y_labels, p_bar


if __name__ == "__main__":
    x_inst, p_probs, y_labels, p_bar = cone_experiment_h1(n_samples=1000, x_lower=0.0, x_upper=1.0,)
    print(x_inst.shape)
    print(p_probs.shape)
    print(y_labels.shape)

    x_inst, p_probs, y_labels, p_bar, weights = cone_experiment_h0(n_samples=1000, deg_fct=0,
                                                                     x_lower=0.0, x_upper=1.0,
                                                                        n_dim=1, w_1=0.2)
    
    print(p_bar)




    




    






                                      