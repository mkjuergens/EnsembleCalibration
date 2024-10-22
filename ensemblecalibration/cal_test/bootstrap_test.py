from typing import Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ensemblecalibration.utils.helpers import multinomial_label_sampling, calculate_pbar
from ensemblecalibration.utils.minimization import calculate_min


def npbe_test_ensemble_v2(
    alpha: list,
    x_inst: np.ndarray,
    p_preds: np.ndarray,
    y_labels: np.ndarray,
    params: dict,
    verbose: bool = True,
):
    """new version of the bootstrapping test using uniform sampling of the polytope for testing
    whether there exists a calibrated version in the convex hull.
    This version uses the validation set to calculate the optimal weights.

    Parameters
    ----------
    alpha : list
        significance level(s) of the test
    x_inst : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing predictions for each instance and classifier
    p_preds : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing probabilistic predictions for each instance and classifier
    y_labels : np.ndarray of shape (n_samples,)
        array containing labels
    params : dict
        dictionary of test parameters
    Returns
    -------
    decision, (p_vals, stats)
        decision: integer defining whether tso reject (1) or accept (0) the null hypothesis
        ( p_vals: array of p values for each predictor )
        ( stats: array of test statistics for each predictor )

    """

    # calculate optimal weights
    _, _, p_bar_test, y_labels_test = calculate_min(x_inst, p_preds, y_labels, params,
                                                     verbose=verbose, val=True, test=False)
    
    # run bootstrap test
    decision, p_val, stat = npbe_test_vaicenavicius(alpha, p_bar_test, y_labels_test, params)
    print("Decision: ", decision)

    return decision, p_val, stat

def npbe_test_ensemble(
    alpha: list,
    x_inst: np.ndarray,
    p_preds: np.ndarray,
    y_labels: np.ndarray,
    params: dict,
    verbose: bool = True,
):
   
   
    """new version of the bootstrapping test using uniform sampling of the polytope for testing
    whether there exists a calibrated version in the convex hull

    Parameters
    ----------
    alpha : list
        significance level(s) of the test
    x_inst : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing predictions for each instance and classifier
    p_preds : np.ndarray of shape (n_samples, n_predictors, n_classes)
        tensor containing probabilistic predictions for each instance and classifier
    y_labels : np.ndarray of shape (n_samples,)
        array containing labels
    params : dict
        dictionary of test parameters
    Returns
    -------
    decision, (p_vals, stats)
        decision: integer defining whether tso reject (1) or accept (0) the null hypothesis
        ( p_vals: array of p values for each predictor )
        ( stats: array of test statistics for each predictor )

    """

    # calculate optimal weights
    _, _, p_bar_test, y_labels_test = calculate_min(x_inst, p_preds, y_labels, params,
                                                     verbose=verbose)
    
    # run bootstrap test
    decision, p_val, stat = npbe_test_vaicenavicius(alpha, p_bar_test, y_labels_test, params)
    print("Decision: ", decision)

    return decision, p_val, stat


def npbe_test_vaicenavicius(
    alpha: Union[list, float], p_probs: Union[np.ndarray, torch.Tensor], 
    y_labels: Union[np.ndarray, torch.Tensor], params: dict
):
    """
    Non-parametric bootstrapping test for a single classifier setting: 
    Vaicenavicius et al. (2019).

    Parameters
    ----------
    alpha: list | float
        Significance level(s) of the test
    p_probs : np.ndarray | torch.Tensor
        Tensor of probabilistic predictions of shape (n_instances, n_classes)
    y_labels : np.ndarray | torch.Tensor
        Labels
    params : dict
        Test parameters
    Returns
    -------
    decision, p_val, stat
        Decision of the test, p-value and value of the test statistic for the real data
    """

    # Ensure p_probs and y_labels are torch tensors
    if isinstance(p_probs, np.ndarray):
        p_probs = torch.from_numpy(p_probs).float()
    if isinstance(y_labels, np.ndarray):
        y_labels = torch.from_numpy(y_labels).long()

    # save values of bootstrap statistics here
    stats_h0 = torch.zeros(params["n_resamples"])

    # Precompute the range for sampling
    n_instances = p_probs.size(0)

    # Iterate over bootstrap samples
    for b in range(params["n_resamples"]):
        # extract bootstrap sample directly using torch sampling
        indices = torch.randint(0, n_instances, (n_instances,))
        p_probs_b = p_probs[indices]
        
        # sample labels according to categorical distribution
        y_b = multinomial_label_sampling(p_probs_b, tensor=True)
        
        # calculate test statistic under null hypothesis
        stats_h0[b] = params["obj"](p_probs_b, y_b, params)

    # Calculate statistic on real data
    stat = params["obj"](p_probs, y_labels, params)
    
    # Convert stat to numpy if needed
    stat = stat.detach().cpu().numpy() if isinstance(stat, torch.Tensor) else stat

    # Calculate alpha-quantile of empirical distribution of the test statistic
    q_alpha = np.quantile(stats_h0.cpu().numpy(), 1 - np.array(alpha))

    # Decision: reject test if stat > q_alpha
    decision = (np.abs(stat) > q_alpha).astype(int).tolist()

    # P-value: fraction of bootstrap samples larger than the real statistic
    p_val = (stats_h0 > torch.tensor(stat)).float().mean().item()

    return decision, p_val, stat


def multinomial_label_sampling(probs, tensor: bool = False):
    """
    Draw samples from a categorical distribution defined by a probability matrix.
    Parameters
    ----------
    probs : np.ndarray or torch.Tensor
        Probability matrix of shape (n_samples, n_classes) where rows sum to one.
    tensor : bool, optional
        Whether to return the output as a torch tensor, by default False
    Returns
    -------
    np.ndarray or torch.Tensor
        Array or tensor of sampled class labels for each instance.
    """
    if isinstance(probs, np.ndarray):
        probs = torch.from_numpy(probs).float()

    # Use PyTorch's multinomial to sample from each row of the probability matrix
    y = torch.multinomial(probs, num_samples=1).squeeze(1)

    return y if tensor else y.numpy()


def test_train_val_split(
    p_preds: Union[np.ndarray, torch.Tensor],
    y_labels: Union[np.ndarray, torch.Tensor],
    x_inst: Union[np.ndarray, torch.Tensor],
    test_size: float = 0.5,
    val_size: float = 0.2,
):
    # Ensure inputs are numpy arrays for compatibility with train_test_split
    if isinstance(p_preds, torch.Tensor):
        p_preds = p_preds.cpu().numpy()
    if isinstance(y_labels, torch.Tensor):
        y_labels = y_labels.cpu().numpy()
    if isinstance(x_inst, torch.Tensor):
        x_inst = x_inst.cpu().numpy()

    x_test, X_temp, y_test, y_temp, preds_test, predictions_temp = train_test_split(
        x_inst, y_labels, p_preds, test_size=test_size, random_state=42
    )

    x_train, x_val, y_train, y_val, preds_train, preds_val = train_test_split(
        X_temp, y_temp, predictions_temp, test_size=val_size, random_state=42
    )

    return (
        (x_test, y_test, preds_test),
        (x_train, y_train, preds_train),
        (x_val, y_val, preds_val),
    )
