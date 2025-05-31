from typing import Union

import numpy as np
import torch

from src.utils.helpers import multinomial_label_sampling
from src.utils.minimization import calculate_min, minimize_const_weights


def npbe_test_ensemble_v2(
    alpha: list,
    x_inst: np.ndarray,
    p_preds: np.ndarray,
    y_labels: np.ndarray,
    params: dict,
    verbose: bool = True,
    use_val: bool = True,
    use_test: bool = True,
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
    verbose : bool, optional
        whether to print the results and loss, by default True
    use_val : bool, optional
        whether to use the validation set for training, by default False
    Returns
    -------
    decision, (p_vals, stats)
        decision: integer defining whether tso reject (1) or accept (0) the null hypothesis
        ( p_vals: array of p values for each predictor )
        ( stats: array onf test statistics for each predictor )
    """

    # calculate optimal weights
    _, _, _, p_bar_test, y_labels_test, _ = calculate_min(
        x_inst, p_preds, y_labels, params, verbose=verbose, val=use_val, test=use_test
    )

    # run bootstrap test
    decision, p_val, stat = npbe_test_vaicenavicius(
        alpha, p_bar_test, y_labels_test, params
    )
    if verbose:
        print(f"Shape of p_bar_test: {p_bar_test.shape}")
        print(f"Shape of instance values: {x_inst.shape}")
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
    _, _, _, p_bar_test, y_labels_test, _ = calculate_min(
        x_inst, p_preds, y_labels, params, verbose=verbose
    )

    # run bootstrap test
    decision, p_val, stat = npbe_test_vaicenavicius(
        alpha, p_bar_test, y_labels_test, params
    )
    # print("Decision: ", decision)

    return decision, p_val, stat


def npbe_test_vaicenavicius(
    alpha: Union[list, float],
    p_probs: Union[np.ndarray, torch.Tensor],
    y_labels: Union[np.ndarray, torch.Tensor],
    params: dict,
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
    q_alpha = np.quantile(stats_h0.cpu().detach().numpy(), 1 - np.array(alpha))

    # Decision: reject test if stat > q_alpha
    decision = (np.abs(stat) > q_alpha).astype(int).tolist()

    # P-value: fraction of bootstrap samples larger than the real statistic
    p_val = (stats_h0 > torch.tensor(stat)).float().mean().item()

    return decision, p_val, stat





# def test_train_val_split(
#     p_preds: Union[np.ndarray, torch.Tensor],
#     y_labels: Union[np.ndarray, torch.Tensor],
#     x_inst: Union[np.ndarray, torch.Tensor],
#     test_size: float = 0.5,
#     val_size: float = 0.2,
# ):
#     # Ensure inputs are numpy arrays for compatibility with train_test_split
#     if isinstance(p_preds, torch.Tensor):
#         p_preds = p_preds.cpu().numpy()
#     if isinstance(y_labels, torch.Tensor):
#         y_labels = y_labels.cpu().numpy()
#     if isinstance(x_inst, torch.Tensor):
#         x_inst = x_inst.cpu().numpy()

#     x_test, X_temp, y_test, y_temp, preds_test, predictions_temp = train_test_split(
#         x_inst, y_labels, p_preds, test_size=test_size, random_state=42
#     )

#     x_train, x_val, y_train, y_val, preds_train, preds_val = train_test_split(
#         X_temp, y_temp, predictions_temp, test_size=val_size, random_state=42
#     )

#     return (
#         (x_test, y_test, preds_test),
#         (x_train, y_train, preds_train),
#         (x_val, y_val, preds_val),
#     )


def npbe_test_ensemble_v0(
    alpha: list,
    x_inst: torch.tensor,
    p_preds: torch.tensor,
    y_labels: torch.tensor,
    params: dict,
    verbose: bool = False,
    use_val: bool = True,
    use_test: bool = True,
):
    """
    old version of the bootstrapping test using uniform sampling of the polytope for testing
    """
    # solve minimization problem, return predictions on test data as well as test data
    _, _, _, p_bar_test, y_labels_test, p_preds_test = calculate_min(
        x_inst,
        p_preds,
        y_labels,
        params,
        verbose=verbose,
        val=use_val, # set val and test to False because we purely see it as an optimization problem
        test=use_test,
    )

    # stats = np.zeros(params["n_resamples"])
    # for b in range(params["n_resamples"]):
    #     # extract bootstrap sample
    #     P_b = np.random.sample(p_preds_test.tolist(), p_preds_test.shape[0])
    #     P_b = np.stack(P_b)
    #     # sample predictions using a predefined sampling method
    #     P_bar_b = sample_p_bar(p_probs=P_b)
    #     # round to 5 decimal digits for numerical stability
    #     P_bar_b = np.trunc(P_bar_b * 10**3) / (10**3)
    #     P_bar_b = np.clip(P_bar_b, 0, 1)
    #     y_b = multinomial_label_sampling(P_bar_b)
    #     # np.apply_along_axis(multinomial_label_sampling, 1, P_bar_b)
    #     # perform test
    #     stats[b] = params["test"](P_bar_b, y_b, params)

    # # calculate 1-alpha quantile from sampling distribution
    # q_alpha = np.quantile(stats, 1 - (np.array(alpha)))

    # # solve minimization problem
    # _, _, p_bar_test, y_labels_test = calculate_min(
    #     x_inst, p_preds, y_labels, params, verbose=verbose, val=True, test=False
    # )

    # if params["optim"] == "neldermead":
    #     if params["x_dependency"]:
    #         l = solve_neldermead2D(P, y, params)
    #     else:
    #         l = solve_neldermead1D(P, y, params)
    # elif params["optim"] == "COBYLA":
    #     if params["x_dependency"]:
    #         l = solve_cobyla2D(P, y, params)
    #     else:
    #         l = solve_cobyla1D(P, y, params)

    # else:
    # #     raise NotImplementedError

    # stat = params["obj"](p_bar_test, y_labels_test, params)
    # decision = list(map(int, np.abs(stat) > q_alpha))
    # p_val = np.mean(stats > stat)

    n_resamples = params["n_resamples"]
    stats = torch.zeros(n_resamples, device=p_preds_test.device)
    n_samples = p_preds_test.shape[0]

    for b in range(n_resamples):
        # Extract bootstrap sample indices with replacement
        indices = torch.randint(0, n_samples, (n_samples,), device=p_preds_test.device)
        P_b = p_preds_test[indices]  # Shape: (n_samples, n_ensembles, n_classes)

        # Sample predictions using a predefined sampling method
        P_bar_b = sample_p_bar_torch(p_probs=P_b)

        # Round to 5 decimal digits for numerical stability
        P_bar_b = torch.trunc(P_bar_b * 1e3) / 1e3
        P_bar_b = torch.clamp(P_bar_b, 0, 1)

        # Sample labels
        y_b = multinomial_label_sampling(P_bar_b, tensor=True)

        # Perform test
        stats[b] = params["obj"](P_bar_b, y_b, params)

    # Calculate 1-alpha quantile from sampling distribution
    q_alpha = torch.quantile(stats, 1 - torch.tensor(alpha, device=p_preds_test.device))

    # Solve minimization problem again
    # _, _, p_bar_test, y_labels_test = calculate_min(
    #     x_inst, p_preds, y_labels, params, verbose=verbose, val=True, test=False
    # )

    # Calculate test statistic
    stat = params["obj"](p_bar_test, y_labels_test, params)

    # Make decision
    decision = (torch.abs(stat) > q_alpha).int().tolist()
    p_val = (stats > stat).float().mean().item()

    return decision, p_val, stat


def sample_p_bar(p_probs: np.ndarray):
    """function for sampling new convex combinations of the predictions using coefficinets
    sampled uniformly from the (K-1) simplex.

    Parameters
    ----------
    P : np.ndarray of shape(n_predictors, n_classes) or
     (n_samples, n_predictors, n_classes)
        matrix containing the ensemble predictors
    size : int
        number of samples to generate

    Returns
    -------
    np.ndarray of shape (size, n_classes)
        array containing samples
    """
    if p_probs.ndim == 2:
        M, K = p_probs.shape
    elif p_probs.ndim == 3:
        N, M, K = p_probs.shape

    lambdas = np.random.dirichlet([1] * M, size=1)[0, :]
    preds_new = lambdas @ p_probs

    return preds_new


def sample_p_bar_torch(p_probs: torch.Tensor):
    """
    Sample new convex combinations of the predictions using coefficients
    sampled uniformly from the simplex.

    Parameters
    ----------
    p_probs : torch.Tensor of shape (n_samples, n_ensembles, n_classes)

    Returns
    -------
    torch.Tensor of shape (n_samples, n_classes)
    """
    n_samples, n_ensembles, n_classes = p_probs.shape

    # Sample lambdas from Dirichlet distribution for each sample
    lambdas = torch.distributions.Dirichlet(torch.ones(n_ensembles, device=p_probs.device)).sample((n_samples,))

    # Compute convex combination for each sample
    P_bar = torch.einsum('ij,ijk->ik', lambdas, p_probs)

    return P_bar
