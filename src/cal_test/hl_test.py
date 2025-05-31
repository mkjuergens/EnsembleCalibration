from typing import Optional, Union, Tuple
import numpy as np
from scipy.stats import chi2

from src.utils.minimization import calculate_min


def hl_test_stat(
    p_preds: np.ndarray,
    y_labels: np.ndarray,
    params: dict,
    return_p_val: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Multiclass Hosmer-Lemeshow χ² test.
    params["n_bins"]       : number of probability bins, default 10.
    """
    n_bins = params.get("n_bins", 10)
    n_classes = p_preds.shape[1]

    # sort descending by 1-P(reference class) (here class 0)
    idx = np.argsort(1 - p_preds[:, 0])[::-1]
    bins = np.array_split(idx, n_bins)

    stat = 0.0
    for k in range(n_classes):
        for b in bins:
            o_bk = np.sum(y_labels[b] == k)
            p_bk = np.sum(p_preds[b, k])
            if p_bk > 0:  # avoid div-by-zero
                stat += (o_bk - p_bk) ** 2 / p_bk

    df = (n_bins - 2) * (n_classes - 1)
    pval = 1.0 - chi2.cdf(stat, df)

    return (stat, pval) if return_p_val else stat


def hl_test(alpha, p_bar, y_labels, params):
    """Hosmer-Lemeshow test for calibration of predicted probabilities.

    Parameters
    ----------
    alpha : list or float
        Significance level(s) for the test. If a single float is provided, it is used for all bins.
    p_bar : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    y_labels : np.ndarray
        True labels of shape (n_samples,).
    params : dict
        Parameters for the test, including 'n_bins' for the number of bins.
    Returns
    -------
    decision : list
        List of decisions (1 for reject, 0 for do not reject) for each significance level.
    pval : float
        p-value of the test statistic.
    stat : float
        Test statistic value.
    """
    stat, pval = hl_test_stat(p_bar, y_labels, params, return_p_val=True)
    # reject if p-value < alpha   →   decision = 1 if reject
    decision = (np.asarray(alpha) > pval).astype(int).tolist()
    return decision, pval, stat



def hl_test_ensemble_v2(
    alpha: list,
    x_inst: np.ndarray,
    p_preds: np.ndarray,
    y_labels: np.ndarray,
    params: dict,
    verbose: bool = True,
    use_val: bool = True,
    use_test: bool = True,
):
    """Ensemble version of the Hosmer-Lemeshow test for calibration of predicted probabilities.

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
    decision, p_val, stat = hl_test(
        alpha=alpha,
        p_bar=p_bar_test,
        y_labels=y_labels_test,
        params=params,
    )
    if verbose:
        print(f"Shape of p_bar_test: {p_bar_test.shape}")
        print(f"Shape of instance values: {x_inst.shape}")
        print("Decision: ", decision)

    return decision, p_val, stat