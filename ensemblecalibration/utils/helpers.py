import torch
import re
import numpy as np
from scipy.stats import multinomial


def matrix_kernel(p: np.ndarray, q: np.ndarray, dist_fct, sigma: float = 2.0):
    """returns the matrix-valued kernel evaluated at two point predictions

    Parameters
    ----------
    p : np.ndarray
        first point prediction
    q : np.ndarray
        second point prediction
    sigma : float
        bandwidth
    dist_fct : _type_
        distance measure. Options: {tv_distance, l2_distance}

    Returns
    -------
    np.ndarray
        _description_
    """
    p = p.squeeze()
    q = q.squeeze()

    assert len(p) == len(q), "vectors need to be of the same length"
    id_k = np.eye(len(p))  # identity matrix
    return np.exp((-1 / sigma) * (dist_fct(p, q) ** 2) * id_k)


def multinomial_label_sampling(probs: np.ndarray, tensor: bool = False):
    """draws a sample y from the categorical distribution
    defined by a probaibility vector.

    Parameters
    ----------
    probs : np.ndarray
        probability vector that sums up to one

    Returns
    -------
    np.ndarray or torch.tensor

    """
    try:
        draws = multinomial(1, probs).rvs(size=1)[0, :]
        y = np.argmax(draws)

    except ValueError as e:
        y = np.argmax(probs)

    if tensor:
        y = torch.tensor(y, dtype=torch.long)

    return y


def calculate_pbar(
    weights_l: torch.Tensor,
    p_preds: torch.Tensor,
    reshape: bool = False,
    n_dims: int = 2,
):
    """function to calculate the tensor of convex combinations. Taeks as input
    the weights (per instance) and the tensor of probabilistic predictions.

    Parameters
    ----------
    weights_l : torch.Tensor
        weight tensor of shape (n_samples, n_predictors) or just (n_predictors,)
    p_preds : torch.Tensor
        tensor containing all predictions per instance and predictor,
        of shape (n_samples, n_predictors, n_classes)
    reshape : bool, optional
        whether to reshape the weights. Only needed in case of instance-dependency.
         By default False
    n_dims : int, optional
        number of dimensions of the weight tensor. If 2, we have instance-wise dependency of
        the convex combination P_bar. By default 2

    Returns
    -------
    torch.Tensor or np.ndarray
        tensor of shape (n_samples, n_classes) containing the convex combinations
    """

    # number of samples for which we have predictions
    n_inst = p_preds.shape[0]
    if reshape:
        assert (
            len(weights_l) % n_inst == 0
        ), " weight vector needs to be a multiple of the number of rows"
        weights_l = weights_l.reshape(n_inst, -1)

    # assert (
    #     weights_l.shape[:2] == p_preds.shape[:2]
    # ), f"number of samples need to be the same for P and weights_l, but are {weights_l.shape[0]} and {p_preds.shape[0]} respectively"
    if n_dims == 2:
        # calculate convex combination: sum over the second dimension
        p_bar = torch.sum(weights_l.unsqueeze(2) * p_preds, dim=1)
    elif n_dims == 1:
        # convert to numpy array if needed
        if isinstance(weights_l, torch.Tensor):
            weights_l = weights_l.numpy()
        if isinstance(p_preds, torch.Tensor):
            p_preds = p_preds.numpy()
        p_bar = np.matmul(np.swapaxes(p_preds, 1, 2), weights_l)

    return p_bar


def sample_function(x: np.ndarray, deg: int = 1, ivl: tuple = (0, 1)):
    """
    Arguments:
      x : ndarray (n_samples,)
        Inputs.
      deg: int (default=1)
        Degree of polynomial function.

    Output:
      y : ndarray (n_samples,)
        Function values.
    """

    y = np.polyval(np.polyfit(x, np.random.rand(len(x)), deg), x)
    # use min max scaling to ensure values in [0,1]
    y = ab_scale(y, ivl[0], ivl[1])
    return y


def ab_scale(x: np.ndarray, a: float, b: float):
    """scales array x to [a,b]
    Parameters
    ----------
    x : np.ndarray
        Array to be scaled.
    a : float
        Lower bound.
    b : float
        Upper bound.

    Returns
    -------
    np.ndarray
        Scaled array.
    """
    return ((b - a) * ((x - np.min(x)) / (np.max(x) - np.min(x)))) + a


def clean_and_convert(s):
    # Extract numbers using regular expression
    numbers = re.findall(r"np\.float64\((.*?)\)", s)
    # Convert to list of floats
    return [float(num) for num in numbers]


def process_df(df):

    # apply clean and convert to all columns
    for col in df.columns:
        try:
            df[col] = df[col].apply(clean_and_convert)
        except:
            pass

    return df
