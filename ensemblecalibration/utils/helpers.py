import os

import torch
import re
import numpy as np
import pandas as pd
from scipy.stats import multinomial
from sklearn.model_selection import train_test_split


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


def multinomial_label_sampling(probs, tensor: bool = False):
    """
    Draws samples y from the categorical distribution defined by a probability matrix.

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

    # Ensure probs is a 2D tensor
    if probs.ndim == 1:
        probs = probs.unsqueeze(0)

    # Use PyTorch's multinomial to sample from each row of the probability matrix
    y = torch.multinomial(probs, num_samples=1).squeeze(1)  # Sampling 1 value per row

    if not tensor:
        y = y.numpy()

    return y


def test_train_val_split(
    p_preds: np.ndarray,
    y_labels: np.ndarray,
    x_inst: np.ndarray,
    test_size: float = 0.5,
    val_size: float = 0.2,
):

    x_test, X_temp, y_test, y_temp, preds_test, predictions_temp = train_test_split(
        x_inst, y_labels, p_preds, test_size=test_size, random_state=42
    )

    # Step 2: Split Temp into Validation (20%) and Test (20%)
    x_train, x_val, y_train, y_val, preds_train, preds_val = train_test_split(
        X_temp, y_temp, predictions_temp, test_size=val_size, random_state=42
    )

    return (
        (x_test, y_test, preds_test),
        (x_train, y_train, preds_train),
        (x_val, y_val, preds_val),
    )


# def multinomial_label_sampling(probs: np.ndarray, tensor: bool = False):
#     """draws a sample y from the categorical distribution
#     defined by a probaibility vector.

#     Parameters
#     ----------
#     probs : np.ndarray
#         probability vector that sums up to one

#     Returns
#     -------
#     np.ndarray or torch.tensor

#     """
#     try:
#         draws = multinomial(1, probs).rvs(size=1)[0, :]
#         y = np.argmax(draws)

#     except ValueError as e:
#         y = np.argmax(probs)

#     if tensor:
#         y = torch.tensor(y).long()

#     return y


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


def ab_scale(x, a: float, b: float):
    """
    Scales array x to the range [a, b].

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Array or tensor to be scaled.
    a : float
        Lower bound.
    b : float
        Upper bound.

    Returns
    -------
    np.ndarray or torch.Tensor
        Scaled array or tensor.
    """
    is_tensor = isinstance(x, torch.Tensor)
    if not is_tensor:
        # If input is a numpy array
        x_min, x_max = np.min(x), np.max(x)
        scaled = ((b - a) * ((x - x_min) / (x_max - x_min))) + a
    else:
        # If input is a torch tensor
        x_min, x_max = torch.min(x), torch.max(x)
        scaled = ((b - a) * ((x - x_min) / (x_max - x_min))) + a

    return scaled


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


def is_data_encapsulated(data):
    """function that checks if the data is encapsulated in a list of lists of lists,
    which is needed for the results to be saved in a csv file.

    Parameters
    ----------
    data : list or some other data structure
        list of results

    Returns
    -------
    bool
        True if data is encapsulated, False otherwise
    """
    # Check if data is a list
    if isinstance(data, list) and len(data) > 0:
        # Check if the first element is a list
        if isinstance(data[0], list):
            # Check if the elements inside the first element are also lists
            if len(data[0]) > 0 and isinstance(data[0][0], list):
                # Data is encapsulated (list of lists of lists)
                return True
            else:
                # Data is not encapsulated (list of lists)
                return False
    return False


def save_results(results_list, save_dir, file_name: str, col_names: list):
    """saves a list of results to a csv file. The list has to be in an encapsulated format

    Parameters
    ----------
    results_list : list
        list of results
    save_dir : str
        directory where the results will
    file_name : str
        name of the file
    col_names : list
        list of column names
    """

    # create directory for results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # filename
    # file_name = prefix + "_{}.csv".format(n_resamples)
    save_dir_file = os.path.join(save_dir, file_name)
    # encapulated data if needed
    if not is_data_encapsulated(results_list):
        results_list = [results_list]
    # create dataframe
    results_df = pd.DataFrame(results_list)
    results_df.columns = col_names
    # save results
    results_df.to_csv(save_dir_file, index=False)


# Function to make the config serializable
def make_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(element) for element in obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif callable(obj):
        # For functions and classes
        return getattr(obj, "__name__", repr(obj))
    elif hasattr(obj, "__class__"):
        # For class instances, return the class name
        return obj.__class__.__name__
    else:
        # Fallback to string representation
        return repr(obj)
