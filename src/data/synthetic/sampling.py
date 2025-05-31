from typing import Tuple, Optional, Any
from typing import List
import torch
import numpy as np
from torch import Tensor
from torch.distributions import Dirichlet, Categorical
from sklearn.metrics.pairwise import rbf_kernel

from src.utils.helpers import calculate_pbar, ab_scale, multinomial_label_sampling


def sample_dirichlet_experiment(
    n_samples: int,
    n_classes: int,
    n_members: int,
    x_bound: Tuple[float, float],
    x_dep: bool,
    h0: bool,
    deg: int = 2,
    uncertainty: float = 0.1,
    deg_h1: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """_summary_

    Parameters
    ----------
    n_samples : int
        _description_
    n_classes : int
        _description_
    n_members : int
        _description_
    x_bound : Tuple[float, float]
        _description_
    x_dep : bool
        _description_
    h0 : bool
        _description_
    deg : int
        _description_
    uncertainty : float, optional
        _description_, by default 0.1
    deg_h1 : Optional[float], optional
        _description_, by default None

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]
        _description_
    """
    # Sample instance features
    x = torch.rand(n_samples, 1) * (x_bound[1] - x_bound[0]) + x_bound[0]
    # Dirichlet prior for ensemble members
    dir_prior = Dirichlet(torch.ones(n_classes, device=x.device))
    dir_params = dir_prior.concentration * n_classes / uncertainty
    # Sample constant predictions per member and repeat across instances
    p_preds = Dirichlet(dir_params).sample((n_members,))  # (M, K)
    p_preds = p_preds.unsqueeze(0).expand(n_samples, -1, -1)  # (N, M, K)

    if h0:
        p_bar, weights = sample_pbar_h0_generic(x, p_preds, x_dep, deg)
    else:
        p_bar = sample_pbar_h1_generic(x, p_preds, setting=None, deg=deg_h1)
        weights = None

    y = Categorical(probs=p_bar).sample()
    return x, p_preds, p_bar, y, weights


def sample_gp_experiment(
    n_samples: int,
    x_bound: Tuple[float, float],
    bounds_p: List[List[float]],
    x_dep: bool,
    h0: bool,
    deg: int,
    kernel: Any = rbf_kernel,
    setting: Optional[int] = None,
    deg_h1: Optional[float] = None,
    **kernel_kwargs
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """_summary_

    Parameters
    ----------
    n_samples : int
        _description_
    x_bound : Tuple[float, float]
        _description_
    bounds_p : list
        _description_
    kernel : _type_
        _description_
    x_dep : bool
        _description_
    h0 : bool
        _description_
    deg : int
        _description_
    setting : Optional[int], optional
        _description_, by default None
    deg_h1 : Optional[float], optional
        _description_, by default None

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]
        _description_
    """
    # 1) sample inputs
    x_np = np.random.uniform(x_bound[0], x_bound[1], n_samples)
    x = torch.from_numpy(x_np).float().view(-1, 1)

    # 2) for each [min, max] in bounds_p, get one GP‐draw p, then make [p,1-p]
    preds_list = []
    for bnd in bounds_p:
        p0 = gp_sample_prediction(
            x_np, kernel, bnd, **kernel_kwargs
        )  # returns (N,) np or torch
        if not isinstance(p0, torch.Tensor):
            p0 = torch.from_numpy(p0).float()
        p1 = 1.0 - p0
        # now stack into (N,2)
        two_class = torch.stack([p0, p1], dim=1)
        preds_list.append(two_class)

    # 3) stack into (N, M, 2)
    p_preds: Tensor = torch.stack(preds_list, dim=1)

    # 4) do the H0/H1 logic as before
    if h0:
        p_bar, weights = sample_pbar_h0_generic(x, p_preds, x_dep, deg)
    else:
        p_bar = sample_pbar_h1_generic(x, p_preds, setting=setting, deg=deg_h1)
        weights = None

    # 5) sample labels
    y = multinomial_label_sampling(p_bar, tensor=True).view(-1)
    return x, p_preds, p_bar, y, weights


def sample_pbar_h0_generic(
    x: Tensor, p_preds: Tensor, x_dep: bool, deg: int
) -> Tuple[Tensor, Tensor]:
    N, M, K = p_preds.shape
    if x_dep:
        # simple polynomial on x for first weight
        coeffs = torch.randn(deg + 1, device=x.device)
        poly = polyval_torch(coeffs, x.squeeze())
        w0 = torch.sigmoid(poly).unsqueeze(1)
    else:
        w0 = torch.rand(1, device=x.device).expand(N, 1)
    if M == 2:
        weights = torch.cat([w0, 1 - w0], dim=1)
    else:
        weights = torch.softmax(torch.rand(N, M, device=x.device), dim=1)
    p_bar = calculate_pbar(weights, p_preds)
    return p_bar, weights


def sample_pbar_h1_generic(
    x: Tensor,
    p_preds: Tensor,
    setting: Optional[int] = None,
    deg: Optional[float] = 0.1,
    **kwargs
) -> Tensor:
    N, M, K = p_preds.shape
    # Choose random class per instance
    c = torch.randint(0, K, (N,), device=x.device)
    p_c = torch.eye(K, device=x.device)[c]
    # Uniform mixture as boundary proxy
    p_b = calculate_pbar(torch.full((N, M), 1 / M, device=x.device), p_preds)
    p_bar = deg * p_c + (1 - deg) * p_b
    return p_bar


def gp_sample_prediction(
    x: np.ndarray, kernel: Any, bounds_p: Tuple[float, float], **kernel_kwargs
) -> Tensor:
    cov = kernel(x.reshape(-1, 1), x.reshape(-1, 1), **kernel_kwargs)
    mean = np.zeros_like(x)
    samp = np.random.multivariate_normal(mean, cov)
    scaled = ab_scale(samp, bounds_p[0], bounds_p[1])
    return torch.from_numpy(scaled).float()



def polyval_torch(coeffs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Evaluate a 1-D polynomial at points `z` using Horner’s rule.
    `coeffs` shape = (d+1,), highest power first.
    """
    result = torch.zeros_like(z, dtype=coeffs.dtype, device=z.device)
    for c in coeffs:
        result = result * z + c
    return result
