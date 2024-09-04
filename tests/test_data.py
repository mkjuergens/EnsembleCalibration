import pytest
import torch
import numpy as np

from ensemblecalibration.data.gp_binary import exp_gp
from ensemblecalibration.data.multiclass_dirichlet import (
    exp_dirichlet,
    sample_dirichlet_h1,
    sample_dir_params,
)

def test_sample_dirichlet():
    n_samples = 1000
    n_ens = 5
    n_classes = 3
    bounds = [0.0, 5.0]
    # sample data
    x_inst, p_preds, p_bar, y_labels, weights_l = (
        exp_dirichlet(
            n_samples=n_samples,
            n_classes=n_classes,
            n_members=n_ens,
            x_bound=bounds,
            h0=True,
            x_dep=True,
        )
    )
    assert p_preds.shape == (n_samples, n_ens, n_classes)
    assert weights_l.shape == (n_samples, n_ens)

def test_sample_dirichlet_h1():
    n_samples = 1000
    n_ens = 5
    n_classes = 3
    bounds = [0.0, 5.0]
    # sample data
    # sample x_inst from uniform distribution
    x_inst = torch.tensor(
        np.random.uniform(bounds[0], bounds[1], n_samples), dtype=torch.float32
    )
    p_preds, p_bar, y_labels = (
        sample_dirichlet_h1(x_inst=x_inst, n_ens=n_ens, n_classes=n_classes)
    )
    assert p_preds.shape == (n_samples, n_ens, n_classes)
    assert p_bar.shape == (n_samples, n_classes)
    assert y_labels.shape == (n_samples,)

if __name__ == "__main__":
    pytest.main()