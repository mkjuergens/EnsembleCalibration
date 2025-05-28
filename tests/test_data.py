import pytest
import numpy as np

from src.data.synthetic.gp_binary import exp_gp
from src.data.synthetic.multiclass_dirichlet_new import syn_exp_multiclass

def test_sample_dirichlet():
    n_samples = 1000
    n_ens = 5
    n_classes = 3
    bounds = [0.0, 5.0]
    # sample data for null hypothesis
    x_inst, p_preds, p_bar, y_labels, weights_l = syn_exp_multiclass(
        n_samples=n_samples, n_classes=n_classes, n_predictors=n_ens, h0=True
    )
        
    assert p_preds.shape == (n_samples, n_ens, n_classes)
    assert weights_l.shape == (n_samples, n_ens)

    # sample data for alternative hypothesis
    x_inst, p_preds, p_bar, y_labels = syn_exp_multiclass(
        n_samples=n_samples, n_classes=n_classes, n_predictors=n_ens, h0=False
    )
    assert p_preds.shape == (n_samples, n_ens, n_classes)
    assert p_bar.shape == (n_samples, n_classes)
    assert y_labels.shape == (n_samples,)
    assert np.allclose(p_bar.sum(axis=1), 1.0)


if __name__ == "__main__":
    pytest.main()