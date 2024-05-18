import numpy as np
from ensemblecalibration.meta_model import get_optim_lambda_mlp
from ensemblecalibration.data.dataset import MLPDataset
from ensemblecalibration.utils.helpers import calculate_pbar

def calculate_min(x_inst: np.ndarray, p_probs: np.ndarray, y_labels: np.ndarray,
                                 params: dict):
    if params["optim"] == "mlp":
        dataset = MLPDataset(x_train=x_inst, P=p_probs, y=y_labels)
        l_weights, loss = get_optim_lambda_mlp(dataset, loss=params["loss"], n_epochs=params["n_epochs"],
                                         lr=params["lr"], batch_size = params["batch_size"],
                                         hidden_dim=params["hidden_dim"],
                                         hidden_layers=params["hidden_layers"],
                                         patience=params["patience"])
    else:
        raise NotImplementedError("Only MLP is implemented")
    # calculate p_bar
    p_bar = calculate_pbar(l_weights, p_probs, reshape=False)
    # calculate test statistic
    minstat = params["obj"](p_bar, y_labels, params)
    
    return minstat, l_weights