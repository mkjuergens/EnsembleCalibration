import torch
from torch.utils.data import random_split

from src.data.synthetic.binary_new import BinaryExperiment
from src.data.dataset import MLPDataset

def create_synthetic_dataset(dataset_cfg):
    """
    Creates a synthetic dataset (train, val, test) according to dataset_cfg.
    This might use your 'BinaryExperiment' class that has .generate_data() 
    for GP or logistic synthetic data.

    Parameters
    ----------
    dataset_cfg : dict
        Example:
          {
            "method": "gp",
            "n_samples": 5000,
            "n_ens": 5,
            "scale_noise": 0.5,
            "kernel_width": 0.1,
            ...
          }

    Returns
    -------
    (train_set, val_set, test_set): Torch dataset objects
        Each one yields something like (p_preds, y, x) or (x, y), 
        depending on how your pipeline is set up.
    """ 
    
    method = dataset_cfg["method"]  # "gp" or "logistic"
    n_samples = dataset_cfg["n_samples"]
    n_ens = dataset_cfg.get("n_ens", 5)
    scale_noise = dataset_cfg["scale_noise"]
    offset_range = dataset_cfg["offset_range"]
    kernel_width = dataset_cfg.get("kernel_width", 0.5)

    # create the experiment
    exp = BinaryExperiment(
        method=method,
        n_samples=n_samples,
        n_ens=n_ens,
        scale_noise=scale_noise,
        offset_range=offset_range,
        kernel_width=kernel_width
        # add any other needed init args
    )
    exp.generate_data() 

    x_inst = exp.x_inst  # shape (N,1)
    y_labels = exp.y_labels  # shape (N,)
    p_preds = torch.tensor(exp.ens_preds, dtype=torch.float32)
    p_true = torch.tensor(exp.p_true, dtype=torch.float32)

    # 3) Build a single 'master' TensorDataset
    #    Suppose your pipeline wants a tuple (p_preds, y, x).
    #    Adjust as needed for your pipeline.
    master_dataset = MLPDataset(x_train=x_inst, P=p_preds, y=y_labels, p_true=p_true)
    # 4) Split into train/val/test
    #    We'll do an 80/10/10 split as an example
    total_n = len(master_dataset)
    n_train = int(0.8 * total_n)
    n_val   = int(0.1 * total_n)
    n_test  = total_n - n_train - n_val

    train_set, val_set, test_set = random_split(
        master_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    return train_set, val_set, test_set