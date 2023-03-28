from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ensemblecalibration.nn_training.distances import tv_distance_tensor
from ensemblecalibration.nn_training.model import MLPCalW, DirichletDataset
from ensemblecalibration.nn_training.losses import SKCELoss
from ensemblecalibration.calibration.experiments import (
    experiment_h0_feature_dependency,
    experiment_h0_feature_dependency,
)

def get_optim_lambda_mlp(dataset_train: torch.utils.data.Dataset, loss,
                     n_epochs: int = 100, lr: float = 0.001):
    
    model, loss_train = train_mlp(dataset_train=dataset_train, loss=loss, n_epochs=n_epochs,
                                  lr=lr, print_losses=False)
    p_probs = torch.from_numpy(dataset_train.p_probs).float()
    optim_weights = model(p_probs)
    optim_weights = optim_weights.detach().numpy()
    return optim_weights

def train_mlp(
    dataset_train: torch.utils.data.Dataset, loss, dataset_val: Optional[torch.utils.data.Dataset] = None, 
    n_epochs: int = 100, lr: float = 0.001, batch_size: int = 128,
    print_losses: bool = True):
    """trains the MLP model to predict the optimal weight matrix for the given ensemble model
    such that the calibration error of the convex combination is minimized.

    Parameters
    ----------
    dataset_train : torch.utils.data.Dataset
        dataset containing probabilistic predictions of ensembnle members used for training
    datasset_val : torch.utils.data.Dataset
        validation dataset
    loss : _type_
        loss taking a tuple of the probabilistic predictions, the weights of the convex combination 
        and the labels as input     
    n_epochs : int
        number of training epochs
    lr : float
        learning rate
    batch_size : int, optional
        _description_, by default 128
    print_losses : bool, optional
        whether to print train and validation loss at every epoch, by default True

    Returns
    -------
    model, loss_train, loss_val
        _description_
    """
    n_classes = dataset_train.n_classes
    model = MLPCalW(in_channels=n_classes, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_train = np.zeros(n_epochs)
    loader_train = DataLoader(dataset_train, batch_size=batch_size)
    # save validation losses if needed
    if dataset_val is not None:
        loader_val = DataLoader(dataset_val, batch_size=len(dataset_val))
        loss_val = np.zeros(n_epochs)

    for n in range(n_epochs):
        # train
        loss_epoch = []
        model.train()
        for i, (p_probs, y_labels_train) in enumerate(loader_train):
            p_probs = p_probs.float()
            weights_l = model(p_probs)
            l = loss(p_probs, weights_l, y_labels_train)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_epoch.append(l.item())
        
        loss_n = sum(loss_epoch)/(len(loader_train))
        loss_train[n] = loss_n
        if print_losses:
            print(f'Epoch: {n} train loss: {loss_n}')
        # validation 
        if dataset_val is not None:
            val_loss_epoch = []
            model.eval()
            for i, (p_probs_val, y_labels_val) in enumerate(loader_val):
                p_probs_val = p_probs_val.float()
                weights_val = model(p_probs_val)
                loss_val_n = loss(p_probs_val, weights_val, y_labels_val)
                val_loss_epoch.append(loss_val_n.item())
            loss_val_n =sum(val_loss_epoch)/len(val_loss_epoch)
            if print_losses:
                print(f"Validation loss: {loss_val_n}")
            loss_val[n] = loss_val_n

    if dataset_val is not None:
        return model, loss_train, loss_val
    else:
        return model, loss_train

        



if __name__ == "__main__":
    # define variables
    N_ENSEMBLE = 10
    N_EPOCHS = 1000
    LR = 0.0001

    N = 1000
    M = 10
    K = 10
    U = 0.01

    # nn training
    loss = SKCELoss(use_median_bw=True, dist_fct=tv_distance_tensor)
    dataset = DirichletDataset(n_samples=1000, n_members=10, n_classes=10, u_s=0.01)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optim_weights = get_optim_lambda_mlp(dataset_train=dataset, loss=loss, n_epochs=1)
    #print(optim_weights)



