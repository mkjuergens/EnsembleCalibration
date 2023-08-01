from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader

from ensemblecalibration.nn_training.distances import tv_distance_tensor
from ensemblecalibration.nn_training.model import MLPCalW
from ensemblecalibration.nn_training.losses import SKCELoss
from ensemblecalibration.calibration.experiments import (
    experiment_h0_feature_dependency,
    experiment_h0_feature_dependency,
)

def get_optim_lambda_mlp(dataset_train: torch.utils.data.Dataset, loss,
                     n_epochs: int = 100, lr: float = 0.001, batch_size: int = 128,
                     optim=torch.optim.Adam, shuffle: bool = True):
    """function for finding the weight vector which results in the lowest calibration error,
    using a MLP model. The model is trained to predict the optimal weight vector for the given

    Parameters
    ----------
    dataset_train : torch.utils.data.Dataset
        dataset which contains instances, as well as probabilistic predictions of ensemble members
        and labels
    loss : _type_
        loss of the form loss(p_probs, weights, y_labels) indicating the calibration error of the
    n_epochs : int, optional
        number of epochs the model is trained, by default 100
    lr : float, optional
        lewarning rate, by default 0.001

    Returns
    -------
    optim_weights
        resulting optimal weight vector
    """
    model = MLPCalW(in_channels=dataset_train.n_features, out_channels=dataset_train.n_ens,
                    hidden_dim=32)
    model, loss_train = train_mlp(model, dataset_train=dataset_train, loss=loss, n_epochs=n_epochs,
                                  lr=lr, print_losses=False, batch_size=batch_size, optim=optim)
    # use features as input to model instead of probs
    x_inst = dataset_train.x_train
    optim_weights = model(x_inst)
    optim_weights = optim_weights.detach().numpy()
    return optim_weights


def train_one_epoch(model, loss, loader_train, optimizer, loader_val: Optional[DataLoader] = None):
     
    loss_epoch_train = 0
    # iterate over train dataloader
    for (p_probs, y_labels_train, x_train) in loader_train:
        p_probs, x_train = p_probs.float(), x_train.float()
        # predict weights as the output of the model on the given instances
        weights_l = model(x_train)
        # calculate loss
        loss_train = loss(p_probs, weights_l, y_labels_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        loss_epoch_train += loss_train.item()
    
    loss_epoch_train /= len(loader_train)
    loss_epoch_val = None
    if loader_val is not None:
        loss_epoch_val = 0
        model.eval()
        for (p_probs, y_labels_val, x_val) in loader_val:
            p_probs, x_val = p_probs.float(), x_val.float()
            weights_l = model(x_val)
            loss_val = loss(p_probs, weights_l, y_labels_val)
            loss_epoch_val += loss_val.item()

        loss_epoch_val /= len(loader_val)
    
    return model, loss_epoch_train, loss_epoch_val


def train_mlp(model,
    dataset_train: torch.utils.data.Dataset, loss, dataset_val: Optional[torch.utils.data.Dataset] = None, 
    n_epochs: int = 100, lr: float = 0.001, batch_size: int = 128, lower_bound: float = 0.0,
    print_losses: bool = True, every_n_epoch: int = 1, optim=torch.optim.Adam, shuffle: bool = True):
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
    lower_bound : float, optional
        lower bound for the loss function, by default 0.0. Can be used e.g. to set "true" value 
        of miscalibration as a lower threshold.
    print_losses : bool, optional
        whether to print train and validation loss at every epoch, by default True
    every_n_epoch : int, optional
        print losses every n epochs, by default 1

    Returns
    -------
    model, loss_train, loss_val
        _description_
    """
    optimizer = optim(model.parameters(), lr=lr)
    loss_train = np.zeros(n_epochs)
    loss_val = np.zeros(n_epochs)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    loader_val = None
    # save validation losses if needed
    if dataset_val is not None:
        loader_val = DataLoader(dataset_val, batch_size=len(dataset_val))
        loss_val = np.zeros(n_epochs)

    for n in range(n_epochs):
        # train
        model.train()

        model, loss_epoch_train, loss_epoch_val = train_one_epoch(model, loss, loader_train, optimizer,
                                                                  loader_val )
        loss_train[n] = loss_epoch_train
        if loss_epoch_val is not None:
            loss_val[n] = loss_epoch_val
        if print_losses:
            if (n +1) % every_n_epoch == 0:
                print(f'Epoch: {n} train loss: {loss_epoch_train} val loss: {loss_epoch_val}')

    return model, loss_train, loss_val
        
        
"""
        for i, (p_probs, y_labels_train, x_train) in enumerate(loader_train):
            p_probs = p_probs.float()
            x_train = x_train.float()
            # model takes instances as input
            weights_l = model(x_train)
            l = loss(p_probs, weights_l, y_labels_train)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_epoch.append(l.item())
        
        loss_n = sum(loss_epoch)/(len(loader_train))
        if loss_n < lower_bound:
            loss_n = lower_bound
        loss_train[n] = loss_n
        if print_losses:
            if (n +1) % every_n_epoch == 0:
                print(f'Epoch: {n} train loss: {loss_n}')
        # validation 
        if dataset_val is not None:
            val_loss_epoch = []
            model.eval()
            for i, (p_probs_val, y_labels_val, x_val) in enumerate(loader_val):
                p_probs_val = p_probs_val.float()
                weights_val = model(x_val)
                x_val = x_val.float()
                loss_val_n = loss(p_probs_val, weights_val, y_labels_val)
                val_loss_epoch.append(loss_val_n.item())
            loss_val_n =sum(val_loss_epoch)/len(val_loss_epoch)
            if print_losses:
                if (n + 1) % every_n_epoch == 0:
                    print(f"Validation loss: {loss_val_n}")
            loss_val[n] = loss_val_n

    if dataset_val is not None:
        return model, loss_train, loss_val
    else:
        return model, loss_train

    
"""

        

