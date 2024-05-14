from typing import Any, Optional

import torch
import numpy as np
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from ensemblecalibration.meta_model.mlp_model import MLPCalW
from ensemblecalibration.meta_model.losses import CalibrationLossBinary



def get_optim_lambda_mlp(
    dataset_train: torch.utils.data.Dataset,
    loss: CalibrationLossBinary,
    dataset_val: Optional[torch.utils.data.Dataset] = None,
    n_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 128,
    hidden_dim: int = 8,
    hidden_layers: int = 0,
    optim=torch.optim.Adam,
    shuffle: bool = True,
    patience: int = 15,
):
    """function for finding the weight vector which results in the lowest calibration error,
    using an MLP model. The model is trained to predict the optimal weight vector for the given

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
    model = MLPCalW(
        in_channels=dataset_train.n_features,
        out_channels=dataset_train.n_ens,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    )
    # assert that daataset has x_train attribute
    assert hasattr(dataset_train, "x_train"), "dataset needs to have x_train attribute"

    model, loss_train, loss_val = train_mlp(
        model,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        loss=loss,
        n_epochs=n_epochs,
        lr=lr,
        print_losses=False,
        every_n_epoch=50,
        batch_size=batch_size,
        optim=optim,
        shuffle=shuffle,
        patience=patience,
    )
    # use features as input to model instead of probs
    x_inst = torch.from_numpy(dataset_train.x_train).float()
    model.eval()
    optim_weights = model(x_inst)
    optim_weights = optim_weights.detach().numpy()
    return optim_weights


def train_one_epoch(
    model,
    loss,
    loader_train,
    optimizer,
    loader_val: Optional[DataLoader] = None,
    lr_scheduler=None,
    best_loss_val: Optional[float] = None,
    save_best_model: bool = True,
    best_model: Optional[dict] = None,
):
    """
    training loop for one epoch for the given model, loss function, data loaders and optimizers.
    Optionally, a learning rate scheduler can be used.

    Parameters
    ----------
    model : torch.nn.Module
        model to be trained
    loss :
        loss function used for training
    loader_train: torch.utils.data.DataLoader
        data loader for training data
    optimizer : torch.optim.Optimizer
        optimizer used for training
    loader_val: torch.utils.data.DataLoader
        data loader for validation data, by default None
    lr_scheduler : of type torch.optim.lr_scheduler, optional
        learning rate scheduler, by default None (i.e., no scheduler is used)
    best_loss_val: float, Optional
        best validation loss so far, by default None (i.e. no validation loss is used for model 
        saving)
    save_best_model: bool, optional
        whether to save the best model, by default True
    """

    # check if validation data is provided if best model should be saved
    if save_best_model:
        assert (
            loader_val is not None
        ), "loader_val needs to be provided if save_best_model is True"

    loss_epoch_train = 0
    # iterate over train dataloader
    for p_probs, y_labels_train, x_train in loader_train:
        p_probs, x_train = p_probs.float(), x_train.float()
        optimizer.zero_grad()
        # predict weights as the output of the model on the given instances
        weights_l = model(x_train)
        # calculate loss
        loss_train = loss(p_probs, weights_l, y_labels_train)
        # set gradients to zero
        loss_train.backward()
        optimizer.step()
        loss_epoch_train += loss_train.item()
    if lr_scheduler is not None:
        # make a step in the learning rate scheduler:
        lr_scheduler.step(loss_epoch_train)

    loss_epoch_train /= len(loader_train)
    loss_epoch_val = None
    if loader_val is not None:
        loss_epoch_val = 0
        model.eval()
        for p_probs, y_labels_val, x_val in loader_val:
            p_probs, x_val = p_probs.float(), x_val.float()
            weights_l = model(x_val)
            loss_val = loss(p_probs, weights_l, y_labels_val)
            loss_epoch_val += loss_val.item()

        loss_epoch_val /= len(loader_val)
        if best_loss_val is None or loss_epoch_val < best_loss_val:
            best_loss_val = loss_epoch_val
            best_model = model.state_dict()

    return model, loss_epoch_train, loss_epoch_val, best_loss_val, best_model


def train_mlp(
    model,
    dataset_train: torch.utils.data.Dataset,
    loss,
    dataset_val: Optional[torch.utils.data.Dataset] = None,
    n_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 128,
    lower_bound: float = 0.0,
    print_losses: bool = True,
    every_n_epoch: int = 1,
    optim=torch.optim.Adam,
    shuffle: bool = True,
    save_best_model: bool = False,
    lr_scheduler=None,
    patience: int = 10,
    **kwargs,
):
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
    optim : torch.optim.Optimizer, optional
        optimizer used for training, by default torch.optim.Adam
    shuffle : bool, optional
        whether to shuffle the training data, by default True
    save_best_model: bool, optional
        whether to save the best model, by default True
    lr_scheduler : of type torch.optim.lr_scheduler, optional
        learning rate scheduler, by default None (i.e., no scheduler is used)
    patience : int, optional
        number of epochs without improvement after which the training is stopped, by default 10
    kwargs : dict
        additional keyword arguments passed to the learning rate scheduler

    Returns
    -------
    model, loss_train, loss_val
        _description_
    """
    optimizer = optim(model.parameters(), lr=lr)
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(
            optimizer, **kwargs
        )  # kwargs can be e.g. step_size or gamma
    loss_train = []
    loss_val = []
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    loader_val = None
    # save best validation loss here
    best_loss_val = float("inf")
    best_model = model.state_dict()
    # save validation losses if needed
    if dataset_val is not None:
        loader_val = DataLoader(dataset_val, batch_size=len(dataset_val))

    early_stopping = EarlyStoping(patience=patience)
    for n in range(n_epochs):
        # train
        model.train()
        model, loss_epoch_train, loss_epoch_val, best_loss_val, best_model = (
            train_one_epoch(
                model,
                loss,
                loader_train,
                optimizer,
                loader_val,
                lr_scheduler=lr_scheduler,
                best_loss_val=best_loss_val,
                save_best_model=save_best_model,
                best_model=best_model,
            )
        )
        loss_train.append(loss_epoch_train)
        if loss_epoch_val is not None:
            loss_val.append(loss_epoch_val)
        if print_losses:
            if (n + 1) % every_n_epoch == 0:
                print(
                    f'Epoch: {n} train loss: {loss_epoch_train} val loss: {loss_epoch_val} \n lr: {optimizer.param_groups[0]["lr"]}'
                )

        # check using early stopping if training should be stopped
        early_stopping(loss_train, loss_val)
        if early_stopping.stop:
            print(f"Early stopping at epoch {n}")
            break

    if save_best_model:
        model.load_state_dict(best_model)
    return model, loss_train, loss_val


class EarlyStoping:
    """
    class for early stopping of training by evaluating the decreases of the training loss.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0) -> None:
        """
        Parameters
        ----------
        patience : int, optional
            number of epochs to wait unitl early stopping
        min_delta : float, optional
            , by default 0
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.stop = False

    def __call__(self, train_loss, *args: Any, **kwds: Any) -> Any:

        if len(train_loss) < 2:
            return
        elif (train_loss[-2] - train_loss[-1]) < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
