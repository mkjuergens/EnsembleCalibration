from typing import Any, Optional

import copy
import torch
import numpy as np
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from ensemblecalibration.meta_model.mlp_model import MLPCalW
from ensemblecalibration.meta_model.losses import CalibrationLoss


def get_optim_lambda_mlp(
    dataset_train: torch.utils.data.Dataset,
    dataset_val: torch.utils.data.Dataset,
    dataset_test: torch.utils.data.Dataset,
    loss: CalibrationLoss,
    n_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 128,
    hidden_dim: int = 8,
    hidden_layers: int = 0,
    optim=torch.optim.Adam,
    patience: int = 15,
    device: str = "cpu",
    verbose: bool = False,
):
    """function for finding the weight vector which results in the lowest calibration error,
    using an MLP model. The model is trained to predict the optimal weight vector for the given

    Parameters
    ----------
    dataset_train : torch.utils.data.Dataset
        dataset which contains instances, as well as probabilistic predictions of ensemble members
        and labels
    dataset_val : torch.utils.data.Dataset
        dataset which contains instances, as well as probabilistic predictions of ensemble members
    dataset_test : torch.utils.data.Dataset
        dataset which contains instances, as well as probabilistic predictions of ensemble members
    loss : _type_
        loss of the form loss(p_probs, weights, y_labels) indicating the calibration error of the
    n_epochs : int, optional
        number of epochs the model is trained, by default 100
    lr : float, optional
        lewarning rate, by default 0.001
    batch_size : int, optional
        batch size, by default 128
    hidden_dim : int, optional
        number of hidden units in the hidden layers, by default 8
    hidden_layers : int, optional
        number of hidden layers, by default 0
    optim : torch.optim.Optimizer, optional
        optimizer used for training, by default torch.optim.Adam
    shuffle : bool, optional
        whether to shuffle the training data, by default True
    patience : int, optional
        number of epochs without improvement after which the training is stopped, by default 15
    device : str, optional
        device on which the calculations are performed, by default "cpu"
    verbose : bool, optional
        whether to print the training loss at every epoch, by default False


    Returns
    -------
    optim_weights
        resulting optimal weight vector
    loss_train
        training loss
    loss_val
        validation loss
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
        loss=loss,
        dataset_val=dataset_val,
        device=device,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        optim=optim,
        patience=patience,
        verbose=verbose,

    )
    # use features as input to model instead of probs
    # new: use test data
    x_inst = (
        torch.from_numpy(dataset_test.x_train).float()
        if isinstance(dataset_test.x_train, np.ndarray)
        else dataset_test.x_train
    )
    model.eval()
    # device
    x_inst = x_inst.to(device)
    optim_weights = model(x_inst)
    optim_weights = optim_weights.detach().cpu()
    return optim_weights, loss_train, loss_val


def train_one_epoch(
    model,
    loss,
    loader_train,
    optimizer,
    loader_val=None,
    lr_scheduler=None,
    device: str = "cpu",
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
    lr_scheduler : torch.optim.lr_scheduler, optional
        learning rate scheduler, by default None
    device : str, optional
        device on which the calculations are performed, by default 'cpu'
    """

    #torch.autograd.set_detect_anomaly(True)
    loss_epoch_train = 0
    model.train()
    # iterate over train dataloader
    for p_probs, y_labels_train, x_train in loader_train:
        p_probs = p_probs.float().to(device)
        x_train = x_train.float().to(device)
        y_labels_train = y_labels_train.to(device)
        
        optimizer.zero_grad()
        # predict weights as the output of the model on the given instances
        # print max and min of x_train
        weights_l = model(x_train)
        # calculate loss
        loss_train = loss(p_probs, weights_l, y_labels_train)
        # set gradients to zero
        loss_train.backward()
        optimizer.step()
        loss_epoch_train += loss_train.item()
    if loader_val is not None:
        loss_epoch_val = 0
        model.eval()
        for p_probs, y_labels_val, x_val in loader_val:
            p_probs = p_probs.float().to(device)
            x_val = x_val.float().to(device)
            y_labels_val = y_labels_val.to(device)
            weights_l = model(x_val)
            loss_val = loss(p_probs, weights_l, y_labels_val)
            loss_epoch_val += loss_val.item()
        loss_epoch_val /= len(loader_val)

        if lr_scheduler is not None:
            # make a step in the learning rate scheduler:
            lr_scheduler.step(loss_epoch_val/len(loader_val))

    loss_epoch_train /= len(loader_train)

    return loss_epoch_train, loss_epoch_val if loader_val is not None else None

def build_optimizer(model, optimizer: str = "adam", lr: float = 0.001):
    """builds the optimizer for the given model

    Parameters
    ----------
    model : torch.nn.Module
        model for which the optimizer is built
    optim : str, optional
        optimizer used, by default "adam"
    lr : float, optional
        learning rate, by default 0.001

    Returns
    -------
    torch.optim.Optimizer
        optimizer
    """
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        raise ValueError("Optimizer not implemented")
    return optimizer


def train_mlp(
    model,
    dataset_train: torch.utils.data.Dataset,
    loss,
    dataset_val: Optional[torch.utils.data.Dataset] = None,
    device: str = "cpu",
    n_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 128,
    optim=torch.optim.Adam,
    shuffle: bool = True,
    lr_scheduler=None,
    early_stopping: bool = True,
    patience: int = 10,
    verbose: bool = False,
    **kwargs,
):
    """trains the MLP model to predict the optimal weight matrix for the given ensemble model
    such that the calibration error of the convex combination is minimized.

    Parameters
    ----------
    dataset_train : torch.utils.data.Dataset
        dataset containing probabilistic predictions of ensembnle members used for training
    loss : 
        loss taking a tuple of the probabilistic predictions, the weights of the convex combination
        and the labels as input
    dataset_val : torch.utils.data.Dataset, optional
        dataset containing probabilistic predictions of ensemble members used for validation,
    n_epochs : int
        number of training epochs
    lr : float
        learning rate
    batch_size : int, optional
        _description_, by default 128
    print_losses : bool, optional
        whether to print train and validation loss at every epoch, by default True
    every_n_epoch : int, optional
        print losses every n epochs, by default 1
    optim : torch.optim.Optimizer, optional
        optimizer used for training, by default torch.optim.Adam
    shuffle : bool, optional
        whether to shuffle the training data, by default True
    lr_scheduler : of type torch.optim.lr_scheduler, optional
        learning rate scheduler, by default None (i.e., no scheduler is used)
    patience : int, optional
        number of epochs without improvement after which the training is stopped, by default 10
    print_losses : bool, optional
        whether to print the training loss at every epoch, by default True
    kwargs : dict
        additional keyword arguments passed to the learning rate scheduler

    Returns
    -------
    model, loss_train, loss_val
        _description_
    """
    model.to(device)
    optimizer = optim(model.parameters(), lr=lr)
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(
            optimizer, **kwargs
        )  # kwargs can be e.g. step_size or gamma
    loss_train = []
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    if dataset_val is not None:
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        loss_val = []
    else:
        loader_val = None

    stopper = EarlyStoping(patience=patience, verbose=verbose)
    if verbose:
        print(f"Training on {device}")
    for n in range(n_epochs):
        # train
        model.train()
        loss_epoch_train, loss_epoch_val = (
            train_one_epoch(
                model,
                loss,
                loader_train=loader_train,
                loader_val=loader_val,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,

            )
        )
        loss_train.append(loss_epoch_train)
        if dataset_val is not None:
            loss_val.append(loss_epoch_val)
        # print loss every 20th epoch
        if verbose:
            if n % 20 == 0:
                print(f"Epoch {n}: Train Loss: {loss_epoch_train}")

        # check using early stopping if training should be stopped
        # check every n epochs
        if early_stopping:
            assert dataset_val is not None, "Validation dataset is required for early stopping"
            stopper(loss_val, copy.deepcopy(model)) 
            if stopper.early_stopping_flag:
                model = stopper.best_model
                break

    return model, loss_train, loss_val


# class EarlyStoping:
#     """
#     class for early stopping of training by evaluating the decreases of the training loss.
#     """

#     def __init__(self, patience: int = 10, min_delta: float = 0) -> None:
#         """
#         Parameters
#         ----------
#         patience : int, optional
#             number of epochs to wait unitl early stopping
#         min_delta : float, optional
#             , by default 0
#         """
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.stop = False

#     def __call__(self, train_loss, *args: Any, **kwds: Any) -> Any:

#         if len(train_loss) < 2:
#             return
#         elif (train_loss[-2] - train_loss[-1]) < self.min_delta:
#             self.counter += 1
#         elif (train_loss[-2] - train_loss[-1]) > self.min_delta:
#             self.counter = 0
#         if self.counter >= self.patience:
#             self.stop = True

class EarlyStoping:
    """
    class for early stopping of training by evaluating the decreases of the training loss.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = False) -> None:
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
        self.best_loss = None
        self.best_epoch = 0
        self.best_model = None
        self.early_stopping_flag = False
        self.verbose = verbose

    def __call__(self, train_loss, model, *args: Any, **kwds: Any) -> Any:

        if self.best_loss is None:
            self.best_loss = train_loss[-1]
            self.best_epoch = len(train_loss)
            self.best_model = model

        elif train_loss[-1] < self.best_loss - self.min_delta:
            self.counter = 0
            self.best_loss = train_loss[-1]
            self.best_epoch = len(train_loss)
            self.best_model = model
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopping_flag = True
                if self.verbose:
                    print(f"Early stopping at epoch {len(train_loss)}")
                    print(f"Best epoch: {self.best_epoch}, Best loss: {self.best_loss}")
                return self.best_model
