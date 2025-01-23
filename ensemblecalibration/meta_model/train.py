from typing import Any, Optional
import copy
import torch
import numpy as np
import torch.utils
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data

from torch import nn
from ensemblecalibration.losses import CalibrationLoss, BaseCombinerLoss


class EarlyStopping:
    """
    Class for early stopping of training by evaluating the decreases in validation loss.
    """

    def __init__(
        self, patience: int = 10, min_delta: float = 0.0, verbose: bool = False
    ):
        """
        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait until early stopping, by default 10.
        min_delta : float, optional
            Minimum improvement in loss to reset patience, by default 0.0
        verbose : bool, optional
            Whether to print debug information, by default False
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_epoch = 0
        self.best_model = None
        self.early_stopping_flag = False
        self.verbose = verbose

    def __call__(self, val_losses, model):
        """
        Checks if early stopping criterion is met.

        Parameters
        ----------
        val_losses : list of float
            A list of validation losses for each completed epoch.
        model : nn.Module
            The current model, so we can store its best state.
        """
        current_loss = val_losses[-1]
        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_epoch = len(val_losses)
            self.best_model = copy.deepcopy(model)
        elif current_loss < self.best_loss - self.min_delta:
            self.counter = 0
            self.best_loss = current_loss
            self.best_epoch = len(val_losses)
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopping_flag = True
                if self.verbose:
                    print(f"Early stopping at epoch {len(val_losses)}")
                    print(
                        f"Best epoch: {self.best_epoch}, Best val loss: {self.best_loss}"
                    )


def train_model(
    model: nn.Module,
    dataset_train,
    loss_fn,
    dataset_val=None,
    train_mode="joint",
    device="cpu",
    n_epochs=50,
    lr=1e-3,
    batch_size=128,
    shuffle=True,
    optimizer_class=optim.Adam,
    # For alternating mode
    subepochs_comb=1,
    subepochs_cal=1,
    # Early stopping params
    early_stopping=False,
    patience=10,
    min_delta=0.0,
    verbose=False
):
    """
    Trains a model in one of three modes:
      1) 'joint' -> end-to-end training of comb_model + cal_model
      2) 'avg_then_calibrate' -> fix combination to ensemble average, only train calibrator
      3) 'alternating' -> freeze comb_model while training cal_model, and vice versa in blocks

    Integrates Early Stopping if early_stopping=True.

    Parameters
    ----------
    model : nn.Module
        Typically a CredalSetCalibrator or similar, which has:
          - comb_model for alpha net
          - cal_model for calibration
        For 'avg_then_calibrate', the comb_model is effectively unused/frozen.
    dataset_train : Dataset
        Returns (p_preds, y, x) or (x, y). Must at least provide p_preds if doing joint or alt.
    loss_fn : nn.Module
        The loss function (Brier/log-loss) that can handle p_bar or (p_preds, weights_l).
    dataset_val : Dataset, optional
        For validation, by default None.
    train_mode : str
        'joint', 'avg_then_calibrate', or 'alternating'.
    device : str, optional
        'cpu' or 'cuda'.
    n_epochs : int, optional
        Number of epochs/cycles, by default 50.
    lr : float, optional
        Learning rate, by default 1e-3.
    batch_size : int, optional
        By default 128.
    shuffle : bool, optional
        Shuffle training data, by default True.
    optimizer_class : optional
        Optimizer for parameters, by default Adam.
    subepochs_comb : int, optional
        For 'alternating' mode, how many sub-epochs to train comb_model per block, by default 1.
    subepochs_cal : int, optional
        For 'alternating' mode, how many sub-epochs to train cal_model per block, by default 1.
    early_stopping : bool, optional
        If True, we apply early stopping on the validation loss.
    patience : int, optional
        Number of epochs/cycles with no improvement in val loss before stopping.
    min_delta : float, optional
        Minimum improvement in val loss to reset patience, by default 0.0.
    verbose : bool, optional
        Print progress, by default False.

    Returns
    -------
    (model, train_losses, val_losses) : (nn.Module, list[float], list[float] or None)
        The trained model (potentially rolled back to best state if early stopping),
        plus the list of training losses, and list of validation losses if val set is provided.
    """
    # 1) Create data loaders
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
    if dataset_val is not None:
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
        val_losses = []
    else:
        loader_val = None
        val_losses = None

    # 2) Prepare for different train modes
    train_losses = []

    if train_mode == "avg_then_calibrate":
        # We only train cal_model => fix comb_model or skip it
        optimizer = optimizer_class(model.cal_model.parameters(), lr=lr)

    elif train_mode == "joint":
        # We do standard end-to-end => all parameters
        optimizer = optimizer_class(model.parameters(), lr=lr)

    elif train_mode == "alternating":
        # We'll have separate optimizers for comb and cal
        comb_params = model.comb_model.parameters()
        cal_params = model.cal_model.parameters()
        optimizer_comb = optimizer_class(comb_params, lr=lr)
        optimizer_cal = optimizer_class(cal_params, lr=lr)
    else:
        raise ValueError("Invalid train_mode, must be one of 'joint', 'avg_then_calibrate', 'alternating'.")

    # 3) Initialize EarlyStopping if needed
    stopper = None
    if early_stopping and (loader_val is not None):
        stopper = EarlyStopping(patience=patience, min_delta=min_delta, verbose=verbose)

    ############################################
    # Helper function: run one epoch of "joint"
    ############################################
    def train_one_epoch_joint():
        model.train()
        epoch_loss = 0.0
        for batch in loader_train:
            if len(batch) == 3:
                p_preds_batch, y_batch, x_batch = batch
            else:
                p_preds_batch, y_batch = batch
                x_batch = None

            p_preds_batch = p_preds_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch, p_preds_batch)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                p_cal, p_bar, weights_l = outputs
            else:
                p_cal = outputs
                p_bar = None
                weights_l = None

            # measure loss on p_cal or p_bar
            loss = loss_fn(y=y_batch, p_bar=p_cal)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader_train)
        return epoch_loss

    ############################################
    # Helper: run one epoch of "avg_then_calibrate"
    ############################################
    def train_one_epoch_avg():
        model.cal_model.train()
        epoch_loss = 0.0

        for batch in loader_train:
            if len(batch) == 3:
                p_preds_batch, y_batch, x_batch = batch
            else:
                p_preds_batch, y_batch = batch
                x_batch = None

            p_preds_batch = p_preds_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            # average across ensemble dimension => shape (batch_size, C)
            p_bar = p_preds_batch.mean(dim=1)
            # pass through calibrator
            p_cal = model.cal_model(p_bar)
            # compute loss on p_cal
            loss = loss_fn(y=y_batch, p_bar=p_cal)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader_train)
        return epoch_loss

    ############################################
    # Helper: run one subepoch for comb_model or cal_model in "alternating"
    ############################################
    def train_comb_subepoch():
        for param in model.cal_model.parameters():
            param.requires_grad = False
        for param in model.comb_model.parameters():
            param.requires_grad = True

        sub_loss = 0.0
        model.train()
        for batch in loader_train:
            if len(batch) == 3:
                p_preds_batch, y_batch, x_batch = batch
            else:
                p_preds_batch, y_batch = batch
                x_batch = None

            p_preds_batch = p_preds_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device)

            optimizer_comb.zero_grad()
            outputs = model(x_batch, p_preds_batch)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                p_cal, p_bar, weights_l = outputs
            else:
                p_cal = None
                p_bar = outputs
                weights_l = None

            loss = loss_fn(y=y_batch, p_bar=p_bar)
            loss.backward()
            optimizer_comb.step()
            sub_loss += loss.item()

        return sub_loss / len(loader_train)

    def train_cal_subepoch():
        for param in model.comb_model.parameters():
            param.requires_grad = False
        for param in model.cal_model.parameters():
            param.requires_grad = True

        sub_loss = 0.0
        model.train()
        for batch in loader_train:
            if len(batch) == 3:
                p_preds_batch, y_batch, x_batch = batch
            else:
                p_preds_batch, y_batch = batch
                x_batch = None

            p_preds_batch = p_preds_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device)

            optimizer_cal.zero_grad()
            outputs = model(x_batch, p_preds_batch)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                p_cal, p_bar, weights_l = outputs
            else:
                p_cal = outputs
                p_bar = None
                weights_l = None

            loss = loss_fn(y=y_batch, p_bar=p_cal)
            loss.backward()
            optimizer_cal.step()
            sub_loss += loss.item()

        return sub_loss / len(loader_train)

    ############################################
    # MAIN TRAINING LOOP
    ############################################
    val_losses = [] if loader_val else None

    if train_mode == "joint":
        # For each epoch, train "joint"
        for epoch in range(n_epochs):
            epoch_loss = train_one_epoch_joint()
            train_losses.append(epoch_loss)

            val_loss = None
            if loader_val is not None:
                val_loss = evaluate_model(model, loader_val, loss_fn, device=device)
                val_losses.append(val_loss)

                # Check early stopping
                if stopper is not None:
                    stopper(val_losses, model)
                    if stopper.early_stopping_flag:
                        model = stopper.best_model
                        if verbose:
                            print(f"Stopped early at epoch {epoch+1}")
                        break

            # Print progress
            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{n_epochs}: train={epoch_loss:.4f}, val={val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{n_epochs}: train={epoch_loss:.4f}")

    elif train_mode == "avg_then_calibrate":
        # For each epoch, only calibrate on the average
        for epoch in range(n_epochs):
            epoch_loss = train_one_epoch_avg()
            train_losses.append(epoch_loss)

            val_loss = None
            if loader_val is not None:
                val_loss = evaluate_model_avg(model, loader_val, loss_fn, device=device)
                val_losses.append(val_loss)

                # Check early stopping
                if stopper is not None:
                    stopper(val_losses, model)
                    if stopper.early_stopping_flag:
                        model = stopper.best_model
                        if verbose:
                            print(f"Stopped early at epoch {epoch+1}")
                        break

            # Print progress
            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{n_epochs}: train={epoch_loss:.4f}, val={val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{n_epochs}: train={epoch_loss:.4f}")

    elif train_mode == "alternating":
        # We'll treat each iteration as 1 'cycle'
        for cycle in range(n_epochs):
            # Step A: train comb_model subepochs_comb times
            for _ in range(subepochs_comb):
                comb_loss = train_comb_subepoch()

            # Step B: train cal_model subepochs_cal times
            for _ in range(subepochs_cal):
                cal_loss = train_cal_subepoch()

            # We'll record the last subepoch's loss as the "train loss" for this cycle
            train_losses.append(cal_loss)

            val_loss = None
            if loader_val is not None:
                val_loss = evaluate_model(model, loader_val, loss_fn, device=device)
                val_losses.append(val_loss)

                # Check early stopping
                if stopper is not None:
                    stopper(val_losses, model)
                    if stopper.early_stopping_flag:
                        model = stopper.best_model
                        if verbose:
                            print(f"Stopped early at cycle {cycle+1}")
                        break

            if verbose and (cycle == 0 or (cycle + 1) % 5 == 0):
                if val_loss is not None:
                    print(f"Cycle {cycle+1}/{n_epochs}: comb_loss={comb_loss:.4f}, cal_loss={cal_loss:.4f}, val={val_loss:.4f}")
                else:
                    print(f"Cycle {cycle+1}/{n_epochs}: comb_loss={comb_loss:.4f}, cal_loss={cal_loss:.4f}")

    return model, train_losses, val_losses

########################################
# Helper: Evaluate model in "joint" or "alternating" scenario
########################################
def evaluate_model(model, loader, loss_fn, device="cpu"):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                p_preds_batch, y_batch, x_batch = batch
            else:
                p_preds_batch, y_batch = batch
                x_batch = None

            p_preds_batch = p_preds_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device)

            outputs = model(x_batch, p_preds_batch)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                p_cal, p_bar, weights_l = outputs
            else:
                p_cal = None
                p_bar = outputs
                weights_l = None

            loss_val = loss_fn(y=y_batch, p_bar=p_bar)
            val_loss += loss_val.item()
    return val_loss / len(loader)


########################################
# Helper: Evaluate model in "avg_then_calibrate" scenario
########################################
def evaluate_model_avg(model, loader, loss_fn, device="cpu"):
    """
    We skip alpha net and average ensemble predictions,
    then pass through model.cal_model.
    """
    model.cal_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                p_preds_batch, y_batch, x_batch = batch
            else:
                p_preds_batch, y_batch = batch
                x_batch = None

            p_preds_batch = p_preds_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device)

            # average
            p_bar = p_preds_batch.mean(dim=1)  # shape (batch_size, n_classes)
            # calibrate
            p_cal = model.cal_model(p_bar)
            loss_val = loss_fn(y=y_batch, p_bar=p_cal)
            val_loss += loss_val.item()

    return val_loss / len(loader)


# def train_model(
#     model: nn.Module,
#     dataset_train: torch.utils.data.Dataset,
#     loss_fn: BaseCombinerLoss,
#     dataset_val: torch.utils.data.Dataset = None,
#     train_mode: str = "joint",
#     device: str = "cpu",
#     n_epochs: int = 50,
#     lr: float = 1e-3,
#     batch_size: int = 128,
#     shuffle: bool = True,
#     optimizer_class=optim.Adam,
#     lr_scheduler_class=None,
#     early_stopping=True,
#     patience=10,
#     verbose=False,
#     alternating_optimization=False,
#     subepochs_comb=1,
#     subepochs_cal=1,
#     **scheduler_kwargs,
# ):
#     """
#     Main training function. If alternating_optimization=False, do standard joint training.
#     Otherwise, do alternating freeze-thaw between comb_model and cal_model.

#     Parameters
#     ----------
#     model : nn.Module
#        Typically a CredalSetCalibrator or similar, which has:
#           - comb_model for alpha net
#           - cal_model for calibration
#         For 'avg_then_calibrate', the comb_model is effectively unused/frozen.
#     dataset_train : Dataset
#         Training dataset. Yields (p_preds, y, x) or (x, y).
#     loss_fn : _BaseCombinerLoss
#         E.g. GeneralizedBrierLoss or GeneralizedLogLoss.
#     dataset_val : Dataset, optional
#         Validation dataset, by default None.
#     train_mdoe : str, optional
#         "joint", "alternating", or "average_then_calibrate", by default "joint".
#         determines which training mode to use.
#     device : str, optional
#         CPU or GPU, by default "cpu".
#     n_epochs : int, optional
#         Number of epochs (or cycles if alternating), by default 50.
#     lr : float, optional
#         Learning rate, by default 1e-3.
#     batch_size : int, optional
#         Batch size, by default 128.
#     shuffle : bool, optional
#         Shuffle training data, by default True.
#     optimizer_class : optional
#         Which optimizer to use, by default Adam.
#     lr_scheduler_class : optional
#         Learning rate scheduler, by default None.
#     early_stopping : bool, optional
#         If True, use early stopping, by default True.
#     patience : int, optional
#         Patience for early stopping, by default 10.
#     verbose : bool, optional
#         Print progress, by default False.
#     alternating_optimization : bool, optional
#         Whether to freeze-thaw between comb_model and cal_model, by default False.
#     subepochs_comb : int, optional
#         If alternating, number of subepochs to train comb_model each cycle, by default 1.
#     subepochs_cal : int, optional
#         If alternating, number of subepochs to train cal_model each cycle, by default 1.
#     **scheduler_kwargs
#         Additional arguments for the LR scheduler.

#     Returns
#     -------
#     model : nn.Module
#         Trained model (potentially rolled back to best state if early stopping).
#     train_losses : list of float
#         Training losses per epoch (or cycle).
#     val_losses : list of float or None
#         Validation losses if dataset_val is provided, otherwise None.
#     """
#     model.to(device)

#     # Build loader
#     loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
#     if dataset_val is not None:
#         loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
#         val_losses = []
#     else:
#         loader_val = None
#         val_losses = None

#     if train_mode == "avg_then_calibrate":
#         # We only train cal_model => fix comb_model or skip it
#         # The combination is simply the average of ensemble predictions
#         optimizer = optimizer_class(model.cal_model.parameters(), lr=lr)

#     elif train_mode == "joint":
#         # We do standard end-to-end => all parameters
#         optimizer = optimizer_class(model.parameters(), lr=lr)

#     elif train_mode == "alternating":
#         # We'll have separate optimizers for comb and cal
#         comb_params = model.comb_model.parameters()
#         cal_params = model.cal_model.parameters()
#         optimizer_comb = optimizer_class(comb_params, lr=lr)
#         optimizer_cal = optimizer_class(cal_params, lr=lr)
#     else:
#         raise ValueError(
#             "Invalid train_mode, must be one of 'joint', 'avg_then_calibrate', 'alternating'."
#         )

#     # LR scheduler?
#     lr_scheduler = None
#     if lr_scheduler_class is not None:
#         lr_scheduler = lr_scheduler_class(optimizer, **scheduler_kwargs)

#     stopper = EarlyStopping(patience=patience, verbose=verbose)
#     train_losses = []

#     if not alternating_optimization:
#         # Standard joint training
#         if verbose:
#             print("==> Using standard (joint) training...")

#         for epoch in range(n_epochs):
#             train_loss = train_one_epoch(
#                 model=model,
#                 data_loader=loader_train,
#                 loss_fn=loss_fn,
#                 optimizer=optimizer,
#                 device=device,
#             )
#             train_losses.append(train_loss)

#             val_loss = None
#             if loader_val is not None:
#                 val_loss = evaluate(model, loader_val, loss_fn, device=device)
#                 val_losses.append(val_loss)

#                 if lr_scheduler is not None:
#                     # If using e.g. ReduceLROnPlateau:
#                     lr_scheduler.step(val_loss)

#                 if early_stopping:
#                     stopper(val_losses, model)
#                     if stopper.early_stopping_flag:
#                         if verbose:
#                             print(f"Early stopping at epoch {epoch+1}")
#                         model = stopper.best_model
#                         break

#             if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
#                 if val_loss is not None:
#                     print(
#                         f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
#                     )
#                 else:
#                     print(f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}")

#     else:
#         # Alternating optimization
#         if verbose:
#             print(
#                 "==> Using alternating optimization between comb_model and cal_model..."
#             )
#         num_cycles = n_epochs  # treat each epoch as a "cycle" if desired
#         for cycle_idx in range(num_cycles):
#             # Each cycle => freeze calibrator, train comb, freeze comb, train calibrator
#             train_loss = train_alternating(
#                 model=model,
#                 data_loader=loader_train,
#                 loss_fn=loss_fn,
#                 optimizer_comb=optimizer_comb,
#                 optimizer_cal=optimizer_cal,
#                 device=device,
#                 subepochs_comb=subepochs_comb,
#                 subepochs_cal=subepochs_cal,
#             )
#             train_losses.append(train_loss)

#             # Validation
#             val_loss = None
#             if loader_val is not None:
#                 val_loss = evaluate(model, loader_val, loss_fn, device=device)
#                 val_losses.append(val_loss)

#                 # If you want an LR scheduler step after each cycle
#                 if lr_scheduler is not None:
#                     lr_scheduler.step(val_loss)

#                 if early_stopping:
#                     stopper(val_losses, model)
#                     if stopper.early_stopping_flag:
#                         if verbose:
#                             print(
#                                 f"Early stopping at cycle {cycle_idx+1} / epoch approx."
#                             )
#                         model = stopper.best_model
#                         break

#             if verbose and (cycle_idx == 0 or (cycle_idx + 1) % 5 == 0):
#                 if val_loss is not None:
#                     print(
#                         f"Cycle {cycle_idx+1}/{num_cycles}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
#                     )
#                 else:
#                     print(
#                         f"Cycle {cycle_idx+1}/{num_cycles}: train_loss={train_loss:.4f}"
#                     )

#     return model, train_losses, val_losses


# def train_alternating(
#     model: nn.Module,
#     data_loader: DataLoader,
#     loss_fn: BaseCombinerLoss,
#     optimizer_comb: optim.Optimizer,
#     optimizer_cal: optim.Optimizer,
#     device: str = "cpu",
#     subepochs_comb: int = 1,
#     subepochs_cal: int = 1,
#     verbose: bool = False,
# ) -> float:
#     """
#     Runs one 'full' cycle of alternating optimization:
#       1) Freeze calibrator, train comb_model for subepochs_comb epochs,
#       2) Freeze comb_model, train calibrator for subepochs_cal epochs.

#     Returns average training loss of the final subepoch (for logging).
#     """

#     # 0) Distinguish comb_model vs cal_model
#     comb_params = list(model.comb_model.parameters())
#     cal_params = list(model.cal_model.parameters())

#     # Step A: Freeze calibrator, train comb_model
#     for param in cal_params:
#         param.requires_grad = False
#     for param in comb_params:
#         param.requires_grad = True

#     if verbose:
#         print(
#             "comb_model requires_grad params:",
#             sum(p.requires_grad for p in model.comb_model.parameters()),
#         )
#         print(
#             "cal_model requires_grad params:",
#             sum(p.requires_grad for p in model.cal_model.parameters()),
#         )

#     loss_value = 0.0
#     for _ in range(subepochs_comb):
#         loss_value = 0.0
#         model.train()
#         for batch in data_loader:
#             if len(batch) == 3:
#                 p_preds_batch, y_batch, x_batch = batch
#                 p_preds_batch = p_preds_batch.to(device, dtype=torch.float32)
#             else:
#                 x_batch, y_batch = batch
#                 p_preds_batch = None

#             x_batch = x_batch.to(device, dtype=torch.float32)
#             y_batch = y_batch.to(device)

#             optimizer_comb.zero_grad()

#             # forward pass
#             outputs = model(x_batch, p_preds_batch)
#             if isinstance(outputs, tuple) and len(outputs) == 3:
#                 p_cal, p_bar, weights_l = outputs
#             else:
#                 p_cal = None
#                 p_bar = outputs
#                 weights_l = None

#             # compute loss (with flexible approach)
#             # If you want the combination to happen inside loss, pass p_preds + weights_l
#             # But presumably we have p_bar from model:
#             this_loss = loss_fn(y=y_batch, p_bar=p_bar)
#             this_loss.backward()
#             optimizer_comb.step()
#             loss_value += this_loss.item()

#         loss_value /= len(data_loader)

#     # Step B: Freeze comb_model, train calibrator
#     for param in comb_params:
#         param.requires_grad = False
#     for param in cal_params:
#         param.requires_grad = True

#     for _ in range(subepochs_cal):
#         loss_value = 0.0
#         model.train()
#         for batch in data_loader:
#             if len(batch) == 3:
#                 p_preds_batch, y_batch, x_batch = batch
#                 p_preds_batch = p_preds_batch.to(device, dtype=torch.float32)
#             else:
#                 x_batch, y_batch = batch
#                 p_preds_batch = None

#             x_batch = x_batch.to(device, dtype=torch.float32)
#             y_batch = y_batch.to(device)

#             optimizer_cal.zero_grad()

#             outputs = model(x_batch, p_preds_batch)
#             if isinstance(outputs, tuple) and len(outputs) == 3:
#                 p_cal, p_bar, weights_l = outputs
#             else:
#                 p_cal = outputs
#                 p_bar = None
#                 weights_l = None

#             this_loss = loss_fn(y=y_batch, p_bar=p_cal)
#             this_loss.backward()
#             optimizer_cal.step()
#             loss_value += this_loss.item()

#         loss_value /= len(data_loader)

#     # Return final subepoch's average loss
#     return loss_value


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: BaseCombinerLoss,  # one of the GeneralizedBrierLoss or GeneralizedLogLoss
    optimizer: optim.Optimizer,
    device: str = "cpu",
) -> float:
    """
    Runs one epoch of training over data_loader in a standard (joint) fashion:
      - Model can output either 'weights' or 'p_bar' internally,
        but the loss function is flexible enough to handle both scenarios.

    Parameters
    ----------
    model : nn.Module
        The model (could be a CredalSetCalibrator) that typically returns
        (p_cal, p_bar, weights_l) or something similar in forward(...).
    data_loader : DataLoader
        Training data loader. Each batch might be (p_preds, y, x) or (x, y).
    loss_fn : _BaseCombinerLoss
        The adapted loss function that can handle p_bar or (p_preds, weights_l).
    optimizer : torch.optim.Optimizer
        Optimizer for the model parameters.
    device : str, optional
        Device to run on, by default "cpu".

    Returns
    -------
    float
        Average training loss over the epoch.
    """
    model.train()
    epoch_loss = 0.0

    for batch in data_loader:
        # Commonly, batch = (p_preds, y, x)
        if len(batch) == 3:
            p_preds_batch, y_batch, x_batch = batch
            p_preds_batch = p_preds_batch.to(device, dtype=torch.float32)
        else:
            # fallback if dataset only yields (x, y)
            x_batch, y_batch = batch
            p_preds_batch = None

        x_batch = x_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Forward pass:
        #   if CredalSetCalibrator, something like: p_cal, p_bar, weights_l = model(x_batch, p_preds_batch)
        outputs = model(
            x_batch, p_preds_batch
        )  # adapt depending on your model signature

        # Suppose your model returns (p_cal, p_bar, weights_l)
        if isinstance(outputs, tuple) and len(outputs) == 3:
            p_cal, p_bar, weights_l = outputs
        else:
            # If your model only returns p_bar or something
            p_cal = outputs
            p_bar = None
            weights_l = None

        # The adapted loss function can handle either:
        #   (p_bar=..., y=...) or
        #   (p_preds=..., weights_l=..., y=...)
        # So if your model already computed p_bar, we do:
        loss_value = loss_fn(y=y_batch, p_bar=p_cal)
        #   Alternatively, if you want the loss to handle combo internally:
        #   loss_value = loss_fn(y=y_batch, p_preds=p_preds_batch, weights_l=weights_l)

        loss_value.backward()
        optimizer.step()

        epoch_loss += loss_value.item()

    epoch_loss /= len(data_loader)
    return epoch_loss


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: BaseCombinerLoss,
    device: str = "cpu",
) -> float:
    """
    Evaluates model on data_loader, returning average loss.

    Parameters
    ----------
    model : nn.Module
        The model to be evaluated (e.g., CredalSetCalibrator).
    data_loader : DataLoader
        Validation or test data loader.
    loss_fn : _BaseCombinerLoss
        Loss function (Brier or log loss) that can handle p_bar or p_preds, weights_l.
    device : str, optional
        Device, by default "cpu".

    Returns
    -------
    float
        Average loss over the dataset.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                p_preds_batch, y_batch, x_batch = batch
                p_preds_batch = p_preds_batch.to(device, dtype=torch.float32)
            else:
                x_batch, y_batch = batch
                p_preds_batch = None

            x_batch = x_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device)

            outputs = model(x_batch, p_preds_batch)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                p_cal, p_bar, weights_l = outputs
            else:
                p_cal = None
                p_bar = outputs
                weights_l = None

            # Evaluate with the flexible loss
            loss_val = loss_fn(y=y_batch, p_bar=p_bar)
            val_loss += loss_val.item()

    val_loss /= len(data_loader)
    return val_loss


def get_optim_lambda_mlp(
    dataset_train: torch.utils.data.Dataset,
    dataset_val: torch.utils.data.Dataset,
    dataset_test: torch.utils.data.Dataset,
    model: torch.nn.Module,
    loss: CalibrationLoss,
    n_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 128,
    optim=torch.optim.Adam,
    patience: int = 15,
    device: str = "cpu",
    verbose: bool = False,
    stratified: bool = False,
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
    model : torch.nn.Module
        model used for training
    loss : _type_
        loss of the form loss(p_probs, weights, y_labels) indicating the calibration error of the
    n_epochs : int, optional
        number of epochs the model is trained, by default 100
    lr : float, optional
        lewarning rate, by default 0.001
    batch_size : int, optional
        batch size, by default 128
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
    #"""
    # assert that daataset has x_train attribute
    assert hasattr(dataset_train, "x_train"), "dataset needs to have x_train attribute"

    model, loss_train, loss_val = train_model(
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
        stratified=stratified,
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
    with torch.no_grad():
        x_inst = x_inst.to(device)
        optim_weights = model(x_inst).detach().cpu()

    return optim_weights, loss_train, loss_val


# def train_model(
#     model,
#     dataset_train: torch.utils.data.Dataset,
#     loss,
#     dataset_val: Optional[torch.utils.data.Dataset] = None,
#     device: str = "cpu",
#     n_epochs: int = 100,
#     lr: float = 0.001,
#     batch_size: int = 128,
#     optim=torch.optim.Adam,
#     shuffle: bool = True,
#     lr_scheduler=None,
#     early_stopping: bool = True,
#     patience: int = 10,
#     verbose: bool = False,
#     stratified: bool = False,
#     **kwargs,
# ):
#     """trains the MLP model to predict the optimal weight matrix for the given ensemble model
#     such that the calibration error of the convex combination is minimized.

#     Parameters
#     ----------
#     dataset_train : torch.utils.data.Dataset
#         dataset containing probabilistic predictions of ensembnle members used for training
#     loss :
#         loss taking a tuple of the probabilistic predictions, the weights of the convex combination
#         and the labels as input
#     dataset_val : torch.utils.data.Dataset, optional
#         dataset containing probabilistic predictions of ensemble members used for validation,
#     n_epochs : int
#         number of training epochs
#     lr : float
#         learning rate
#     batch_size : int, optional
#         _description_, by default 128
#     print_losses : bool, optional
#         whether to print train and validation loss at every epoch, by default True
#     every_n_epoch : int, optional
#         print losses every n epochs, by default 1
#     optim : torch.optim.Optimizer, optional
#         optimizer used for training, by default torch.optim.Adam
#     shuffle : bool, optional
#         whether to shuffle the training data, by default True
#     lr_scheduler : of type torch.optim.lr_scheduler, optional
#         learning rate scheduler, by default None (i.e., no scheduler is used)
#     patience : int, optional
#         number of epochs without improvement after which the training is stopped, by default 10
#     print_losses : bool, optional
#         whether to print the training loss at every epoch, by default True
#     kwargs : dict
#         additional keyword arguments passed to the learning rate scheduler

#     Returns
#     -------
#     model, loss_train, loss_val
#         _description_
#     """
#     model.to(device)
#     optimizer = optim(model.parameters(), lr=lr, weight_decay=1e-5)
#     if lr_scheduler is not None:
#         lr_scheduler = lr_scheduler(
#             optimizer, **kwargs
#         )  # kwargs can be e.g. step_size or gamma
#     loss_train = []
#     if stratified:
#         sampler_train = StratifiedSampler(
#             dataset_train, batch_size=batch_size, num_classes=dataset_train.n_classes
#         )
#         loader_train = DataLoader(
#             dataset_train, batch_size=batch_size, sampler=sampler_train
#         )
#     else:
#         loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=shuffle)
#     if dataset_val is not None:
#         loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
#         loss_val = []
#     else:
#         loader_val = None

#     stopper = EarlyStoping(patience=patience, verbose=verbose)
#     if verbose:
#         print(f"Training on {device}")
#     for n in range(n_epochs):
#         # train
#         model.train()
#         loss_epoch_train, loss_epoch_val = train_epoch(
#             model,
#             loss,
#             loader_train=loader_train,
#             loader_val=loader_val,
#             optimizer=optimizer,
#             lr_scheduler=lr_scheduler,
#             device=device,
#         )
#         loss_train.append(loss_epoch_train)
#         if dataset_val is not None:
#             loss_val.append(loss_epoch_val)
#         # print loss every 20th epoch
#         if verbose:
#             if n % 20 == 0:
#                 print(f"Epoch {n}: Train Loss: {loss_epoch_train}")
#                 if dataset_val is not None:
#                     print(f"Epoch {n}: Validation Loss: {loss_epoch_val}")

#         # check using early stopping if training should be stopped
#         # check every n epochs
#         if early_stopping:
#             assert (
#                 dataset_val is not None
#             ), "Validation dataset is required for early stopping"
#             stopper(loss_val, copy.deepcopy(model))
#             if stopper.early_stopping_flag:
#                 model = stopper.best_model
#                 break

#     return model, loss_train, loss_val


# def train_epoch(
#     model,
#     loss,
#     loader_train,
#     optimizer,
#     loader_val=None,
#     lr_scheduler=None,
#     device: str = "cpu",
#     clip_gradients: bool = False,
# ):
#     """
#     training loop for one epoch for the given model, loss function, data loaders and optimizers.
#     Optionally, a learning rate scheduler can be used.

#     Parameters
#     ----------
#     model : torch.nn.Module
#         model to be trained
#     loss :
#         loss function used for training
#     loader_train: torch.utils.data.DataLoader
#         data loader for training data
#     optimizer : torch.optim.Optimizer
#         optimizer used for training
#     lr_scheduler : torch.optim.lr_scheduler, optional
#         learning rate scheduler, by default None
#     device : str, optional
#         device on which the calculations are performed, by default 'cpu'
#     """

#     # torch.autograd.set_detect_anomaly(True)
#     loss_epoch_train = 0
#     model.train()
#     # iterate over train dataloader
#     for p_probs, y_labels_train, x_train in loader_train:
#         p_probs = p_probs.float().to(device)
#         x_train = x_train.float().to(device)
#         y_labels_train = y_labels_train.to(device)

#         optimizer.zero_grad()
#         # predict weights as the output of the model on the given instances
#         # print max and min of x_train
#         weights_l = model(x_train)
#         # calculate loss
#         loss_train = loss(p_probs, weights_l, y_labels_train)
#         # set gradients to zero
#         loss_train.backward()
#         if clip_gradients:
#             # print("Clipping gradients")
#             utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         loss_epoch_train += loss_train.item()
#     if loader_val is not None:
#         loss_epoch_val = 0
#         model.eval()
#         for p_probs, y_labels_val, x_val in loader_val:
#             p_probs = p_probs.float().to(device)
#             x_val = x_val.float().to(device)
#             y_labels_val = y_labels_val.to(device)
#             weights_l = model(x_val)
#             loss_val = loss(p_probs, weights_l, y_labels_val)
#             loss_epoch_val += loss_val.item()
#         loss_epoch_val /= len(loader_val)

#         if lr_scheduler is not None:
#             # make a step in the learning rate scheduler:
#             lr_scheduler.step(loss_epoch_val / len(loader_val))

#     loss_epoch_train /= len(loader_train)

#     return loss_epoch_train, loss_epoch_val if loader_val is not None else None


# class EarlyStoping:
#     """
#     class for early stopping of training by evaluating the decreases of the training loss.
#     """

#     def __init__(
#         self, patience: int = 10, min_delta: float = 0.0, verbose: bool = False
#     ) -> None:
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
#         self.best_loss = None
#         self.best_epoch = 0
#         self.best_model = None
#         self.early_stopping_flag = False
#         self.verbose = verbose

#     def __call__(self, train_loss, model, *args: Any, **kwds: Any) -> Any:

#         if self.best_loss is None:
#             self.best_loss = train_loss[-1]
#             self.best_epoch = len(train_loss)
#             self.best_model = model

#         elif train_loss[-1] < self.best_loss - self.min_delta:
#             self.counter = 0
#             self.best_loss = train_loss[-1]
#             self.best_epoch = len(train_loss)
#             self.best_model = model

#         else:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stopping_flag = True
#                 if self.verbose:
#                     print(f"Early stopping at epoch {len(train_loss)}")
#                     print(f"Best epoch: {self.best_epoch}, Best loss: {self.best_loss}")
#                 return self.best_model
