import torch
import wandb

from ensemblecalibration.data.experiments_cal_test import get_experiment
from ensemblecalibration.meta_model import MLPCalW
from ensemblecalibration.losses.cal_losses import LpLoss, MMDLoss, SKCELoss, BrierLoss
from ensemblecalibration.config import create_config_recal
from ensemblecalibration.meta_model.train import train_epoch_1

# import lr schedulers
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_loss(loss, bw, lambda_bce):
    if loss == "MMD":
        loss = MMDLoss(bw=bw, lambda_bce=lambda_bce)
    elif loss == "LP":
        loss = LpLoss(p=2, lambda_bce=lambda_bce)
    elif loss == "SKCE":
        loss = SKCELoss(bw=bw, lambda_bce=lambda_bce)
    elif loss == "Brier":
        loss = BrierLoss(lambda_bce=lambda_bce)
    return loss


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


def train(config=None):

    config_default = create_config_recal(exp_name="gp")
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        loss = get_loss(config.loss, config.bw, config.lambda_bce)

        _, loader, dataset = get_experiment(
            config=config_default[config.loss], h0=True, batch_size=config.batch_size
        )
        model = MLPCalW(
            in_channels=1,
            out_channels=1,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            use_relu=True,
        )
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate)
        # loss = MMDLoss(bw=config.bw, lambda_bce=config.lambda_bce)
        # lr_scheduler = ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.1, patience=50, verbose=True
        # )
        for epoch in range(config.epochs):
            loss_avg = train_epoch_1(
                model,
                loss=loss,
                loader_train=loader,
                optimizer=optimizer,
            )
            wandb.log({"loss": loss_avg, "epoch": epoch})
            # log learning rate
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})

        # get real weights and real loss
        weights_l_real = dataset.weights_l
        p_preds = dataset.p_probs
        real_loss = loss(p_preds, weights_l_real, dataset.y_true)
        wandb.log(
            {"real_loss": real_loss, "bw": config.bw, "lambda_bce": config.lambda_bce}
        )

        # get loss of model
        weights_l_pred = model(dataset.x_train)
        model_loss = loss(p_preds, weights_l_pred, dataset.y_true)
        wandb.log({"model_loss": model_loss})
        # plot weights predicted by model
        wandb.log({"weights_l_pred": weights_l_pred[:, 0].detach().numpy()})
        # plot weights of real data
        wandb.log({"weights_l_real": weights_l_real[:, 0].detach().numpy()})


if __name__ == "__main__":
    # hyperparameter sweep

    train()
