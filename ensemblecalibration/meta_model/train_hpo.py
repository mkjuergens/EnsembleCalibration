import torch
import wandb

from ensemblecalibration.data.experiments import get_experiment
from ensemblecalibration.meta_model import MLPCalW
from ensemblecalibration.meta_model.losses import LpLoss, MMDLoss, SKCELoss
from ensemblecalibration.config import create_config_binary_classification
from ensemblecalibration.meta_model.train import train_one_epoch, build_optimizer
from ensemblecalibration.utils.helpers import calculate_pbar
# import lr schedulers
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(config=None):

    config_default = create_config_binary_classification(exp_name="gp")
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader, dataset = get_experiment(
            config=config_default, h0=True, batch_size=config.batch_size
        )
        model = MLPCalW(
            in_channels=1,
            out_channels=1,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            use_relu=False
        )
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate)
        loss = MMDLoss(bw=config.bw, lambda_bce=config.lambda_bce)
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        for epoch in range(config.epochs):
            loss_avg = train_one_epoch(model, loss=loss,
                                               loader_train=loader, optimizer=optimizer, 
                                               lr_scheduler=lr_scheduler)
            wandb.log({"loss": loss_avg, "epoch": epoch})
            # log learning rate
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
        
        # get real weights and real loss
        weights_l_real = dataset.weights_l
        p_preds = dataset.p_probs
        real_loss = loss(p_preds, weights_l_real, dataset.y_true)
        wandb.log({"real_loss": real_loss, "bw": config.bw, "lambda_bce": config.lambda_bce})

        # get loss of model
        weights_l_pred = model(dataset.x_train)
        model_loss = loss(p_preds, weights_l_pred, dataset.y_true)
        wandb.log({"model_loss": model_loss})
        #plot weights predicted by model
        wandb.log({"weights_l_pred": weights_l_pred[:, 0].detach().numpy()})
        #plot weights of real data
        wandb.log({"weights_l_real": weights_l_real[:, 0].detach().numpy()})



if __name__ == "__main__":
    # hyperparameter sweep
    
    train()