import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint



class MLPCalW(nn.Module):
    """
    class of the MLP model used to learn the optimal convex combination for the respective
    credal set. 
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        hidden_layers: int = 1,
        use_relu: bool = True,
    ):
        """multi layer perceptron for training the optimal weights of a convex combination of
          predictors in order to receive a calibrated model.
        Parameters
        ----------
        in_channels : int
            number of classes for which point predictions are given in the input tensor
        hidden_dim : int
            hidden dimension of the MLP inner layer
        hidden_layers : int, optional
            number of hidden layers, by default 1
        relu : bool, optional
            whether to use ReLU activation function, by default True
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # layers
        layers = []
        if hidden_layers == 0:
            layers.append(nn.Linear(self.in_channels, out_channels))
        else:
            layers.append(nn.Linear(self.in_channels, self.hidden_dim))
            for i in range(hidden_layers):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                # apply ReLU activation
                if use_relu:
                    layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(self.hidden_dim, out_channels))

        self.layers = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in: torch.Tensor):
        out = self.layers(x_in)
        # reshape output to matrix of weights of two dimensions (N, M)
        out = out.view(-1, out.shape[1])
        # apply softmax to get weights in (0,1) and summing up to 1
        out = self.softmax(out)
        return out


# same model as above, but with pytorch-lightning
class MLPCalWLightning(pl.LightningModule):
    def __init__(
        self,
        loss_fct: nn.Module,
        lr: float,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        hidden_layers: int = 1,
        use_relu: bool = True,
        use_scheduler: bool = False,

    ):
        """meta learning model for learning the optimal weights of a convex combination of
        predictors in order to receive a calibrated model.

        Parameters
        ----------
        loss_fct : nn.Module
            loss function to be used for training
        lr : float
            learning rate
        in_channels : int
            dimension of instance space
        out_channels : int
            number of predictors in the ensemble
        hidden_dim : int
            
        hidden_layers : int, optional
            _description_, by default 1
        use_relu : bool, optional
            _description_, by default True
        lr_scheduler : _type_, optional
            _description_, by default None
        """
        super().__init__()
        self.model = MLPCalW(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            use_relu=use_relu,
        )
        self.loss_fct = loss_fct
        self.lr = lr
        self.use_scheduler = use_scheduler
        # log hyperparameters
        self.save_hyperparameters()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # batch consists of probabilities in credal set, labels and features
        p_probs, y, x = batch
        pred_lambda = self.model(x.float())
        # calculate loss
        loss = self.loss_fct(p_probs, pred_lambda, y)
        # log
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):

        p_probs, y, x = batch
        pred_lambda = self.model(x.float())
        # calculate loss
        loss = self.loss_fct(p_probs, pred_lambda, y)
        # log
        self.log('val_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler}
 
