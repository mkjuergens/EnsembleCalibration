import torch
import pytorch_lightning as pl

import torch.nn as nn





class MLPCalW(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, n_layers: int, hidden_dim: int) -> None:
        """multi layer precetron for training the optimal weights of a convex combination of predistors in order 
            to receive a calibrated model.
        Parameters
        ----------
        in_channels : int
            _description_
        out_channels : int
            _description_
        n_layers : int
            _description_
        hidden_dim : int
            _description_
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # layers 3 layers to be able t
        self.layers = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_channels),
            nn.Tanh()

        )


    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out




