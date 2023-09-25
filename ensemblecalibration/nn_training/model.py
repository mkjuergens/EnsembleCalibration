import torch
import numpy as np
from torch import nn

from torch.utils.data import DataLoader, Dataset

from ensemblecalibration.calibration.experiments import (
    get_ens_alpha,
    experiment_h0_feature_dependency,
)


class MLPCalW(nn.Module):
    """
    class of the MLP model used to
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
            layers.append(nn.Linear(self.hidden_dim, out_channels))

        self.layers = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        self.use_relu = use_relu

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in: torch.Tensor):
        out = self.layers(x_in)
        if self.use_relu:
            out = self.relu(out)
        # reshape output to matrix of weights of two dimensions (N, M)
        out = out.view(-1, out.shape[1])
        # apply softmax to get weights in (0,1) and summing up to 1
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    x_train = np.random.random(size=(1000, 1))
    print(x_train.shape)
    model = MLPCalW(in_channels=1, hidden_dim=10, out_channels=5,
                    hidden_layers = 0, use_relu=False)
    x_train = torch.from_numpy(x_train).float()
    out_weights = model(x_train)
    print(out_weights)
