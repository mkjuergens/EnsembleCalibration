import torch
import numpy as np
from torch import nn


class MLPCalW(nn.Module):
    """
    class of the MLP model used to 
    """

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        """multi layer perceptron for training the optimal weights of a convex combination of
          predictors in order to receive a calibrated model.
        Parameters
        ----------
        in_channels : int
            number of ensemble members for which point predictions are given in the input tensor    
        hidden_dim : int
            hidden dimension of the 
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # layers 3 layers to be able t
        self.layers = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, x_in: torch.Tensor):
        out = self.layers(x_in)
        # reshape output to matrix of weights of two dimensions (N, M) 
        out = out.view(-1, out.shape[1])
        return out
    
if __name__ == "__main__":
    n_members = 10
    P = np.random.dirichlet([1]*n_members, size=(100,10))
    in_p = torch.from_numpy(P).float()
    model = MLPCalW(in_channels=n_members, out_channels=10, n_layers=3, hidden_dim=64)
    out = model(in_p)
    print(out.shape)
    