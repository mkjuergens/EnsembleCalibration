import torch
import numpy as np
from torch import nn

from torch.utils.data import DataLoader, Dataset

from ensemblecalibration.calibration.experiments import get_ens_alpha, experiment_h0_feature_dependency


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
            number of classes for which point predictions are given in the input tensor    
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
            nn.Linear(self.hidden_dim, 1)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in: torch.Tensor):
        out = self.layers(x_in)
        # reshape output to matrix of weights of two dimensions (N, M) 
        out = out.view(-1, out.shape[1])
        out = self.softmax(out)
        return out
    
class MLPDataset(Dataset):
    """Dataset for training the MLP 
    """

    def __init__(self, P: np.ndarray, y: np.ndarray):
        super().__init__()
        self.p_probs = P
        self.y_true = y
        self.n_classes = P.shape[2]

    def __len__(self):
        return len(self.p_probs)
    
    def __getitem__(self, index):
        return self.p_probs[index], self.y_true[index]
    

class DirichletDataset(Dataset):
    """Dataset containing probabilistic predictions of an ensemble of synthetic classifiers,
    as well as a (randomly) generated convex combination, from whose predictions the labels are samples
    using a categorical distribution.   

    """

    def __init__(self, n_samples: int = 10000, n_members: int = 10, n_classes: int = 10, 
                  u_s: float = 0.01) -> None:
        """

        Parameters
        ----------
        n_samples : int, optional
            number of instances contained in the dataset, by default 10000
        n_members : int, optional
            number of ensemble members, by default 10
        n_classes : int, optional
            number of classes to predict, by default 10
        u_s : float, optional
            parameter of the distribution controling the spread within the K-1 simplex, by default 0.01
        """
        super().__init__()
        self.n_members = n_members
        self.n_classes = n_classes
        self.u_s = u_s
        # generate "optimal" weights which need to be estimated by the model
        p_probs, y_true, weights_l = experiment_h0_feature_dependency(n_samples,n_members, n_classes, u_s,
                                                                      return_optim_weight=True)
        self.p_probs = p_probs
        self.y_true = y_true
        self.optim_weights = weights_l

    def __len__(self):
        return len(self.p_probs)

    def __getitem__(self, index):
        return self.p_probs[index], self.y_true[index]
        
if __name__ == "__main__":
    dataset = DirichletDataset(n_samples=1000, n_members=10, n_classes=3)
    loader = DataLoader(dataset, batch_size=100)
    p_probs, y = next(iter(loader))
    print(p_probs.shape)
    model = MLPCalW(in_channels=3, hidden_dim=32)
    out = model(p_probs.float())
    print(out.shape)

    