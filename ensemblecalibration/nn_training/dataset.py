from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset

from ensemblecalibration.nn_training.experiments import binary_experiment_nn

class MLPDataset(Dataset):
    """
    Dataset containing probabilistic predictions of an ensemble of synthetic classifiers,
    
    """

    def __init__(self, x_train: np.ndarray, P: np.ndarray, y: np.ndarray):
        """
        Parameters
        ----------
        x_train : np.ndarray
            array of shape (N, F) containing the training data (instances)
        P : np.ndarray
            tensor of shape (N, M, K) containing probabilistic predictions for each instance and each predictor
        y : np.ndarray
            array of shape (N,) containing labels
        """
        super().__init__()
        self.p_probs = P
        self.y_true = y
        self.n_classes = P.shape[2]
        self.n_ens = P.shape[1]
        self.n_features = x_train.shape[1]
        self.x_train = x_train

    def __len__(self):
        return len(self.p_probs)
    
    def __getitem__(self, index):
        
        return self.p_probs[index], self.y_true[index], self.x_train[index]
    

class MLPDataModule(pl.LightningDataModule):

    def __init__(self, p_probs: torch.Tensor, y_labels: torch.Tensor, ratio_train: float = 0.8,
                 batch_size: int = 32) -> None:
        super().__init__()
        self.p_probs = p_probs
        self.y_labels = y_labels
        self.n_classes = p_probs.shape[2]
        self.batch_size = batch_size

        # split into train and test
        self.train_p_probs = self.p_probs[:int(ratio_train*len(self.p_probs))]
        self.train_y_labels = self.y_labels[:int(ratio_train*len(self.p_probs))]
        self.data_train = [self.train_p_probs, self.train_y_labels]

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)
    

class DirichletDataset(Dataset):
    """Dataset containing probabilistic predictions of an ensemble of synthetic classifiers,
    as well as a (randomly) generated convex combination, from whose predictions the labels are samples
    using a categorical distribution.   

    """

    def __init__(self, n_samples: int = 10000, n_members: int = 10, n_classes: int = 10, 
                  u_s: float = 0.01, experiment=binary_experiment_nn, lambda_fct: str = "linear") -> None:
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
        self.n_samples = n_samples
        self.n_members = n_members
        self.n_classes = n_classes
        self.u_s = u_s
        # generate "optimal" weights which need to be estimated by the model
        p_probs, y_true, weights_l = experiment(n_instances=self.n_samples, n_classes=self.n_classes,
                                                fct=lambda_fct, uct=self.u_s)
        self.p_probs = p_probs
        self.y_true = y_true
        self.optim_weights = weights_l

    def __len__(self):
        return len(self.p_probs)

    def __getitem__(self, index):
        return self.p_probs[index], self.y_true[index]
