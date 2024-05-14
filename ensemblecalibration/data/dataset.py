import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


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
            tensor of shape (N, M, K) containing probabilistic predictions
            for each instance and each predictor
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

    def __init__(self, x_inst,  p_probs: torch.Tensor, y_labels: torch.Tensor, 
                 ratio_train: float = 0.8, batch_size: int = 32) -> None:
        super().__init__()
        self.x_inst = x_inst
        self.p_probs = p_probs
        self.y_labels = y_labels
        self.n_classes = p_probs.shape[2]
        self.batch_size = batch_size
        self.ratio_train = ratio_train
    
    def setup(self, stage: str):
        # train data
        self.x_train = self.x_inst[:int(self.ratio_train*len(self.x_inst))]
        self.train_p_probs = self.p_probs[:int(self.ratio_train*len(self.p_probs))]
        self.train_y_labels = self.y_labels[:int(self.ratio_train*len(self.p_probs))]
        self.dataset_train = MLPDataset(x_train=self.x_train, P=self.train_p_probs, y=self.train_y_labels)

        # validation data
        self.x_val = self.x_inst[int(self.ratio_train*len(self.x_inst)):]
        self.val_p_probs = self.p_probs[int(self.ratio_train*len(self.p_probs)):]
        self.val_y_labels = self.y_labels[int(self.ratio_train*len(self.p_probs)):]
        self.dataset_val = MLPDataset(x_train=self.x_val, P=self.val_p_probs, y=self.val_y_labels)


    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)