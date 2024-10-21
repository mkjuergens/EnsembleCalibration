from collections import defaultdict
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class MLPDataset(Dataset):
    """
    Dataset containing probabilistic predictions of an ensemble of synthetic classifiers,

    """

    def __init__(
        self,
        x_train: np.ndarray,
        P: np.ndarray,
        y: np.ndarray,
        weights_l: np.ndarray = None,
    ):
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
        weights_l : np.ndarray, optional
            array of shape (N, M) containing the weights of the convex combination
            of the probabilistic predictions, by default None
        """
        super().__init__()
        self.p_probs = P
        self.y_true = y
        self.n_classes = P.shape[2]
        self.n_ens = P.shape[1]
        self.n_features = x_train.shape[1]
        self.x_train = x_train
        self.weights_l = weights_l

    def __len__(self):
        return len(self.p_probs)

    def __getitem__(self, index):

        return self.p_probs[index], self.y_true[index], self.x_train[index]


class MLPDataModule(pl.LightningDataModule):

    def __init__(
        self,
        x_inst,
        p_probs: torch.Tensor,
        y_labels: torch.Tensor,
        ratio_train: float = 0.8,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.x_inst = x_inst
        self.p_probs = p_probs
        self.y_labels = y_labels
        self.n_classes = p_probs.shape[2]
        self.batch_size = batch_size
        self.ratio_train = ratio_train

    def setup(self, stage: str):
        # train data
        self.x_train = self.x_inst[: int(self.ratio_train * len(self.x_inst))]
        self.train_p_probs = self.p_probs[: int(self.ratio_train * len(self.p_probs))]
        self.train_y_labels = self.y_labels[: int(self.ratio_train * len(self.p_probs))]
        self.dataset_train = MLPDataset(
            x_train=self.x_train, P=self.train_p_probs, y=self.train_y_labels
        )

        # validation data
        self.x_val = self.x_inst[int(self.ratio_train * len(self.x_inst)) :]
        self.val_p_probs = self.p_probs[int(self.ratio_train * len(self.p_probs)) :]
        self.val_y_labels = self.y_labels[int(self.ratio_train * len(self.p_probs)) :]
        self.dataset_val = MLPDataset(
            x_train=self.x_val, P=self.val_p_probs, y=self.val_y_labels
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)

# stratified sampler for Lp calibration error and CIFAR10

class StratifiedSampler:
    def __init__(self, dataset: MLPDataset, batch_size: int, num_classes: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.class_indices = defaultdict(list)

        # Group dataset indices by class
        for idx, (_, label, _) in enumerate(dataset):
            self.class_indices[label].append(idx)

    def __iter__(self):
        batch = []
        all_classes = list(self.class_indices.keys())
        
        while len(batch) < len(self.dataset):
            current_batch = []

            # Ensure at least two examples from each class in the batch
            for class_idx in all_classes:
                if len(self.class_indices[class_idx]) >= 2:
                    samples = np.random.choice(self.class_indices[class_idx], 2, replace=False)
                    current_batch.extend(samples)

            # Shuffle remaining examples and fill the batch
            remaining_indices = [idx for sublist in self.class_indices.values() for idx in sublist]
            np.random.shuffle(remaining_indices)

            for idx in remaining_indices:
                if len(current_batch) < self.batch_size:
                    current_batch.append(idx)
                else:
                    break

            np.random.shuffle(current_batch)
            batch.extend(current_batch)

        return iter(batch)

    def __len__(self):
        return len(self.dataset) // self.batch_size