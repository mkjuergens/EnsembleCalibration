from collections import defaultdict
from typing import Optional, Union
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class MLPDataset(Dataset):
    """
    A dataset that returns:
      - p_probs[i]: (M, K) ensemble predictions for instance i
      - y_true[i]:  scalar label
      - x_train[i]: input features or image, shape can be (F,) or (C,H,W)

    Optionally can store p_true or weights_l if needed.
    """

    def __init__(
        self,
        x_train: np.ndarray,
        P: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        p_true: Optional[np.ndarray] = None,
        weights_l: Optional[np.ndarray] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Parameters
        ----------
        x_train : np.ndarray
            shape (N, F) or (N, C, H, W) containing the inputs (images or features)
        P : np.ndarray
            shape (N, M, K) => ensemble predictions
        y : np.ndarray
            shape (N,) integer labels
        p_true : np.ndarray, optional
            shape (N, K) => ground-truth probabilities if available
        weights_l : np.ndarray, optional
            shape (N, M), optional weights
        device: Optional[Union[str, torch.device]], optional
            Device to which the tensors should be moved, by default None
        """
        super().__init__()

        N = x_train.shape[0]

        self.device = device if device else torch.device("cpu")

        self.p_probs = self._to_tensor(P, dtype=torch.float32)
        self.x_train = self._to_tensor(x_train, dtype=torch.float32)
        self.y_true = self._to_tensor(y, dtype=torch.long)

        # Basic attributes
        self.n_samples = N
        self.n_classes = self.p_probs.shape[2]
        self.n_ens = self.p_probs.shape[1]

        self.p_true: Optional[torch.Tensor] = None
        self.weights_l: Optional[torch.Tensor] = None

        if p_true is not None:
            # Convert to torch tensor if needed
            self.p_true = self._to_tensor(p_true, dtype=torch.float32)
        if weights_l is not None:
            # Convert to torch tensor if needed
            self.weights_l = self._to_tensor(weights_l, dtype=torch.float32)

    def _to_tensor(
        self, data: Union[np.ndarray, torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Converts data to a PyTorch tensor of the specified dtype and moves to device
        ."""
        if not isinstance(data, torch.Tensor):
            tensor_data = torch.from_numpy(data)
        else:
            tensor_data = data
        return tensor_data.to(dtype=dtype, device=self.device)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return a tuple: (p_probs[idx], y_true[idx], x_train[idx])
        return (self.p_probs[idx], self.y_true[idx], self.x_train[idx])


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
                    samples = np.random.choice(
                        self.class_indices[class_idx], 2, replace=False
                    )
                    current_batch.extend(samples)

            # Shuffle remaining examples and fill the batch
            remaining_indices = [
                idx for sublist in self.class_indices.values() for idx in sublist
            ]
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


if __name__ == "__main__":
    # Dummy data
    N_samples = 100
    N_features = 10
    N_ensemble = 5
    N_classes = 3

    # x_train: (N, F) - Features
    # x_train_img: (N, C, H, W) - Images (e.g., 1, 28, 28 for MNIST-like)
    X_feat = np.random.rand(N_samples, N_features).astype(np.float32)
    X_img = np.random.rand(N_samples, 1, 28, 28).astype(np.float32)

    # P: (N, M, K) - Ensemble predictions (softmax outputs)
    P_ensemble = np.random.rand(N_samples, N_ensemble, N_classes).astype(np.float32)
    P_ensemble = P_ensemble / P_ensemble.sum(axis=2, keepdims=True) # Normalize to make them probabilities

    # y: (N,) - True labels
    Y_labels = np.random.randint(0, N_classes, size=N_samples).astype(np.int64)

    # p_true: (N, K) - Optional true probabilities
    P_true_gt = np.random.rand(N_samples, N_classes).astype(np.float32)
    P_true_gt = P_true_gt / P_true_gt.sum(axis=1, keepdims=True)

    # weights_l: (N, M) - Optional optimal weights
    W_optimal = np.random.rand(N_samples, N_ensemble).astype(np.float32)
    W_optimal = W_optimal / W_optimal.sum(axis=1, keepdims=True) # Normalize to make them convex weights


    dataset_feat = MLPDataset(
        x_train=X_feat,
        P=P_ensemble,
        y=Y_labels,
        p_true=P_true_gt,
        weights_l=W_optimal,
        device='cpu' # or 'cuda' if available
    )

    print(f"Dataset length: {len(dataset_feat)}")
    sample_item = dataset_feat[0]