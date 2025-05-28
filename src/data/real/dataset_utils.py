import os
from collections import defaultdict
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


DATASETS = {
    "CIFAR10": {
        "dataset": torchvision.datasets.CIFAR10,
        "classes": 10,
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    },
    "CIFAR100": {
        "dataset": torchvision.datasets.CIFAR100,
        "classes": 100,
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    },
    "MNIST": {
        "dataset": torchvision.datasets.MNIST,
        "classes": 10,
        "transform": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    },
}


def load_dataset(name, batch_size=128, val_split=0.1):
    dataset_info = DATASETS[name]
    full_trainset = dataset_info["dataset"](
        root="./data", train=True, download=True, transform=dataset_info["transform"]
    )
    testset = dataset_info["dataset"](
        root="./data", train=False, download=True, transform=dataset_info["transform"]
    )

    # Split training set into training and validation sets
    train_size = int((1 - val_split) * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    # Create DataLoaders
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader, valloader, testloader, dataset_info["classes"]


def save_results(dataset_name, ensemble_size, predictions, instances, labels):
    np.save(f"{dataset_name}_ensemble_{ensemble_size}_predictions.npy", predictions)
    np.save(f"{dataset_name}_ensemble_{ensemble_size}_instances.npy", instances)
    np.save(f"{dataset_name}_ensemble_{ensemble_size}_labels.npy", labels)
    print(
        f"Saved predictions, instances, and labels for ensemble of size {ensemble_size} on {dataset_name}\n"
    )


# def load_results(
#     dataset_name, model_type, ensemble_type, ensemble_size=None, directory=None
# ):
#     """
#     Load the saved predictions, instances, and labels from the specified directory.

#     Parameters
#     ----------
#     dataset_name : str
#         The name of the dataset (e.g., 'CIFAR10').
#     model_type : str
#         The type of model architecture used (e.g., 'resnet' or 'vgg').
#     ensemble_type : str
#         The ensemble type ('deep_ensemble' or 'mc_dropout').
#     ensemble_size : int, optional
#         The number of models in the ensemble (only applicable for 'deep_ensemble').
#     directory : str, optional
#         The directory from which to load the results. Default is the current working directory.

#     Returns
#     -------
#     predictions : np.ndarray
#         The predictions made by the ensemble or MCDropout model.
#     instances : np.ndarray
#         The input instances used for the predictions.
#     labels : np.ndarray
#         The true labels of the instances.
#     """
#     # Use the current directory if no directory is provided
#     if directory is None:
#         directory = os.getcwd()

#     # Handle file names based on ensemble type
#     if ensemble_type == "deep_ensemble":
#         # For deep ensembles, use the ensemble_size in the file names
#         predictions_path = os.path.join(
#             directory,
#             f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_predictions.npy",
#         )
#         instances_path = os.path.join(
#             directory,
#             f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_instances.npy",
#         )
#         labels_path = os.path.join(
#             directory,
#             f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_labels.npy",
#         )
#     elif ensemble_type == "mc_dropout":
#         # For MCDropout, no need for ensemble_size in the file names
#         predictions_path = os.path.join(
#             directory,
#             f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_predictions.npy",
#         )
#         instances_path = os.path.join(
#             directory,
#             f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_instances.npy",
#         )
#         labels_path = os.path.join(
#             directory,
#             f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_labels.npy",
#         )
#     else:
#         raise ValueError(f"Unsupported ensemble type: {ensemble_type}")

#     # Load the results
#     predictions = np.load(predictions_path, allow_pickle=True)
#     instances = np.load(instances_path, allow_pickle=True)
#     labels = np.load(labels_path, allow_pickle=True)

#     return predictions, instances, labels


def load_results_real_data(
    dataset_name: str,
    model_type: str,
    ensemble_type: str,
    ensemble_size: int = None,
    directory: str = None,
    file_prefix: str = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the saved predictions, instances, and labels from the specified directory.

    For example, if file_prefix="val", we look for:
       CIFAR10_resnet_deep_ensemble_5_val_predictions.npy
       CIFAR10_resnet_deep_ensemble_5_val_instances.npy
       CIFAR10_resnet_deep_ensemble_5_val_labels.npy
    if ensemble_type='deep_ensemble' with ensemble_size=5.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset (e.g., 'CIFAR10' or 'CIFAR100').
    model_type : str
        The model architecture used (e.g., 'resnet', 'vgg', or custom).
    ensemble_type : str
        The ensemble type ('deep_ensemble' or 'mc_dropout').
    ensemble_size : int, optional
        Number of models in the ensemble (if 'deep_ensemble'), or # MC passes (if 'mc_dropout').
    directory : str, optional
        Directory from which to load the results. Defaults to current working directory if None.
    file_prefix : str, optional
        If provided, it is appended at the *end* of the standard file name. For example, "val" or "test".

    Returns
    -------
    predictions : np.ndarray
        (N, ensemble_size, n_classes).
    instances : np.ndarray
        For real data, shape (N, C, H, W).
    labels : np.ndarray
        (N,).
    """
    if directory is None:
        directory = os.getcwd()

    # We'll treat file_prefix as a *suffix* appended at the end of the base filename
    # e.g.  "..._5_val_predictions.npy"
    suffix_str = f"_{file_prefix}" if file_prefix else ""

    if ensemble_type == "deep_ensemble":
        if ensemble_size is None:
            raise ValueError("ensemble_size must be specified for 'deep_ensemble'.")
        preds_file = f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}{suffix_str}_predictions.npy"
        insts_file = f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}{suffix_str}_instances.npy"
        labels_file = f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}{suffix_str}_labels.npy"

    elif ensemble_type == "mc_dropout":
        if ensemble_size is None:
            raise ValueError("ensemble_size must be specified for 'mc_dropout'.")
        # e.g. CIFAR10_resnet_10_val_mc_dropout_predictions.npy if file_prefix="val"
        # But to keep a consistent pattern, let's do:
        #   <dataset>_<model_type>_<ensemble_size>_mc_dropout_{suffix_str}_predictions.npy
        preds_file = f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout{suffix_str}_predictions.npy"
        insts_file = f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout{suffix_str}_instances.npy"
        labels_file = f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout{suffix_str}_labels.npy"
    else:
        raise ValueError(f"Unsupported ensemble type: {ensemble_type}")

    predictions_path = os.path.join(directory, preds_file)
    instances_path = os.path.join(directory, insts_file)
    labels_path = os.path.join(directory, labels_file)

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not os.path.exists(instances_path):
        raise FileNotFoundError(f"Instances file not found: {instances_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Load them
    predictions = np.load(predictions_path, allow_pickle=True)
    instances = np.load(instances_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    print(f"Loaded predictions from: {predictions_path}")
    print(f"Loaded instances   from: {instances_path}")
    print(f"Loaded labels      from: {labels_path}")
    print(
        f"predictions shape: {predictions.shape}, instances shape: {instances.shape}, labels shape: {labels.shape}"
    )

    return predictions, instances, labels


# def load_results_real_data(
#     dataset_name: str,
#     model_type: str,
#     ensemble_type: str,
#     ensemble_size: int = None,
#     directory: str = None,
#     file_prefix: str = None,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Load the saved predictions, instances, and labels from the specified directory.

#     Parameters
#     ----------
#     dataset_name : str
#         The name of the dataset (e.g., 'CIFAR10' or 'CIFAR100').
#     model_type : str
#         The model architecture used (e.g., 'resnet', 'vgg', or custom).
#     ensemble_type : str
#         The ensemble type ('deep_ensemble' or 'mc_dropout').
#     ensemble_size : int, optional
#         Number of models in the ensemble (if 'deep_ensemble'), or number of MC passes (if 'mc_dropout').
#     directory : str, optional
#         Directory from which to load the results. Defaults to current working directory if None.
#     file_prefix : str, optional
#         Additional prefix to include in file names if you want more control (e.g. a date or run ID).

#     Returns
#     -------
#     predictions : np.ndarray
#         Predictions array with shape:
#           - (N, ensemble_size, n_classes) for 'deep_ensemble'
#           - (N, ensemble_size, n_classes) for 'mc_dropout' (where ensemble_size = #MC passes)
#     instances : np.ndarray
#         The input data array. For real data, shape (N, C, H, W).
#     labels : np.ndarray
#         The true labels array, shape (N,).
#     """
#     if directory is None:
#         directory = os.getcwd()

#     # Build base file name structure
#     # Something like:  <prefix>_CIFAR10_resnet_deep_ensemble_5_predictions.npy
#     # If file_prefix is not None, prepend it
#     prefix_str = f"{file_prefix}_" if file_prefix else ""

#     # Distinguish naming based on ensemble_type
#     if ensemble_type == "deep_ensemble":
#         if ensemble_size is None:
#             raise ValueError("ensemble_size must be specified for 'deep_ensemble'.")
#         preds_file = f"{prefix_str}{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_predictions.npy"
#         insts_file = f"{prefix_str}{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_instances.npy"
#         labels_file = f"{prefix_str}{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_labels.npy"
#     elif ensemble_type == "mc_dropout":
#         # We assume naming like: <prefix>_CIFAR10_resnet_10_mc_dropout_predictions.npy
#         if ensemble_size is None:
#             raise ValueError("ensemble_size must be specified for 'mc_dropout'.")
#         preds_file = f"{prefix_str}{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_predictions.npy"
#         insts_file = f"{prefix_str}{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_instances.npy"
#         labels_file = f"{prefix_str}{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_labels.npy"
#     else:
#         raise ValueError(f"Unsupported ensemble type: {ensemble_type}")

#     # Build full paths
#     predictions_path = os.path.join(directory, preds_file)
#     instances_path = os.path.join(directory, insts_file)
#     labels_path = os.path.join(directory, labels_file)

#     if not os.path.exists(predictions_path):
#         raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
#     if not os.path.exists(instances_path):
#         raise FileNotFoundError(f"Instances file not found: {instances_path}")
#     if not os.path.exists(labels_path):
#         raise FileNotFoundError(f"Labels file not found: {labels_path}")

#     # Load
#     predictions = np.load(predictions_path, allow_pickle=True)
#     instances = np.load(instances_path, allow_pickle=True)
#     labels = np.load(labels_path, allow_pickle=True)

#     print(f"Loaded predictions from: {predictions_path}")
#     print(f"Loaded instances from:   {instances_path}")
#     print(f"Loaded labels from:     {labels_path}")
#     print(
#         f"predictions shape: {predictions.shape}, instances shape: {instances.shape}, labels shape: {labels.shape}"
#     )

#     return predictions, instances, labels
