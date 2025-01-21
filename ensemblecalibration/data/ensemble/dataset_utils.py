import os
from collections import defaultdict
import numpy as np
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from ensemblecalibration.data.dataset import MLPDataset


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


def load_results(
    dataset_name, model_type, ensemble_type, ensemble_size=None, directory=None
):
    """
    Load the saved predictions, instances, and labels from the specified directory.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset (e.g., 'CIFAR10').
    model_type : str
        The type of model architecture used (e.g., 'resnet' or 'vgg').
    ensemble_type : str
        The ensemble type ('deep_ensemble' or 'mc_dropout').
    ensemble_size : int, optional
        The number of models in the ensemble (only applicable for 'deep_ensemble').
    directory : str, optional
        The directory from which to load the results. Default is the current working directory.

    Returns
    -------
    predictions : np.ndarray
        The predictions made by the ensemble or MCDropout model.
    instances : np.ndarray
        The input instances used for the predictions.
    labels : np.ndarray
        The true labels of the instances.
    """
    # Use the current directory if no directory is provided
    if directory is None:
        directory = os.getcwd()

    # Handle file names based on ensemble type
    if ensemble_type == "deep_ensemble":
        # For deep ensembles, use the ensemble_size in the file names
        predictions_path = os.path.join(
            directory,
            f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_predictions.npy",
        )
        instances_path = os.path.join(
            directory,
            f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_instances.npy",
        )
        labels_path = os.path.join(
            directory,
            f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_labels.npy",
        )
    elif ensemble_type == "mc_dropout":
        # For MCDropout, no need for ensemble_size in the file names
        predictions_path = os.path.join(
            directory,
            f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_predictions.npy",
        )
        instances_path = os.path.join(
            directory,
            f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_instances.npy",
        )
        labels_path = os.path.join(
            directory,
            f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_labels.npy",
        )
    else:
        raise ValueError(f"Unsupported ensemble type: {ensemble_type}")

    # Load the results
    predictions = np.load(predictions_path, allow_pickle=True)
    instances = np.load(instances_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    return predictions, instances, labels
