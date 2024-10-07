import os
import torch
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
    print(f"Saved predictions, instances, and labels for ensemble of size {ensemble_size} on {dataset_name}\n")

# # Function to load the saved predictions, instances, and labels
# def load_results(dataset_name, ensemble_size):
#     predictions = np.load(f"{dataset_name}_ensemble_{ensemble_size}_predictions.npy", allow_pickle=True)
#     instances = np.load(f"{dataset_name}_ensemble_{ensemble_size}_instances.npy", allow_pickle=True)
#     labels = np.load(f"{dataset_name}_ensemble_{ensemble_size}_labels.npy", allow_pickle=True)
#     return predictions, instances, labels

def load_results(dataset_name, ensemble_size, directory=None):
    """
    Load the saved predictions, instances, and labels from the specified directory.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset (e.g., 'CIFAR10').
    ensemble_size : int
        The number of models in the ensemble.
    directory : str, optional
        The directory from which to load the results. Default is the current working directory.

    Returns
    -------
    predictions : np.ndarray
        The predictions made by the ensemble.
    instances : np.ndarray
        The input instances used for the predictions.
    labels : np.ndarray
        The true labels of the instances.
    """
    # Use the current directory if no directory is provided
    if directory is None:
        directory = os.getcwd()

    # Create file paths based on the specified directory
    predictions_path = os.path.join(directory, f"{dataset_name}_ensemble_{ensemble_size}_prob_predictions.npy")
    instances_path = os.path.join(directory, f"{dataset_name}_ensemble_{ensemble_size}_instances.npy")
    labels_path = os.path.join(directory, f"{dataset_name}_ensemble_{ensemble_size}_labels.npy")

    # Load the results
    predictions = np.load(predictions_path, allow_pickle=True)
    instances = np.load(instances_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    return predictions, instances, labels
