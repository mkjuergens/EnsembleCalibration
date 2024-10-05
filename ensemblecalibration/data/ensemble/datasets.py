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

# Function to load the saved predictions, instances, and labels
def load_results(dataset_name, ensemble_size):
    predictions = np.load(f"{dataset_name}_ensemble_{ensemble_size}_predictions.npy", allow_pickle=True)
    instances = np.load(f"{dataset_name}_ensemble_{ensemble_size}_instances.npy", allow_pickle=True)
    labels = np.load(f"{dataset_name}_ensemble_{ensemble_size}_labels.npy", allow_pickle=True)
    return predictions, instances, labels



if __name__ == "__main__":
    # test the function
    trainloader, testloader, classes = load_dataset("CIFAR10")
