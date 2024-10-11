import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ensemblecalibration.data.ensemble.dataset_utils import load_dataset, save_results
from ensemblecalibration.data.ensemble.model import get_model


def train_model(
    model,
    trainloader,
    valloader,
    epochs=50,
    patience=5,
    device="cuda",
    dataset_name="",
    project_name: str = "ensemble-calibration",
    ensemble_size=10,
    model_idx=1,
):
    """trains a given model on the training set and validates it on the validation set. The best
    model is saved based on the validation loss.

    Parameters
    ----------
    model : _type_
        model to train
    trainloader : torch.utils.data.DataLoader
        training data loader
    valloader : torch.utils.data.DataLoader
        validation data loader
    epochs : int, optional
        number of epochs, by default 50
    patience : int, optional
        , by default 5
    device : str, optional
             by default "cuda"
    dataset_name : str, optional

    ensemble_size : int, optional
        _description_, by default 1
    model_idx : int, optional
        index of the model that is trained, by default 1

    Returns
    -------
    str
        path to the saved best model
    """
    # Initialize Weights & Biases run
    wandb.init(
        project=project_name,
        reinit=True,
        config={
            "dataset": dataset_name,
            "ensemble_size": ensemble_size,
            "model_index": model_idx,
            "learning_rate": 0.0005,
            "epochs": epochs,
        },
        tags=[
            f"dataset:{dataset_name}",
            f"ensemble_size:{ensemble_size}",
            f"model:{model_idx}",
        ],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    model.train()

    best_loss = float("inf")
    patience_counter = 0

    # Create the directory to save models if it does not exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Define the path to save the best model
    best_model_path = os.path.join(
        model_dir,
        f"best_model_{dataset_name}_ensemble_{ensemble_size}_model_{model_idx}.pth",
    )

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")

        # Validation Phase
        model.eval()
        val_loss = 0.0
        # track also accuracy on validation set
        correct = 0
        total = 0
        with torch.no_grad():
            for val_inputs, val_labels in valloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

                # Calculate accuracy
                _, predicted = torch.max(val_outputs, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        avg_val_loss = val_loss / len(valloader)
        val_accuracy = 100 * correct / total

        print(
            f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%"
        )

        # Log losses and accuracy to Weights & Biases
        wandb.log(
            {
                "Epoch": epoch + 1,
                "Training Loss": avg_train_loss,
                "Validation Loss": avg_val_loss,
                "Validation Accuracy": val_accuracy,
            }
        )

        # Scheduler Step (based on validation loss)
        scheduler.step(avg_val_loss)

        # Check if this is the best model so far
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0  # Reset the counter
            # Save the best model weights
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation loss: {best_loss}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Finish Weights & Biases run
    wandb.finish()
    return best_model_path


# MCDropout: Enable dropout during inference
def enable_mc_dropout(model):
    model.train()  # Ensure dropout is active
    # Keep batch norm in eval mode
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


# Perform MCDropout inference
def mc_dropout_inference(model, inputs, num_samples=50):
    model.train()  # Ensure dropout is active
    enable_mc_dropout(model)  # Only enable dropout layers

    all_probs = []
    for _ in range(num_samples):
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            all_probs.append(probs.cpu().numpy())

    all_probs = np.stack(
        all_probs, axis=0
    )  # Shape: [num_samples, batch_size, num_classes]
    mean_probs = np.mean(all_probs, axis=0)  # Mean over the stochastic passes
    return mean_probs, all_probs


def test_model(model_or_paths, testloader, num_classes, ensemble_type="deep_ensemble", num_samples=50, device="cuda", model_type="resnet"):
    """
    Test either an ensemble of models (deep ensemble) or a single model with MCDropout.
    
    Parameters:
    - model_or_paths: list of model paths (for deep ensemble) or a single model instance (for MCDropout).
    - testloader: DataLoader for the test set.
    - num_classes: Number of classes in the dataset.
    - ensemble_type: Either "deep_ensemble" or "mc_dropout".
    - num_samples: Number of samples for MCDropout (default is 50).
    - device: Device to run the model on ("cuda" or "cpu").
    - model_type: The type of model architecture ('resnet' or 'vgg').
    """
    if ensemble_type == "deep_ensemble":
        # Deep Ensemble: Load each model in the ensemble
        ensemble_models = [get_model(num_classes, model_type=model_type, ensemble_type="deep_ensemble").to(device) for _ in model_or_paths]
        
        # Load the saved best model weights for each model in the ensemble
        for i, model_path in enumerate(model_or_paths):
            ensemble_models[i].load_state_dict(torch.load(model_path))

        predictions = []
        labels_list = []
        instances_list = []

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(testloader, desc=f"Testing {ensemble_type}"):
                inputs = inputs.to(device)
                instances_list.append(inputs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

                # For each model, collect predictions and calculate accuracy
                ensemble_probs = []
                for model in ensemble_models:
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    ensemble_probs.append(probs.cpu().numpy())

                # Take the mean of predictions across the ensemble
                ensemble_probs = np.mean(ensemble_probs, axis=0)  # Shape: [batch_size, num_classes]
                predictions.append(ensemble_probs)

                # Get predicted classes and calculate accuracy
                predicted = np.argmax(ensemble_probs, axis=1)
                total += labels.size(0)
                correct += (predicted == labels.cpu().numpy()).sum()

        test_accuracy = 100 * correct / total
        print(f"Test Accuracy: {test_accuracy}%")

        predictions = np.concatenate(predictions, axis=0)  # Shape: (total_samples, num_classes)
        labels = np.concatenate(labels_list)  # Shape: (total_samples,)
        instances = np.concatenate(instances_list)  # Shape: (total_samples, channels, height, width)

        return predictions, instances, labels

    elif ensemble_type == "mc_dropout":
        model = model_or_paths  # model_or_paths is actually a single model in this case
        model.to(device)
        model.eval()  # Set model to evaluation mode for MCDropout

        predictions = []
        labels_list = []
        instances_list = []

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(testloader, desc=f"Testing {ensemble_type}"):
                inputs = inputs.to(device)
                instances_list.append(inputs.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

                # Perform MCDropout inference
                mean_probs, _ = mc_dropout_inference(model, inputs, num_samples)
                predictions.append(mean_probs)

                # Get predicted classes and calculate accuracy
                predicted = np.argmax(mean_probs, axis=1)
                total += labels.size(0)
                correct += (predicted == labels.cpu().numpy()).sum()

        test_accuracy = 100 * correct / total
        print(f"Test Accuracy: {test_accuracy}%")

        predictions = np.concatenate(predictions, axis=0)  # Shape: (total_samples, num_classes)
        labels = np.concatenate(labels_list)  # Shape: (total_samples,)
        instances = np.concatenate(instances_list)  # Shape: (total_samples, channels, height, width)

        return predictions, instances, labels


if __name__ == "__main__":
    # Configurations
    ensemble_sizes = [5, 10]  # Number of models in ensembles
    datasets_to_use = ["CIFAR10", "CIFAR100"]  # Datasets
    ens_types = [
        "mc_dropout",
        "deep_ensemble"
    ]  # Ensemble type: deep ensemble or MCDropout
    model_types = ["resnet", "vgg"]  # Model architectures

    # Specify the directory to save predictions
    save_directory = "ensemble_results"
    os.makedirs(save_directory, exist_ok=True)

    # Iterate over each dataset
    for dataset_name in datasets_to_use:
        print(f"\nLoading Dataset: {dataset_name}")
        trainloader, valloader, testloader, num_classes = load_dataset(dataset_name)

        # Iterate over each model type (ResNet, VGG, etc.)
        for model_type in model_types:
            print(f"\nUsing Model: {model_type}")

            # Iterate over each ensemble type (deep ensemble, MCDropout)
            for ensemble_type in ens_types:
                print(f"\nRunning {ensemble_type} on {model_type} for {dataset_name}")

                if ensemble_type == "deep_ensemble":
                    # Iterate over ensemble sizes (5, 10)
                    for ensemble_size in ensemble_sizes:
                        print(
                            f"\nTraining Deep Ensemble with {ensemble_size} Models on {dataset_name}\n"
                        )

                        # Store paths to the best models
                        best_model_paths = []

                        # Train each model in the ensemble
                        for idx in range(ensemble_size):
                            print(
                                f"\nTraining model {idx + 1}/{ensemble_size} in the ensemble\n"
                            )
                            model = get_model(
                                num_classes, model_type, ensemble_type=ensemble_type
                            )
                            best_model_path = train_model(
                                model,
                                trainloader,
                                valloader,
                                dataset_name=dataset_name,
                                ensemble_size=ensemble_size,
                                model_idx=idx + 1,
                            )
                            best_model_paths.append(best_model_path)

                        # Test ensemble using the best saved models
                        predictions, instances, labels = test_model(
                            best_model_paths, testloader, num_classes, ensemble_type, model_type=model_type
                        )

                        # Save predictions, instances, and labels
                        np.save(
                            os.path.join(
                                save_directory,
                                f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_predictions.npy",
                            ),
                            predictions,
                        )
                        np.save(
                            os.path.join(
                                save_directory,
                                f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_instances.npy",
                            ),
                            instances,
                        )
                        np.save(
                            os.path.join(
                                save_directory,
                                f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_labels.npy",
                            ),
                            labels,
                        )

                        print(
                            f"Saved predictions for {ensemble_size}-model {ensemble_type} on {dataset_name} with {model_type}\n"
                        )

                elif ensemble_type == "mc_dropout":
                    print(f"\nRunning MCDropout on {model_type} for {dataset_name}")

                    # For MCDropout, a single model is trained and multiple forward passes are used
                    model = get_model(
                        num_classes, model_type, ensemble_type=ensemble_type
                    )
                    best_model_path = train_model(
                        model,
                        trainloader,
                        valloader,
                        dataset_name=dataset_name,
                        ensemble_size=1,  # Only one model, no ensemble size
                        model_idx=1,
                    )

                    # Test MCDropout with multiple stochastic passes
                    predictions, instances, labels = test_model(
                        model,
                        testloader,
                        num_classes,
                        ensemble_type="mc_dropout",
                        num_samples=50,
                        model_type=model_type
                    )

                    # Save predictions, instances, and labels
                    np.save(
                        os.path.join(
                            save_directory,
                            f"{dataset_name}_{model_type}_mc_dropout_predictions.npy",
                        ),
                        predictions,
                    )
                    np.save(
                        os.path.join(
                            save_directory,
                            f"{dataset_name}_{model_type}_mc_dropout_instances.npy",
                        ),
                        instances,
                    )
                    np.save(
                        os.path.join(
                            save_directory,
                            f"{dataset_name}_{model_type}_mc_dropout_labels.npy",
                        ),
                        labels,
                    )

                    print(
                        f"Saved predictions for MCDropout on {dataset_name} with {model_type}\n"
                    )
