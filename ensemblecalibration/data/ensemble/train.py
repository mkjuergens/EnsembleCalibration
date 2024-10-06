import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
import wandb

from ensemblecalibration.data.ensemble.datasets import load_dataset, save_results
from ensemblecalibration.data.ensemble.model import get_resnet_model


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ensemblecalibration.data.ensemble.datasets import load_dataset, save_results
from ensemblecalibration.data.ensemble.model import get_resnet_model


import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ensemblecalibration.data.ensemble.datasets import load_dataset, save_results
from ensemblecalibration.data.ensemble.model import get_resnet_model


def train_model(model, trainloader, valloader, epochs=50, patience=5, device="cuda", dataset_name="", ensemble_size=1, model_idx=1):
    # Initialize Weights & Biases run
    wandb.init(
        project="ensemble-calibration",
        reinit=True,
        config={
            "dataset": dataset_name,
            "ensemble_size": ensemble_size,
            "model_index": model_idx,
            "learning_rate": 0.0005,
            "epochs": epochs
        },
        tags=[f"dataset:{dataset_name}", f"ensemble_size:{ensemble_size}", f"model:{model_idx}"]
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    model.train()

    best_loss = float('inf')
    patience_counter = 0

    # Create the directory to save models if it does not exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Define the path to save the best model
    best_model_path = os.path.join(model_dir, f"best_model_{dataset_name}_ensemble_{ensemble_size}_model_{model_idx}.pth")

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
        with torch.no_grad():
            for val_inputs, val_labels in valloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()

        avg_val_loss = val_loss / len(valloader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")

        # Log losses to Weights & Biases
        wandb.log({
            "Epoch": epoch + 1,
            "Training Loss": avg_train_loss,
            "Validation Loss": avg_val_loss
        })

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


# Test the ensemble models and collect predictions
def test_ensemble(models_paths, testloader, num_classes, device="cuda"):
    ensemble_models = [get_resnet_model(num_classes) for _ in models_paths]

    # Load the saved best model weights for each model in the ensemble
    for i, model_path in enumerate(models_paths):
        ensemble_models[i].load_state_dict(torch.load(model_path))
        ensemble_models[i].to(device)
        ensemble_models[i].eval()

    predictions = []
    labels_list = []
    instances_list = []

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testing Ensemble"):
            inputs = inputs.to(device)
            instances_list.append(inputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            # For each model, collect the predictions for the current batch
            ensemble_preds = []
            for model in ensemble_models:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                ensemble_preds.append(predicted.cpu().numpy())

            # Convert ensemble_preds to have shape (num_models, batch_size)
            ensemble_preds = np.stack(ensemble_preds, axis=0)
            predictions.append(ensemble_preds)

    # Convert lists to numpy arrays
    predictions = np.concatenate(predictions, axis=1)  # Shape: (num_models, total_samples)
    labels = np.concatenate(labels_list)  # Shape: (total_samples,)
    instances = np.concatenate(instances_list)  # Shape: (total_samples, channels, height, width)

    return predictions, instances, labels


if __name__ == "__main__":
    # Number of models in the ensembles
    ensemble_sizes = [5, 10]
    datasets_to_use = ["CIFAR10", "CIFAR100", "MNIST"]

    # Training and testing the ensembles for each dataset
    for dataset_name in datasets_to_use:
        print(f"\nLoading Dataset: {dataset_name}")
        trainloader, valloader, testloader, num_classes = load_dataset(dataset_name)

        for ensemble_size in ensemble_sizes:
            print(f"\nTraining Ensemble with {ensemble_size} Models on {dataset_name}\n")
            ensemble_models = [get_resnet_model(num_classes) for _ in range(ensemble_size)]

            # Store paths to the best models
            best_model_paths = []

            # Train each model in the ensemble
            for idx, model in enumerate(ensemble_models):
                print(f"\nTraining model {idx+1}/{ensemble_size} in the ensemble\n")
                best_model_path = train_model(
                    model, trainloader, valloader,
                    dataset_name=dataset_name,
                    ensemble_size=ensemble_size,
                    model_idx=idx + 1
                )
                best_model_paths.append(best_model_path)

            # Test ensemble using the best saved models
            predictions, instances, labels = test_ensemble(best_model_paths, testloader, num_classes)

            # Save predictions, instances, and labels
            save_results(dataset_name, ensemble_size, predictions, instances, labels)

            print(f"Saved predictions, instances, and labels for ensemble of size {ensemble_size} on {dataset_name}\n")