import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
import wandb

from ensemblecalibration.data.ensemble.datasets import load_dataset, save_results
from ensemblecalibration.data.ensemble.model import get_resnet_model


def train_model(model, trainloader, valloader, epochs=50, patience=5, device="cuda"):
    # Initialize the Weights & Biases run
    wandb.init(project="ensemble-calibration-real-data", reinit=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    best_loss = float('inf')
    patience_counter = 0

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

        # Early Stopping Logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0  # Reset the counter
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Finish the Weights & Biases run
    wandb.finish()


# Test the ensemble models and collect predictions
def test_ensemble(models, testloader, device="cuda"):
    predictions = []
    labels_list = []
    instances_list = []

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testing Ensemble"):
            inputs = inputs.to(device)
            instances_list.append(inputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            ensemble_preds = []
            for model in models:
                model.eval()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                ensemble_preds.append(predicted.cpu().numpy())

            predictions.append(ensemble_preds)

    # Convert lists to numpy arrays
    predictions = np.array(predictions)  # Shape: (num_batches, num_models, batch_size)
    labels = np.concatenate(labels_list)
    instances = np.concatenate(instances_list)

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

            # Train each model in the ensemble
            for idx, model in enumerate(ensemble_models):
                print(f"\nTraining model {idx+1}/{ensemble_size} in the ensemble\n")
                train_model(model, trainloader, valloader)

            # Test ensemble and get predictions
            predictions, instances, labels = test_ensemble(ensemble_models, testloader)

            # Save predictions, instances, and labels
            save_results(dataset_name, ensemble_size, predictions, instances, labels)

            print(f"Saved predictions, instances, and labels for ensemble of size {ensemble_size} on {dataset_name}\n")