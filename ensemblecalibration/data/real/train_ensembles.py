import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ensemblecalibration.data.real.dataset_utils import load_dataset, save_results
from ensemblecalibration.data.real.model import get_model


def train_single_model(
    model: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    device="cuda",
    epochs=50,
    patience=5,
    lr=0.01,
    weight_decay=1e-4,
    dataset_name="",
    ensemble_type="deep_ensemble",
    model_idx=1,
    model_dir: str = "models",
    project_name="ensemble-training",
) -> str:
    """
    Train a single model (ResNet/VGG or MC-Dropout), with early stopping based on validation loss,
    logging to Weights & Biases.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    trainloader : DataLoader
        DataLoader for training set.
    valloader : DataLoader
        DataLoader for validation set.
    device : str, optional
        Compute device, e.g. "cuda" or "cpu".
    epochs : int, optional
        Number of training epochs.
    patience : int, optional
        Early stopping patience.
    lr : float, optional
        Learning rate.
    weight_decay : float, optional
        Weight decay for optimizer.
    dataset_name : str, optional
        Name of dataset (for logging).
    ensemble_type : str, optional
        "deep_ensemble" or "mc_dropout".
    model_idx : int, optional
        Model index if training an ensemble.
    model_dir : str, optional
        Directory to save best checkpoint.
    project_name : str, optional
        W&B project name.

    Returns
    -------
    str
        Path to the saved best model checkpoint file.
    """
    wandb.init(
        project=project_name,
        reinit=True,
        config={
            "dataset": dataset_name,
            "ensemble_type": ensemble_type,
            "model_index": model_idx,
            "learning_rate": lr,
            "epochs": epochs,
        },
        tags=[
            f"dataset:{dataset_name}",
            f"ensemble_type:{ensemble_type}",
            f"model:{model_idx}",
        ],
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    best_loss = float("inf")
    patience_counter = 0

    os.makedirs(model_dir, exist_ok=True)

    best_model_path = os.path.join(
        model_dir,
        f"best_model_{dataset_name}_{ensemble_type}_model_{model_idx}.pth",
    )

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(
            trainloader, desc=f"[Epoch {epoch+1}/{epochs}] Training"
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for val_inputs, val_labels in valloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                batch_loss = criterion(val_outputs, val_labels)
                val_loss += batch_loss.item()

                # Accuracy
                _, predicted = torch.max(val_outputs, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()

        avg_val_loss = val_loss / len(valloader)
        val_accuracy = 100.0 * correct / total

        wandb.log(
            {
                "Epoch": epoch + 1,
                "Train Loss": avg_train_loss,
                "Val Loss": avg_val_loss,
                "Val Accuracy": val_accuracy,
            }
        )

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, "
            f"Val Acc={val_accuracy:.2f}%"
        )

        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch+1} [Val Loss={best_loss:.4f}]")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    wandb.finish()
    return best_model_path


def test_deep_ensemble(
    model_paths, testloader, num_classes, device="cuda", model_type="resnet"
):
    """
    Evaluate a Deep Ensemble of multiple saved models on the test set,
    returning ensemble predictions, instances, and labels.

    Returns:
      predictions: (N, ensemble_size, num_classes)
      instances:   (N, channels, height, width)
      labels:      (N,)
    """
    ensemble_size = len(model_paths)
    ensemble_models = []

    for i, path_ckpt in enumerate(model_paths):
        # Create the same architecture and load weights
        single_model = get_model(
            num_classes,
            model_type=model_type,
            ensemble_type="deep_ensemble",
            device=device,
        )
        single_model.load_state_dict(torch.load(path_ckpt, map_location=device))
        single_model.eval()
        ensemble_models.append(single_model)

    all_preds = []
    all_labels = []
    all_inputs = []

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testing Deep Ensemble"):
            inputs = inputs.to(device)
            batch_size = inputs.shape[0]
            all_inputs.append(inputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # predictions from each model
            ensemble_batch_preds = []
            for model in ensemble_models:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                ensemble_batch_preds.append(probs.cpu().numpy())

            # shape => (ensemble_size, batch_size, num_classes)
            ensemble_batch_preds = np.stack(ensemble_batch_preds, axis=0)
            # transpose => (batch_size, ensemble_size, num_classes)
            ensemble_batch_preds = ensemble_batch_preds.transpose(1, 0, 2)
            all_preds.append(ensemble_batch_preds)

            # ensemble-based accuracy
            mean_probs = np.mean(ensemble_batch_preds, axis=1)
            preds_np = np.argmax(mean_probs, axis=1)
            total += labels.size(0)
            correct += (preds_np == labels.numpy()).sum()

    test_acc = 100.0 * correct / total
    print(f"Test Accuracy (Deep Ensemble) = {test_acc:.2f}%")

    predictions = np.concatenate(all_preds, axis=0)  # (N, ensemble_size, num_classes)
    labels_arr = np.concatenate(all_labels, axis=0)
    instances_arr = np.concatenate(all_inputs, axis=0)

    return predictions, instances_arr, labels_arr


def test_mc_dropout(
    model, testloader, num_classes: int, device="cuda", num_samples: int = 10
):
    """
    Evaluate MCDropout by sampling multiple forward passes per input.

    Parameters
    ----------
    model : nn.Module
        A single MCDropoutModel or similar model with dropout active in eval mode.
    testloader : DataLoader
        Test set loader.
    num_classes : int
        Number of classes in the dataset.
    device : str, optional
        Computation device.
    num_samples : int, optional
        Number of forward passes (MC samples) to use.

    Returns
    -------
    predictions : np.ndarray
        Shape (n_samples, num_samples, num_classes).
    instances : np.ndarray
        Shape (n_samples, channels, height, width).
    labels : np.ndarray
        Shape (n_samples,).
    """
    model.eval()
    model = model.to(device)

    # function that forcibly leaves dropout "on"
    def enable_mc_dropout(m: nn.Module):
        if isinstance(m, nn.Dropout):
            m.train()

    # We'll define a local helper for inference
    def mc_inference(model, inputs):
        model.apply(enable_mc_dropout)  # set dropout layers to train mode
        # gather samples
        samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                out = model(inputs)
                samples.append(torch.softmax(out, dim=1).cpu().numpy())
        # shape: (num_samples, batch_size, num_classes)
        samples = np.stack(samples, axis=0)
        return samples

    all_preds = []
    all_labels = []
    all_inputs = []

    correct = 0
    total = 0

    for inputs, labels in tqdm(testloader, desc="Testing MC-Dropout"):
        inputs = inputs.to(device)
        all_inputs.append(inputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        # shape: (num_samples, batch_size, num_classes)
        samples = mc_inference(model, inputs)
        samples = samples.swapaxes(0, 1)  # (batch_size, num_samples, num_classes)
        all_preds.append(samples)

        # ensemble-based accuracy from mean
        mean_probs = np.mean(samples, axis=1)  # shape (batch_size, num_classes)
        preds_np = np.argmax(mean_probs, axis=1)
        total += labels.size(0)
        correct += (preds_np == labels.numpy()).sum()

    test_acc = 100.0 * correct / total
    print(f"Test Accuracy (MC-Dropout): {test_acc:.2f}%")

    predictions = np.concatenate(
        all_preds, axis=0
    )  # shape (N, num_samples, num_classes)
    labels_arr = np.concatenate(all_labels, axis=0)
    instances_arr = np.concatenate(all_inputs, axis=0)
    return predictions, instances_arr, labels_arr


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ensembles on CIFAR data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR10", "CIFAR100", "MNIST"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet",
        choices=["resnet", "vgg"],
        help="Model architecture.",
    )
    parser.add_argument(
        "--ensemble_type",
        type=str,
        default="deep_ensemble",
        choices=["deep_ensemble", "mc_dropout"],
        help="Ensemble approach.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of models in a deep ensemble, or # of MC passes in MC-Dropout.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training/evaluation.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience for early stopping."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Computation device, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save trained model checkpoints.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="ensemble_results",
        help="Directory to save predictions, instances, and labels.",
    )
    args = parser.parse_args()

    # 1) Load dataset
    print(f"\n=== Loading dataset: {args.dataset} ===")
    trainloader, valloader, testloader, num_classes = load_dataset(
        args.dataset, batch_size=args.batch_size
    )

    # 2) If ensemble_type=deep_ensemble, train 'ensemble_size' models
    if args.ensemble_type == "deep_ensemble":
        best_paths = []
        for idx in range(args.ensemble_size):
            print(
                f"\nTraining model {idx+1}/{args.ensemble_size} for {args.ensemble_type} ..."
            )
            single_model = get_model(
                num_classes=num_classes,
                model_type=args.model_type,
                ensemble_type="deep_ensemble",
                device=args.device,
            )
            ckpt_path = train_single_model(
                model=single_model,
                trainloader=trainloader,
                valloader=valloader,
                device=args.device,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                dataset_name=args.dataset,
                ensemble_type=args.ensemble_type,
                model_idx=idx + 1,
                model_dir=args.model_dir,
            )
            best_paths.append(ckpt_path)

        # Evaluate
        preds, instances, labels = test_deep_ensemble(
            model_paths=best_paths,
            testloader=testloader,
            num_classes=num_classes,
            device=args.device,
            model_type=args.model_type,
        )
        print(
            f"preds shape: {preds.shape}, instances: {instances.shape}, labels: {labels.shape}"
        )

        # Save predictions
        os.makedirs(args.save_dir, exist_ok=True)
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_type}_{args.ensemble_size}_predictions.npy",
            ),
            preds,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_type}_{args.ensemble_size}_instances.npy",
            ),
            instances,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_type}_{args.ensemble_size}_labels.npy",
            ),
            labels,
        )

        print(f"Saved predictions to: {args.save_dir}\n")

    # 3) If ensemble_type=mc_dropout, train 1 model, do 'ensemble_size' passes
    elif args.ensemble_type == "mc_dropout":
        print(
            f"\nTraining a single model with MCDropout for {args.ensemble_size} passes ..."
        )
        mc_model = get_model(
            num_classes=num_classes,
            model_type=args.model_type,
            ensemble_type="mc_dropout",
            device=args.device,
        )
        ckpt_path = train_single_model(
            model=mc_model,
            trainloader=trainloader,
            valloader=valloader,
            device=args.device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            dataset_name=args.dataset,
            ensemble_type="mc_dropout",
            model_idx=1,
            model_dir=args.model_dir,
        )
        # Load best model
        mc_model.load_state_dict(torch.load(ckpt_path, map_location=args.device))

        # Evaluate with multiple passes
        preds, instances, labels = test_mc_dropout(
            model=mc_model,
            testloader=testloader,
            num_classes=num_classes,
            device=args.device,
            num_samples=args.ensemble_size,
        )
        print(
            f"preds shape: {preds.shape}, instances: {instances.shape}, labels: {labels.shape}"
        )

        os.makedirs(args.save_dir, exist_ok=True)
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_size}_mc_dropout_predictions.npy",
            ),
            preds,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_size}_mc_dropout_instances.npy",
            ),
            instances,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_size}_mc_dropout_labels.npy",
            ),
            labels,
        )

        print(f"Saved predictions to: {args.save_dir}\n")


# # # MCDropout: Enable dropout during inference
# # def enable_mc_dropout(model):
# #     model.train()  # Ensure dropout is active
# #     # Keep batch norm in eval mode
# #     for module in model.modules():
# #         if isinstance(module, nn.BatchNorm2d):
# #             module.eval()


# # Perform MCDropout inference
# # def mc_dropout_inference(model, inputs, num_samples=50):
# #     model.train()  # Ensure dropout is active
# #     enable_mc_dropout(model)  # Only enable dropout layers

# #     all_probs = []
# #     for _ in range(num_samples):
# #         with torch.no_grad():
# #             outputs = model(inputs)
# #             probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
# #             all_probs.append(probs.cpu().numpy())

# #     all_probs = np.stack(
# #         all_probs, axis=0
# #     )  # Shape: [num_samples, batch_size, num_classes]
# #     mean_probs = np.mean(all_probs, axis=0)  # Mean over the stochastic passes
# #     return mean_probs, all_probs


# def test_model(
#     model_or_paths,
#     testloader,
#     num_classes,
#     ensemble_type="deep_ensemble",
#     num_samples=10,
#     device="cuda:1",
#     model_type="resnet",
# ):
#     """
#     Test either an ensemble of models (deep ensemble) or a single model with MCDropout.

#     Parameters:
#     - model_or_paths: list of model paths (for deep ensemble) or a single model instance (for MCDropout).
#     - testloader: DataLoader for the test set.
#     - num_classes: Number of classes in the dataset.
#     - ensemble_type: Either "deep_ensemble" or "mc_dropout".
#     - num_samples: Number of samples for MCDropout (default is 50).
#     - device: Device to run the model on ("cuda" or "cpu").
#     - model_type: The type of model architecture ('resnet' or 'vgg').

#     Returns:
#     - predictions: A numpy array of shape (n_instances, n_ensembles, n_classes).
#     - instances: A numpy array of the input instances.
#     - labels: A numpy array of the true labels.
#     """
#     if ensemble_type == "deep_ensemble":
#         # Deep Ensemble: Load each model in the ensemble
#         ensemble_models = [
#             get_model(
#                 num_classes,
#                 model_type=model_type,
#                 ensemble_type="deep_ensemble",
#                 device=device,
#             ).to(device)
#             for _ in model_or_paths
#         ]

#         # Load the saved best model weights for each model in the ensemble
#         for i, model_path in enumerate(model_or_paths):
#             ensemble_models[i].load_state_dict(torch.load(model_path))

#         # Store predictions as (n_instances, n_ensembles, n_classes)
#         all_ensemble_predictions = []
#         labels_list = []
#         instances_list = []

#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for inputs, labels in tqdm(testloader, desc=f"Testing {ensemble_type}"):
#                 inputs = inputs.to(device)
#                 instances_list.append(inputs.cpu().numpy())
#                 labels_list.append(labels.cpu().numpy())

#                 # Initialize an empty list to store predictions from all models for this batch
#                 batch_ensemble_predictions = []

#                 # Collect predictions for each model in the ensemble
#                 for model in ensemble_models:
#                     outputs = model(inputs)
#                     probs = torch.softmax(
#                         outputs, dim=1
#                     )  # Shape: [batch_size, num_classes]
#                     batch_ensemble_predictions.append(
#                         probs.cpu().numpy()
#                     )  # Store the predictions for this model

#                 # Stack predictions across models, resulting in shape (n_ensembles, batch_size, num_classes)
#                 batch_ensemble_predictions = np.stack(
#                     batch_ensemble_predictions, axis=0
#                 )  # Shape: (n_ensembles, batch_size, n_classes)

#                 # Transpose to shape (batch_size, n_ensembles, n_classes) so each instance has predictions from each ensemble member
#                 batch_ensemble_predictions = batch_ensemble_predictions.transpose(
#                     1, 0, 2
#                 )  # Shape: (batch_size, n_ensembles, n_classes)

#                 # Append the batch predictions to the overall predictions list
#                 all_ensemble_predictions.append(batch_ensemble_predictions)

#                 # Calculate accuracy based on the mean ensemble prediction
#                 mean_ensemble_probs = np.mean(
#                     batch_ensemble_predictions, axis=1
#                 )  # Shape: [batch_size, n_classes]
#                 predicted = np.argmax(mean_ensemble_probs, axis=1)
#                 total += labels.size(0)
#                 correct += (predicted == labels.cpu().numpy()).sum()

#         test_accuracy = 100 * correct / total
#         print(f"Test Accuracy: {test_accuracy}%")

#         # Concatenate predictions, labels, and instances for the entire dataset
#         predictions = np.concatenate(
#             all_ensemble_predictions, axis=0
#         )  # Shape: (n_instances, n_ensembles, n_classes)
#         labels = np.concatenate(labels_list)  # Shape: (n_instances,)
#         instances = np.concatenate(
#             instances_list
#         )  # Shape: (n_instances, channels, height, width)

#         return predictions, instances, labels

#     elif ensemble_type == "mc_dropout":
#         model = model_or_paths  # model_or_paths is actually a single model in this case
#         model.to(device)
#         model.eval()  # Set model to evaluation mode for MCDropout

#         predictions = []
#         labels_list = []
#         instances_list = []

#         correct = 0
#         total = 0

#         with torch.no_grad():
#             for inputs, labels in tqdm(testloader, desc=f"Testing {ensemble_type}"):
#                 inputs = inputs.to(device)
#                 instances_list.append(inputs.cpu().numpy())
#                 labels_list.append(labels.cpu().numpy())

#                 # Perform MCDropout inference
#                 mean_probs, probs = mc_dropout_inference(model, inputs, num_samples)
#                 probs = probs.swapaxes(0, 1)
#                 assert probs.shape == (
#                     inputs.shape[0],
#                     num_samples,
#                     num_classes,
#                 ), f"predictions of MC dropout are in wrong shape, namely {probs.shape}"
#                 predictions.append(
#                     probs
#                 )  # changed to append predictions of each samples, not mean

#                 # Get predicted classes and calculate accuracy
#                 predicted = np.argmax(mean_probs, axis=1)
#                 total += labels.size(0)
#                 correct += (predicted == labels.cpu().numpy()).sum()

#         test_accuracy = 100 * correct / total
#         print(f"Test Accuracy: {test_accuracy}%")

#         predictions = np.concatenate(
#             predictions, axis=0
#         )  # Shape: (total_samples, num_samples num_classes)
#         labels = np.concatenate(labels_list)  # Shape: (total_samples,)
#         instances = np.concatenate(
#             instances_list
#         )  # Shape: (total_samples, channels, height, width)

#         return predictions, instances, labels


if __name__ == "__main__":
    main()
    # device = "cuda:1"
    # # Configurations
    # ensemble_sizes = [5, 10]  # Number of models in ensembles
    # datasets_to_use = ["CIFAR10", "CIFAR100"]  # Datasets
    # ens_types = [
    #     "mc_dropout",
    #     "deep_ensemble",
    # ]  # Ensemble type: deep ensemble or MCDropout
    # model_types = ["resnet", "vgg"]  # Model architectures

    # # Specify the directory to save predictions
    # save_directory = "ensemble_results"
    # os.makedirs(save_directory, exist_ok=True)

    # # Iterate over each dataset
    # for dataset_name in datasets_to_use:
    #     print(f"\nLoading Dataset: {dataset_name}")
    #     trainloader, valloader, testloader, num_classes = load_dataset(dataset_name)

    #     # Iterate over each model type (ResNet, VGG, etc.)
    #     for model_type in model_types:
    #         print(f"\nUsing Model: {model_type}")

    #         # Iterate over each ensemble type (deep ensemble, MCDropout)
    #         for ensemble_type in ens_types:
    #             print(f"\nRunning {ensemble_type} on {model_type} for {dataset_name}")

    #             if ensemble_type == "deep_ensemble":
    #                 # Iterate over ensemble sizes (5, 10)
    #                 for ensemble_size in ensemble_sizes:
    #                     print(
    #                         f"\nTraining Deep Ensemble with {ensemble_size} Models on {dataset_name}\n"
    #                     )

    #                     # Store paths to the best models
    #                     best_model_paths = []

    #                     # Train each model in the ensemble
    #                     for idx in range(ensemble_size):
    #                         print(
    #                             f"\nTraining model {idx + 1}/{ensemble_size} in the ensemble\n"
    #                         )
    #                         model = get_model(
    #                             num_classes,
    #                             model_type,
    #                             ensemble_type=ensemble_type,
    #                             device=device,
    #                         )
    #                         best_model_path = train_single_model(
    #                             model,
    #                             trainloader,
    #                             valloader,
    #                             dataset_name=dataset_name,
    #                             ensemble_size=ensemble_size,
    #                             model_idx=idx + 1,
    #                         )
    #                         best_model_paths.append(best_model_path)

    #                     # Test ensemble using the best saved models
    #                     predictions, instances, labels = test_model(
    #                         best_model_paths,
    #                         testloader,
    #                         num_classes,
    #                         ensemble_type,
    #                         model_type=model_type,
    #                     )
    #                     assert predictions.shape == (
    #                         10000,
    #                         ensemble_size,
    #                         num_classes,
    #                     ), "predictions in wrong shape"

    #                     # Save predictions, instances, and labels
    #                     np.save(
    #                         os.path.join(
    #                             save_directory,
    #                             f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_predictions.npy",
    #                         ),
    #                         predictions,
    #                     )
    #                     np.save(
    #                         os.path.join(
    #                             save_directory,
    #                             f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_instances.npy",
    #                         ),
    #                         instances,
    #                     )
    #                     np.save(
    #                         os.path.join(
    #                             save_directory,
    #                             f"{dataset_name}_{model_type}_{ensemble_type}_{ensemble_size}_labels.npy",
    #                         ),
    #                         labels,
    #                     )

    #                     print(
    #                         f"Saved predictions for {ensemble_size}-model {ensemble_type} on {dataset_name} with {model_type}\n"
    #                     )

    #             elif ensemble_type == "mc_dropout":
    #                 for ensemble_size in ensemble_sizes:
    #                     print(f"\nRunning MCDropout on {model_type} for {dataset_name}")

    #                     # For MCDropout, a single model is trained and multiple forward passes are used
    #                     model = get_model(
    #                         num_classes,
    #                         model_type,
    #                         ensemble_type=ensemble_type,
    #                         device=device,
    #                     )
    #                     best_model_path = train_single_model(
    #                         model,
    #                         trainloader,
    #                         valloader,
    #                         dataset_name=dataset_name,
    #                         ensemble_size=1,  # Only one model, no ensemble size
    #                         model_idx=1,
    #                         model_type=ensemble_type,
    #                     )

    #                     # Test MCDropout with multiple stochastic passes
    #                     predictions, instances, labels = test_model(
    #                         model,
    #                         testloader,
    #                         num_classes,
    #                         ensemble_type=ensemble_type,
    #                         num_samples=ensemble_size,
    #                         model_type=model_type,
    #                     )
    #                     assert predictions.shape == (
    #                         10000,
    #                         ensemble_size,
    #                         num_classes,
    #                     ), "wrong shape of preds"

    #                     # Save predictions, instances, and labels
    #                     np.save(
    #                         os.path.join(
    #                             save_directory,
    #                             f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_predictions.npy",
    #                         ),
    #                         predictions,
    #                     )
    #                     np.save(
    #                         os.path.join(
    #                             save_directory,
    #                             f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_instances.npy",
    #                         ),
    #                         instances,
    #                     )
    #                     np.save(
    #                         os.path.join(
    #                             save_directory,
    #                             f"{dataset_name}_{model_type}_{ensemble_size}_mc_dropout_labels.npy",
    #                         ),
    #                         labels,
    #                     )

    #                     print(
    #                         f"Saved predictions for MCDropout on {dataset_name} with {model_type}\n"
    #                     )
