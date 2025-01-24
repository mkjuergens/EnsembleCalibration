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

from ensemblecalibration.data.real.dataset_utils import load_dataset
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


def evaluate_deep_ensemble(
    model_paths, dataloader, num_classes, device="cuda", model_type="resnet"
):
    """
    Evaluate a Deep Ensemble on a given dataloader (val or test).
    Returns predictions, instances, labels in np arrays.
    """
    ensemble_size = len(model_paths)
    ensemble_models = []

    for path_ckpt in model_paths:
        single_model = get_model(
            num_classes=num_classes,
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
        for inputs, labels in tqdm(dataloader, desc="Evaluating Deep Ensemble"):
            inputs = inputs.to(device)
            all_inputs.append(inputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            ensemble_batch_preds = []
            for model in ensemble_models:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                ensemble_batch_preds.append(probs.cpu().numpy())

            # (ensemble_size, batch_size, num_classes)
            ensemble_batch_preds = np.stack(ensemble_batch_preds, axis=0)
            # => (batch_size, ensemble_size, num_classes)
            ensemble_batch_preds = ensemble_batch_preds.transpose(1, 0, 2)
            all_preds.append(ensemble_batch_preds)

            # Accuracy from mean of ensemble
            mean_probs = np.mean(ensemble_batch_preds, axis=1)
            preds_np = np.argmax(mean_probs, axis=1)
            total += labels.size(0)
            correct += (preds_np == labels.numpy()).sum()

    accuracy = 100.0 * correct / total
    print(f"Ensemble accuracy on this split = {accuracy:.2f}%")

    predictions = np.concatenate(all_preds, axis=0)  # (N, ensemble_size, num_classes)
    labels_arr = np.concatenate(all_labels, axis=0)
    instances_arr = np.concatenate(all_inputs, axis=0)
    return predictions, instances_arr, labels_arr


def evaluate_mc_dropout(model, dataloader, num_classes, device="cuda", num_samples=10):
    """
    Evaluate MC-Dropout with num_samples forward passes on a given dataloader.
    Returns predictions, instances, labels in np arrays.
    """
    model.eval()
    model.to(device)

    def enable_mc_dropout(m: nn.Module):
        if isinstance(m, nn.Dropout):
            m.train()

    all_preds = []
    all_labels = []
    all_inputs = []

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating MC-Dropout"):
            inputs = inputs.to(device)
            all_inputs.append(inputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # gather multiple passes
            samples_list = []
            for _ in range(num_samples):
                model.apply(enable_mc_dropout)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                samples_list.append(probs.cpu().numpy())

            # shape => (num_samples, batch_size, num_classes)
            samples_np = np.stack(samples_list, axis=0)
            # => (batch_size, num_samples, num_classes)
            samples_np = samples_np.transpose(1, 0, 2)
            all_preds.append(samples_np)

            # accuracy from mean
            mean_probs = np.mean(samples_np, axis=1)
            preds_np = np.argmax(mean_probs, axis=1)
            total += labels.size(0)
            correct += (preds_np == labels.numpy()).sum()

    accuracy = 100.0 * correct / total
    print(f"MC-Dropout accuracy on this split = {accuracy:.2f}%")

    predictions = np.concatenate(all_preds, axis=0)  # (N, num_samples, num_classes)
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
        help="Number of models in a deep ensemble, or #MC passes in MC-Dropout.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device.")
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

    print(f"\n=== Loading dataset: {args.dataset} ===")
    trainloader, valloader, testloader, num_classes = load_dataset(
        args.dataset, batch_size=args.batch_size
    )

    if args.ensemble_type == "deep_ensemble":
        # 1) Train each model in the ensemble
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

        # 2) Evaluate on validation set
        preds_val, insts_val, labels_val = evaluate_deep_ensemble(
            model_paths=best_paths,
            dataloader=valloader,
            num_classes=num_classes,
            device=args.device,
            model_type=args.model_type,
        )
        print(
            f"Validation set: preds={preds_val.shape}, insts={insts_val.shape}, labels={labels_val.shape}"
        )

        # 3) Evaluate on test set
        preds_test, insts_test, labels_test = evaluate_deep_ensemble(
            model_paths=best_paths,
            dataloader=testloader,
            num_classes=num_classes,
            device=args.device,
            model_type=args.model_type,
        )
        print(
            f"Test set: preds={preds_test.shape}, insts={insts_test.shape}, labels={labels_test.shape}"
        )

        # 4) Save results
        os.makedirs(args.save_dir, exist_ok=True)

        # validation set
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_type}_{args.ensemble_size}_val_predictions.npy",
            ),
            preds_val,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_type}_{args.ensemble_size}_val_instances.npy",
            ),
            insts_val,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_type}_{args.ensemble_size}_val_labels.npy",
            ),
            labels_val,
        )

        # test set
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_type}_{args.ensemble_size}_test_predictions.npy",
            ),
            preds_test,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_type}_{args.ensemble_size}_test_instances.npy",
            ),
            insts_test,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_type}_{args.ensemble_size}_test_labels.npy",
            ),
            labels_test,
        )

        print(f"Saved val & test predictions to: {args.save_dir}\n")

    elif args.ensemble_type == "mc_dropout":
        # 1) Train single model with MC-Dropout
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

        # 2) Evaluate on validation set
        preds_val, insts_val, labels_val = evaluate_mc_dropout(
            model=mc_model,
            dataloader=valloader,
            num_classes=num_classes,
            device=args.device,
            num_samples=args.ensemble_size,
        )
        print(
            f"Validation set: preds={preds_val.shape}, insts={insts_val.shape}, labels={labels_val.shape}"
        )

        # 3) Evaluate on test set
        preds_test, insts_test, labels_test = evaluate_mc_dropout(
            model=mc_model,
            dataloader=testloader,
            num_classes=num_classes,
            device=args.device,
            num_samples=args.ensemble_size,
        )
        print(
            f"Test set: preds={preds_test.shape}, insts={insts_test.shape}, labels={labels_test.shape}"
        )

        # 4) Save results
        os.makedirs(args.save_dir, exist_ok=True)

        # validation set
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_size}_mc_dropout_val_predictions.npy",
            ),
            preds_val,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_size}_mc_dropout_val_instances.npy",
            ),
            insts_val,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_size}_mc_dropout_val_labels.npy",
            ),
            labels_val,
        )

        # test set
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_size}_mc_dropout_test_predictions.npy",
            ),
            preds_test,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_size}_mc_dropout_test_instances.npy",
            ),
            insts_test,
        )
        np.save(
            os.path.join(
                args.save_dir,
                f"{args.dataset}_{args.model_type}_{args.ensemble_size}_mc_dropout_test_labels.npy",
            ),
            labels_test,
        )

        print(f"Saved val & test predictions to: {args.save_dir}\n")


if __name__ == "__main__":
    main()
