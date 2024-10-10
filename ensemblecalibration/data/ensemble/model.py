import torch
import torch.nn as nn
from torchvision import models


# def get_model(num_classes, model: str = "resnet", device="cuda"):
#     if model == "resnet":
#         model = models.resnet18(pretrained=True)
#     elif model == "vgg":
#         model = models.vgg16(pretrained=True)
#     else:
#         raise NotImplementedError("Only resnet and vgg are implemented")
#     model.fc = nn.Linear(
#         model.fc.in_features, num_classes
#     )  # Modify the output layer for the number of classes
#     return model.to(device)


class MCDropoutModel(nn.Module):
    """MC Dropout model with a base model and a dropout layer after the base model"""

    def __init__(self, base_model, num_classes, dropout_prob=0.5):
        super(MCDropoutModel, self).__init__()
        self.base_model = base_model
        # Modify the last fully connected layer for the number of classes
        if isinstance(base_model, models.ResNet):
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()  # Remove the original FC layer
        elif isinstance(base_model, models.VGG):
            in_features = base_model.classifier[-1].in_features
            base_model.classifier[-1] = nn.Identity()  # Remove the original FC layer
        else:
            raise NotImplementedError("Only ResNet and VGG are implemented")

        # Add dropout after the final fully connected layer
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)
        return x


def get_model(
    num_classes,
    model_type: str = "resnet",
    device="cuda",
    dropout_prob=0.5,
    ensemble_type="deep_ensemble",
):
    if model_type == "resnet":
        base_model = models.resnet18(pretrained=True)
    elif model_type == "vgg":
        base_model = models.vgg16(pretrained=True)
    else:
        raise NotImplementedError("Only resnet and vgg are implemented")

    if ensemble_type == "mc_dropout":
        # Return MCDropout model
        mc_model = MCDropoutModel(base_model, num_classes, dropout_prob=dropout_prob)
        return mc_model.to(device)
    elif ensemble_type == "deep_ensemble":
        # Modify only the last fully connected layer for the number of classes (standard deep ensemble)
        if isinstance(base_model, models.ResNet):
            base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        elif isinstance(base_model, models.VGG):
            base_model.classifier[-1] = nn.Linear(
                base_model.classifier[-1].in_features, num_classes
            )
        return base_model.to(device)
    else:
        raise ValueError(
            "Unsupported ensemble type. Use 'deep_ensemble' or 'mc_dropout'."
        )
