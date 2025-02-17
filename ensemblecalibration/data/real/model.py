import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck


# update: Dropout not only in the last layer
class DropoutBlockWrapper(nn.Module):
    """
    Wrap a ResNet BasicBlock or Bottleneck with a dropout after the block forward.
    """

    def __init__(self, original_block, p=0.5):
        super().__init__()
        self.block = original_block
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        out = self.block(x)
        out = self.dropout(out)
        return out


def insert_dropout_in_resnet(resnet_model, p=0.5):
    """
    Recursively wraps each BasicBlock/Bottleneck of a ResNet in DropoutBlockWrapper.
    """
    for name, module in resnet_model.named_children():
        if name.startswith("layer"):  # e.g. layer1, layer2, etc.
            for sub_name, sub_module in module.named_children():
                if isinstance(sub_module, (BasicBlock, Bottleneck)):
                    wrapped = DropoutBlockWrapper(sub_module, p)
                    setattr(module, sub_name, wrapped)
        else:
            insert_dropout_in_resnet(module, p)
    return resnet_model


def insert_dropout_in_vgg(vgg_model, p=0.5):
    """
    Example: Insert dropout in the classifier portion of VGG, after each ReLU.
    You can also add dropout in the convolutional 'features' if desired.
    """
    new_classifier = []
    for layer in vgg_model.classifier:
        new_classifier.append(layer)
        if isinstance(layer, nn.ReLU):
            new_classifier.append(nn.Dropout(p=p))
    # Last layer is typically Linear(..., 1000). We let the final replacement happen in the code below
    vgg_model.classifier = nn.Sequential(*new_classifier)
    return vgg_model


class MCDropoutModel(nn.Module):
    """
    A model that uses MC Dropout. The base_model is assumed to have dropout layers
    inside it if we want more than just one final dropout. We also add a final dropout
    + linear here for classification.
    """

    def __init__(self, base_model, num_classes, dropout_prob=0.5):
        super(MCDropoutModel, self).__init__()
        self.base_model = base_model
        # Identify final in_features
        if isinstance(base_model, models.ResNet):
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif isinstance(base_model, models.VGG):
            in_features = base_model.classifier[-1].in_features
            base_model.classifier[-1] = nn.Identity()
        else:
            raise NotImplementedError("Only ResNet and VGG are implemented for now.")

        # One last dropout + linear
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_model(
    num_classes,
    model_type: str = "resnet",
    device="cuda",
    dropout_prob: float = 0.5,
    ensemble_type: str = "deep_ensemble",
    pretrained: bool = True,
):
    if model_type == "resnet":
        base_model = models.resnet18(pretrained=pretrained)
    elif model_type == "vgg":
        base_model = models.vgg16(pretrained=pretrained)
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
        # 1) Insert dropout in multiple layers within the base model
        if model_type == "resnet":
            base_model = insert_dropout_in_resnet(base_model, p=dropout_prob)
        elif model_type == "vgg":
            base_model = insert_dropout_in_vgg(base_model, p=dropout_prob)
        else:
            raise NotImplementedError("Only resnet and vgg are implemented")
        # 2) Wrap with MCDropoutModel to add final dropout + linear
        mc_model = MCDropoutModel(base_model, num_classes, dropout_prob=dropout_prob)
        return mc_model.to(device)

    elif ensemble_type == "deep_ensemble":
        # Standard final-layer replacement for deep ensemble
        if isinstance(base_model, models.ResNet):
            # replace final fc
            base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        elif isinstance(base_model, models.VGG):
            # replace last classifier layer
            base_model.classifier[-1] = nn.Linear(
                base_model.classifier[-1].in_features, num_classes
            )
        return base_model.to(device)

    else:
        raise ValueError(
            "Unsupported ensemble type. Use 'deep_ensemble' or 'mc_dropout'."
        )
