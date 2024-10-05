import torch
import torch.nn as nn
from torchvision import models


def get_resnet_model(num_classes, device):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the output layer for the number of classes
    return model.to(device)
