import torch
import torch.nn as nn
from torchvision import datasets, transforms


class DeepEnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(DeepEnsembleModel, self).__init__()
        if dataset_name == 'MNIST':
            self.fc1 = nn.Linear(28*28, 512)
        elif dataset_name == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(64*6*6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        if dataset_name == 'MNIST':
            x = x.view(-1, 28*28)
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 64*6*6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x