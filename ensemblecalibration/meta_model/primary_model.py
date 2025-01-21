import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import ResNet18_Weights, VGG19_Weights


class MLPCalW(nn.Module):
    """
    class of the MLP model used to learn the optimal convex combination for the respective
    credal set.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        hidden_layers: int = 1,
        use_relu: bool = True,
        **kwargs,
    ):
        """multi layer perceptron for training the optimal weights of a convex combination of
          predictors in order to receive a calibrated model.
        Parameters
        ----------
        in_channels : int
            number of classes for which point predictions are given in the input tensor
        hidden_dim : int
            hidden dimension of the MLP inner layer
        hidden_layers : int, optional
            number of hidden layers, by default 1
        relu : bool, optional
            whether to use ReLU activation function, by default True
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # layers
        layers = []
        if hidden_layers == 0:
            layers.append(nn.Linear(self.in_channels, out_channels))
        else:
            layers.append(nn.Linear(self.in_channels, self.hidden_dim))
            for i in range(hidden_layers):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                # apply ReLU activation
                if use_relu:
                    layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(self.hidden_dim, out_channels))

        self.layers = nn.Sequential(*layers)
        # remove softmax from here, as it is applied in the loss function? No
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in: torch.Tensor):
        out = self.layers(x_in)
        # reshape output to matrix of weights of two dimensions (N, M)
        out = out.view(-1, out.shape[1])
        # apply softmax to get weights in (0,1) and summing up to 1
        out = self.softmax(out)
        return out


class MLPCalWConv(nn.Module):
    """
    Hybrid CNN-MLP model to process images (e.g., CIFAR-10 or CIFAR-100)
    and learn optimal weights for a convex combination of predictors.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 10,
        hidden_dim: int = 128,
        hidden_layers: int = 1,
        use_relu: bool = True,
        pretrained: bool = True,
        pretrained_model: str = "resnet18",
        
    ):
        """
        Multi-layer perceptron combined with convolutional layers to learn optimal weights
        for a convex combination of predictors.

        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g., 3 for RGB images).
        out_channels : int
            Number of output classes (e.g., 10 for CIFAR-10).
        hidden_dim : int
            Hidden dimension for the MLP inner layer.
        hidden_layers : int, optional
            Number of hidden layers in the MLP, by default 1.
        use_relu : bool, optional
            Whether to use ReLU activation function, by default True.
        pretrained : bool, optional
            Whether to use a pretrained model as the feature extractor, by default False.
        pretrained_model : str, optional
            Pretrained model to use as the feature extractor, either 'resnet18' or 'vgg19',
        """
        super().__init__()

        # # Convolutional feature extractor
        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )

        # # Calculate the size of the flattened feature map
        # self.feature_map_size = 128 * 4 * 4  # Assuming input size of 32x32 (CIFAR)
        # feature extractor
        self.feature_extractor = FeatureExtractor(
            pretrained=pretrained, pretrained_model=pretrained_model
        )
        self.feature_map_size = self.feature_extractor.feature_map_size

        # Fully connected MLP layers nfor learning the convex combination weights
        # layers = []
        # if hidden_layers == 0:
        #     layers.append(nn.Linear(self.feature_map_size, out_channels))
        # else:
        #     layers.append(nn.Linear(self.feature_map_size, hidden_dim))
        #     for _ in range(hidden_layers):
        #         layers.append(nn.Linear(hidden_dim, hidden_dim))
        #         # Apply ReLU activation
        #         if use_relu:
        #             layers.append(nn.LeakyReLU())
        #     layers.append(nn.Linear(hidden_dim, out_channels))

        # self.mlp_layers = nn.Sequential(*layers)

        # # Softmax layer to ensure weights are in (0, 1) and sum to 1
        # self.softmax = nn.Softmax(dim=1)

        self.mlp = MLPCalW(
            in_channels=self.feature_map_size,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            use_relu=use_relu,
        )


    def forward(self, x_in: torch.Tensor):
        # Apply feature extractor
        out = self.feature_extractor(x_in)
        # Apply MLP layers
        out = self.mlp(out)
        return out


class FeatureExtractor(nn.Module):
    """neural network to extract features from images using either a pretrained model
    (ResNet18 or VGG19) or a custom CNN.
    """

    def __init__(self, pretrained: bool = True, pretrained_model: str = "resnet18"):
        """
        Parameters
        ----------
        pretrained : bool, optional
            whether to use a pretrained model as the feature extractor, by default True
        pretrained_model : str, optional
            pretrained model to use as the feature extractor, either 'resnet18' or 'vgg19',
            by default 'resnet18'
        """

        super().__init__()
        self.feature_extractor = None
        self.feature_map_size = None

        if pretrained:
            if pretrained_model == "resnet18":
                self.feature_extractor = models.resnet18(weights = ResNet18_Weights.DEFAULT)
                self.feature_extractor = nn.Sequential(
                    *list(self.feature_extractor.children())[:-1]
                )
                self.feature_map_size = 512
            elif pretrained_model == "vgg19":
                self.feature_extractor = models.vgg19(weights = VGG19_Weights.IMAGENET1K_V1)
                self.feature_extractor = nn.Sequential(
                    *list(self.feature_extractor.children())[:-1]
                )
                self.feature_map_size = 512 * 7 * 7

            else:
                raise ValueError(
                    "Invalid pretrained model. Must be 'resnet18' or 'vgg19'."
                )
        else:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.feature_map_size = 128 * 4 * 4

    def forward(self, x_in: torch.Tensor):
        out = self.feature_extractor(x_in)
        assert not torch.isnan(out).any(), "NaN detected in output of feature layer!"
        out = out.view(out.size(0), -1)
        return out



class MLPCalWithPretrainedModel(nn.Module):
    """
    Hybrid model that uses either a pretrained ResNet18 or VGG19 for feature extraction,
    followed by a linear layer and a Softmax output for classification.
    """

    def __init__(
        self,
        out_channels: int = 10,
        hidden_dim: int = 128,
        hidden_layers: int = 1,
        use_relu: bool = True,
        pretrained_model: str = "resnet",  # Can be 'resnet18' or 'vgg19'
    ):
        """
        Initializes the model with either a pretrained ResNet18 or VGG19 as the feature extractor,
        followed by MLP layers for the final classification.

        Parameters
        ----------
        out_channels : int
            Number of output classes (e.g., 10 for CIFAR-10).
        hidden_dim : int
            Hidden dimension for the MLP inner layer.
        hidden_layers : int, optional
            Number of hidden layers in the MLP, by default 1.
        use_relu : bool, optional
            Whether to use ReLU activation function, by default True.
        pretrained_model : str, optional
            Pretrained model to use as the feature extractor, either 'resnet18' or 'vgg19'.
        """
        super().__init__()

        # Load the pretrained ResNet18 or VGG19 model
        if pretrained_model == "resnet":
            self.feature_extractor = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            # Remove the fully connected layer at the end of ResNet18
            self.feature_extractor = nn.Sequential(
                *list(self.feature_extractor.children())[:-1]
            )
            self.feature_map_size = (
                512  # ResNet18 outputs a 512-dimensional feature vector
            )
        elif pretrained_model == "vgg":
            self.feature_extractor = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            # Remove the classifier layers at the end of VGG19
            self.feature_extractor = nn.Sequential(
                *list(self.feature_extractor.children())[:-1]
            )
            self.feature_map_size = (
                512 * 7 * 7
            )  # VGG19 outputs a 512 x 7 x 7 feature map
        else:
            raise ValueError("pretrained_model must be either 'resnet18' or 'vgg19'")

        # Freeze the pretrained model's weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # self.mlp_layers = nn.Sequential(*layers)
        self.mlp_layers = nn.Sequential(
            nn.Linear(self.feature_map_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(hidden_dim, out_channels),
        )
        self.layers = nn.Sequential(
            nn.Linear(self.feature_map_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_channels),
        )

        # Softmax layer to ensure weights are in (0, 1) and sum to 1
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_in: torch.Tensor):
        # Extract features using the pretrained model
        out = self.feature_extractor(x_in)
        assert not torch.isnan(out).any(), "NaN detected in output of feature layer!"
        # For VGG, flatten the output of the conv layers
        out = out.view(out.size(0), -1)
        # Apply MLP layers
        # for name, param in self.mlp_layers.named_parameters():
        #     if param.grad is not None:
        #         print(f"Layer {name} - Weights min: {param.data.min()}, max: {param.data.max()}")
        #         print(f"Layer {name} - Gradients min: {param.grad.min()}, max: {param.grad.max()}")
        out = self.mlp_layers(out)
        logits = torch.clamp(
            out, min=-10, max=10
        )  # Prevent very large values in MLP output
        assert not torch.isnan(logits).any(), "NaN detected in output of mlp!"
        # Apply softmax to get weights between 0 and 1 and summing up to 1
        out = self.softmax(logits)
        return out


# same model as above, but with pytorch-lightning
class MLPCalWLightning(pl.LightningModule):
    def __init__(
        self,
        loss_fct: nn.Module,
        lr: float,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        hidden_layers: int = 1,
        use_relu: bool = True,
        use_scheduler: bool = False,
    ):
        """meta learning model for learning the optimal weights of a convex combination of
        predictors in order to receive a calibrated model.

        Parameters
        ----------
        loss_fct : nn.Module
            loss function to be used for training
        lr : float
            learning rate
        in_channels : int
            dimension of instance space
        out_channels : int
            number of predictors in the ensemble
        hidden_dim : int

        hidden_layers : int, optional
            _description_, by default 1
        use_relu : bool, optional
            _description_, by default True
        lr_scheduler : _type_, optional
            _description_, by default None
        """
        super().__init__()
        self.model = MLPCalW(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            use_relu=use_relu,
        )
        self.loss_fct = loss_fct
        self.lr = lr
        self.use_scheduler = use_scheduler
        # log hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # batch consists of probabilities in credal set, labels and features
        p_probs, y, x = batch
        pred_lambda = self.model(x.float())
        # calculate loss
        loss = self.loss_fct(p_probs, pred_lambda, y)
        # log
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):

        p_probs, y, x = batch
        pred_lambda = self.model(x.float())
        # calculate loss
        loss = self.loss_fct(p_probs, pred_lambda, y)
        # log
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=50, gamma=0.1
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
