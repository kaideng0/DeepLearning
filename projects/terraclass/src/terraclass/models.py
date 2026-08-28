"""Model definitions for the from-scratch and transfer-learning experiments."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

MODEL_NAMES = ("cnn", "resnet18")


class SimpleCNN(nn.Module):
    """A compact convolutional network suitable for learning the fundamentals."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, images):  # type annotation omitted to keep the lesson uncluttered
        features = self.features(images)
        return self.classifier(features)


def build_model(
    name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Construct a model by name and replace its classifier when necessary."""
    if name == "cnn":
        if freeze_backbone:
            raise ValueError("--freeze-backbone only applies to resnet18")
        return SimpleCNN(num_classes=num_classes)

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model '{name}'. Choose one of: {', '.join(MODEL_NAMES)}")


def default_image_size(model_name: str) -> int:
    """Return a practical image size for each architecture."""
    return 224 if model_name == "resnet18" else 64


def gradcam_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Select the last spatial layer used to create a Grad-CAM heatmap."""
    if model_name == "cnn":
        return model.features[8]
    if model_name == "resnet18":
        return model.layer4[-1].conv2
    raise ValueError(f"Grad-CAM is not configured for model '{model_name}'")
