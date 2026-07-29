"""
Face Recognition Models
========================
Custom CNN (baseline) and ResNet18 transfer-learning model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


IMG_SIZE = 224


class FaceRecognitionCNN(nn.Module):
    """Small grayscale CNN — kept for backward compatibility with old checkpoints."""

    def __init__(self, num_classes, img_size=100):
        super().__init__()
        self.img_size = img_size

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        flat = 256 * (img_size // 16) * (img_size // 16)
        self.fc1 = nn.Linear(flat, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def _features(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self._features(x)
        x = self.dropout(x)
        return self.fc2(x)

    def get_embedding(self, x):
        return self._features(x)


class FaceRecognitionResNet(nn.Module):
    """Pretrained ResNet18 fine-tuned for face identification."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def get_embedding(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)


def build_model(model_name, num_classes, pretrained=True):
    if model_name == "resnet18":
        return FaceRecognitionResNet(num_classes, pretrained=pretrained)
    if model_name == "cnn":
        return FaceRecognitionCNN(num_classes)
    raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    model = FaceRecognitionResNet(num_classes=10)
    dummy = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
    print(f"ResNet output: {model(dummy).shape}")
    print(f"Embedding: {model.get_embedding(dummy).shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
