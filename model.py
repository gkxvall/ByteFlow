from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(
    model_name: str,
    num_classes: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    """Load a torchvision backbone and replace the classifier head for ``num_classes``."""
    name = model_name.lower().strip()
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model_name={model_name!r}. v1 supports: resnet18")
