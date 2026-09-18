from __future__ import annotations

import torch
from torchvision import models


def _single_channel_convolution(layer: torch.nn.Conv2d) -> torch.nn.Conv2d:
    replacement = torch.nn.Conv2d(
        in_channels=1,
        out_channels=layer.out_channels,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=layer.bias is not None,
        padding_mode=layer.padding_mode,
    )
    with torch.no_grad():
        replacement.weight.copy_(layer.weight.mean(dim=1, keepdim=True))
        if layer.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(layer.bias)
    return replacement


def create_classifier(
    architecture: str,
    class_count: int,
    *,
    pretrained: bool = False,
) -> torch.nn.Module:
    """Create a ResNet-18 or VGG-11-BN spectrogram classifier."""

    name = architecture.lower()
    if name in {"resnet", "resnet18"}:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.conv1 = _single_channel_convolution(model.conv1)
        model.fc = torch.nn.Linear(model.fc.in_features, class_count)
        return model

    if name in {"vgg", "vgg11_bn"}:
        weights = models.VGG11_BN_Weights.DEFAULT if pretrained else None
        model = models.vgg11_bn(weights=weights)
        first_layer = model.features[0]
        if not isinstance(first_layer, torch.nn.Conv2d):
            raise TypeError("Unexpected VGG first layer.")
        model.features[0] = _single_channel_convolution(first_layer)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, class_count)
        return model

    raise ValueError("architecture must be 'resnet18' or 'vgg11_bn'")
