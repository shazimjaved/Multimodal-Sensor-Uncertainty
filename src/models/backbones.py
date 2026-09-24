"""
src/models/backbones.py
-----------------------
Feature extraction backbones for BEV sensor representations.
Implements ResNet-18 backbones adapted for 1-channel radar and 3-channel
LiDAR / Camera BEV representations, combined with a lightweight top-down
Feature Pyramid (FPN) neck returning stride-4 (128x128) feature maps.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ResNet18BEVBackbone(nn.Module):
    """
    ResNet-18 backbone with top-down lateral FPN decoder producing
    stride-4 feature representations for 512x512 BEV inputs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        pretrained: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Instantiate torchvision ResNet-18
        base = resnet18(weights=None)

        # Adapt first conv layer if in_channels != 3 (e.g. 1-channel radar)
        if in_channels != 3:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.conv1 = base.conv1

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # ResNet-18 stages:
        # layer1: stride 4  (64 channels,  128x128)
        # layer2: stride 8  (128 channels, 64x64)
        # layer3: stride 16 (256 channels, 32x32)
        # layer4: stride 32 (512 channels, 16x16)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Lateral 1x1 convolutions projecting all stages to out_channels
        self.lat4 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(256, out_channels, kernel_size=1)
        self.lat2 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.lat1 = nn.Conv2d(64, out_channels, kernel_size=1)

        # Smooth convolution on merged stride-4 feature map
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, in_channels, 512, 512)

        Returns:
            feat: Tensor of shape (B, out_channels, 128, 128)
        """
        # Stem
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 64, 256, 256)
        x = self.maxpool(x)                     # (B, 64, 128, 128)

        # Stage features
        c1 = self.layer1(x)                     # (B, 64, 128, 128)
        c2 = self.layer2(c1)                    # (B, 128, 64, 64)
        c3 = self.layer3(c2)                    # (B, 256, 32, 32)
        c4 = self.layer4(c3)                    # (B, 512, 16, 16)

        # Top-down FPN pathway to stride 4 (128x128)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)
        p1 = self.lat1(c1) + F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)

        out = self.smooth(p1)                   # (B, out_channels, 128, 128)
        return out
