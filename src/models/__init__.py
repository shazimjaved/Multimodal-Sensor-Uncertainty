"""
src/models/__init__.py
----------------------
BEV detection model package exports.
"""

from .backbones import ResNet18BEVBackbone
from .fusion import MultimodalBEVFusion
from .centernet_head import CenterNetBEVHead
from .multimodal_detector import BEVMultimodalDetector
from .losses import (
    GaussianFocalLoss,
    RegL1Loss,
    CenterNetLoss,
    build_centernet_targets,
    gaussian_radius,
)

__all__ = [
    "ResNet18BEVBackbone",
    "MultimodalBEVFusion",
    "CenterNetBEVHead",
    "BEVMultimodalDetector",
    "GaussianFocalLoss",
    "RegL1Loss",
    "CenterNetLoss",
    "build_centernet_targets",
    "gaussian_radius",
]
