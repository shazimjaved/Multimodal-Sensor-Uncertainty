"""
src/preprocessing package
-------------------------
Standardized preprocessing modules for Radar, LiDAR, Camera, and common BEV coordinates.
"""

from .bev_grid import BEVGridConfig, encode_bev_boxes
from .radar import RadarPreprocessor
from .lidar import LiDARPreprocessor
from .camera import CameraPreprocessor

__all__ = [
    "BEVGridConfig",
    "encode_bev_boxes",
    "RadarPreprocessor",
    "LiDARPreprocessor",
    "CameraPreprocessor",
]
