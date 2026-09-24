"""
src/degradation/__init__.py
---------------------------
Multimodal sensor degradation engine exports.
"""

from .camera import CameraDegradation
from .lidar import LiDARDegradation
from .radar import RadarDegradation
from .conditions import DegradationEngine, degrade_sample, get_default_engine
from .real_fog_diagnostic import (
    compute_camera_diagnostics,
    compute_lidar_diagnostics,
    compare_synthetic_vs_real_fog
)

__all__ = [
    "CameraDegradation",
    "LiDARDegradation",
    "RadarDegradation",
    "DegradationEngine",
    "degrade_sample",
    "get_default_engine",
    "compute_camera_diagnostics",
    "compute_lidar_diagnostics",
    "compare_synthetic_vs_real_fog",
]
