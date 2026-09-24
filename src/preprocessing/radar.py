"""
src/preprocessing/radar.py
--------------------------
Deterministic preprocessing for Navtech Cartesian radar imagery into
standardized Bird's-Eye View (BEV) tensor representation.
"""

import os
from typing import Optional
import numpy as np
import cv2
import torch

from .bev_grid import BEVGridConfig


class RadarPreprocessor:
    """Preprocesses raw 1152x1152 Navtech Cartesian radar PNG images into BEV tensors."""

    RAW_SIZE = (1152, 1152)
    RAW_CENTER_PX = (576, 576)
    RAW_M_PER_PX = 0.17361

    def __init__(self, bev_cfg: Optional[BEVGridConfig] = None):
        self.bev_cfg = bev_cfg or BEVGridConfig()

        # Compute crop window in raw 1152x1152 pixel coordinates corresponding to metric extent
        # Y in [0, 100m] -> row in [0, 576]
        # X in [-50m, 50m] -> col in [288, 864]
        cx, cy = self.RAW_CENTER_PX
        m = self.RAW_M_PER_PX

        self.row_min = max(0, int(round(cy - self.bev_cfg.y_max / m)))
        self.row_max = min(self.RAW_SIZE[1], int(round(cy - self.bev_cfg.y_min / m)))
        self.col_min = max(0, int(round(cx + self.bev_cfg.x_min / m)))
        self.col_max = min(self.RAW_SIZE[0], int(round(cx + self.bev_cfg.x_max / m)))

    def process_file(self, radar_path: str) -> np.ndarray:
        """
        Load and preprocess a Navtech Cartesian radar PNG image.

        Args:
            radar_path: Path to Navtech Cartesian radar PNG file.

        Returns:
            Normalized BEV array of shape (1, H_bev, W_bev), float32 in [0, 1].
        """
        if not os.path.exists(radar_path):
            raise FileNotFoundError(f"Radar file not found: {radar_path}")

        raw_img = cv2.imread(radar_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise ValueError(f"Failed to decode radar image from {radar_path}")

        return self.process_image(raw_img)

    def process_image(self, raw_img: np.ndarray) -> np.ndarray:
        """
        Crop and resize a raw 1152x1152 radar image into the common BEV grid.

        Args:
            raw_img: 2D uint8 array of shape (1152, 1152).

        Returns:
            Array of shape (1, H_bev, W_bev), float32 in [0, 1].
        """
        # Crop forward-lateral quadrant
        crop = raw_img[self.row_min:self.row_max, self.col_min:self.col_max]

        # Resize to standard BEV resolution
        target_size = (self.bev_cfg.bev_width, self.bev_cfg.bev_height)
        if (crop.shape[1], crop.shape[0]) != target_size:
            interp = cv2.INTER_AREA if (crop.shape[0] > target_size[1]) else cv2.INTER_LINEAR
            bev_img = cv2.resize(crop, target_size, interpolation=interp)
        else:
            bev_img = crop

        # Normalize to [0.0, 1.0] float32
        norm = (bev_img.astype(np.float32) / 255.0)[np.newaxis, :, :]
        return norm
