"""
src/preprocessing/bev_grid.py
-----------------------------
Unified Bird's-Eye View (BEV) spatial coordinate system and grid definitions
for multimodal sensor fusion (Radar, LiDAR, and Camera).
"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2
import torch


@dataclass
class BEVGridConfig:
    """Configuration for standardized BEV rasterization."""
    # Spatial extent in radar metric space (meters)
    x_min: float = -50.0  # Lateral left boundary
    x_max: float = 50.0   # Lateral right boundary
    y_min: float = 0.0    # Longitudinal rear boundary (at radar origin)
    y_max: float = 100.0  # Longitudinal forward boundary
    z_min: float = -2.5   # Vertical bottom boundary
    z_max: float = 4.0    # Vertical top boundary

    # Common raster grid dimensions (pixels)
    bev_height: int = 512 # Grid rows (forward direction)
    bev_width: int = 512  # Grid columns (lateral direction)

    @property
    def dx(self) -> float:
        """Lateral metric resolution (meters per pixel)."""
        return (self.x_max - self.x_min) / float(self.bev_width)

    @property
    def dy(self) -> float:
        """Longitudinal metric resolution (meters per pixel)."""
        return (self.y_max - self.y_min) / float(self.bev_height)

    def metric_to_pixel(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert radar metric coordinates (x, y) in meters to BEV pixel coordinates (col, row).
        Convention:
          col = (x - x_min) / dx
          row = (y_max - y) / dy  (row 0 is forward y_max, row H is radar origin y_min)
        """
        col = (x - self.x_min) / self.dx
        row = (self.y_max - y) / self.dy
        return col, row

    def pixel_to_metric(self, col: np.ndarray, row: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert BEV pixel coordinates (col, row) back to radar metric coordinates (x, y)."""
        x = col * self.dx + self.x_min
        y = self.y_max - row * self.dy
        return x, y

    def is_inside_bev(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Check whether (x, y) coordinates fall within defined metric bounds."""
        return (x >= self.x_min) & (x <= self.x_max) & (y >= self.y_min) & (y <= self.y_max)


def encode_bev_boxes(
    annotations: List[Dict[str, Any]],
    bev_cfg: Optional[BEVGridConfig] = None,
    target_classes: Optional[Dict[str, int]] = None,
    ignore_classes: Optional[set] = None,
    navtech_m_per_px: float = 0.17361,
    radar_center_px: int = 576,
    ignore_dilation_px: int = 2
) -> Dict[str, Any]:
    """
    Parse raw annotations into detection targets and ignore masks.

    Args:
        annotations: List of dicts from annotations.json for a single frame.
        bev_cfg: Standard BEVGridConfig.
        target_classes: Dict mapping target class name to integer ID (default {'car': 0, 'van': 1, 'bus': 2}).
        ignore_classes: Set of class names to treat as ignore regions (default empty set).
        navtech_m_per_px: Metric scale of raw Navtech Cartesian annotations.
        radar_center_px: Origin pixel in raw 1152x1152 Navtech Cartesian radar.
        ignore_dilation_px: Padding around ignored objects to prevent false-negative edge penalties.

    Returns:
        Dict with keys:
            'target_boxes_metric': np.ndarray of shape (N, 5) [x_m, y_m, w_m, l_m, theta_rad]
            'target_boxes_bev': np.ndarray of shape (N, 4) [col, row, w_px, h_px]
            'target_labels': np.ndarray of shape (N,) int64
            'target_class_names': List of str
            'ignore_mask': np.ndarray of shape (bev_height, bev_width) float32 (1.0 = ignore)
            'ignore_boxes_metric': np.ndarray of shape (M, 5) [x_m, y_m, w_m, l_m, theta_rad]
            'raw_annotations': List of original annotation dicts
    """
    bev_cfg = bev_cfg or BEVGridConfig()
    target_classes = target_classes if target_classes is not None else {'car': 0, 'van': 1, 'bus': 2}
    ignore_classes = ignore_classes if ignore_classes is not None else set()

    H, W = bev_cfg.bev_height, bev_cfg.bev_width
    ignore_mask = np.zeros((H, W), dtype=np.float32)

    target_boxes_metric = []
    target_boxes_bev = []
    target_labels = []
    target_names = []

    ignore_boxes_metric = []

    for ann in annotations:
        cname = ann.get('class_name', '')
        pos = ann.get('position', None)
        if not pos or not isinstance(pos, (list, tuple)) or len(pos) < 4:
            continue

        x_tl, y_tl, w_px, h_px = pos[:4]
        rot_deg = float(ann.get('rotation', 0.0))
        rot_rad = float(np.deg2rad(rot_deg))

        # Convert raw Navtech Cartesian upper-left pixel coordinates to center
        cx_px = x_tl + w_px / 2.0
        cy_px = y_tl + h_px / 2.0

        # Convert center to radar metric space (x: lateral right+, y: forward+)
        x_m = (cx_px - radar_center_px) * navtech_m_per_px
        y_m = (radar_center_px - cy_px) * navtech_m_per_px
        w_m = w_px * navtech_m_per_px
        l_m = h_px * navtech_m_per_px

        # Check if center is in front / within BEV range
        if not bev_cfg.is_inside_bev(np.array([x_m]), np.array([y_m]))[0]:
            continue

        # Convert to common BEV pixel coordinates
        col, row = bev_cfg.metric_to_pixel(x_m, y_m)
        w_bev = w_m / bev_cfg.dx
        h_bev = l_m / bev_cfg.dy

        if cname in target_classes:
            target_boxes_metric.append([x_m, y_m, w_m, l_m, rot_rad])
            target_boxes_bev.append([col, row, w_bev, h_bev])
            target_labels.append(target_classes[cname])
            target_names.append(cname)

        elif cname in ignore_classes:
            ignore_boxes_metric.append([x_m, y_m, w_m, l_m, rot_rad])
            # Draw ignore footprint on binary ignore mask
            half_w = (w_bev / 2.0) + ignore_dilation_px
            half_h = (h_bev / 2.0) + ignore_dilation_px
            c1 = int(np.clip(col - half_w, 0, W))
            c2 = int(np.clip(col + half_w + 1, 0, W))
            r1 = int(np.clip(row - half_h, 0, H))
            r2 = int(np.clip(row + half_h + 1, 0, H))
            if c2 > c1 and r2 > r1:
                ignore_mask[r1:r2, c1:c2] = 1.0

    return {
        'target_boxes_metric': np.array(target_boxes_metric, dtype=np.float32) if target_boxes_metric else np.zeros((0, 5), dtype=np.float32),
        'target_boxes_bev': np.array(target_boxes_bev, dtype=np.float32) if target_boxes_bev else np.zeros((0, 4), dtype=np.float32),
        'target_labels': np.array(target_labels, dtype=np.int64) if target_labels else np.zeros((0,), dtype=np.int64),
        'target_class_names': target_names,
        'ignore_mask': ignore_mask,
        'ignore_boxes_metric': np.array(ignore_boxes_metric, dtype=np.float32) if ignore_boxes_metric else np.zeros((0, 5), dtype=np.float32),
        'raw_annotations': annotations
    }


# Backward-compatible alias
encode_targets = encode_bev_boxes

