"""
src/preprocessing/camera.py
---------------------------
Deterministic camera preprocessing and LiDAR-depth-associated
Bird's-Eye View (BEV) projection using validated RADIATE calibration.
"""

import os
from typing import Optional, Tuple
import numpy as np
import cv2
import torch

from .bev_grid import BEVGridConfig
from radiate_fusion import RadiateCalib


class CameraPreprocessor:
    """
    Preprocesses front camera images and maps optical features into common BEV coordinates
    via LiDAR-assisted camera RGB projection into common BEV.
    """

    def __init__(
        self,
        calib: RadiateCalib,
        bev_cfg: Optional[BEVGridConfig] = None,
        bev_dilation_kernel_size: int = 3
    ):
        self.calib = calib
        self.bev_cfg = bev_cfg or BEVGridConfig()
        self.dilation_kernel = (
            cv2.getStructuringElement(cv2.MORPH_RECT, (bev_dilation_kernel_size, bev_dilation_kernel_size))
            if bev_dilation_kernel_size > 1 else None
        )

    def process_file(
        self,
        cam_path: str,
        pts_radar: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load camera image, rectify, normalize, and construct LiDAR-assisted BEV projection.

        Args:
            cam_path: Path to ZED left camera PNG image (raw, unrectified).
            pts_radar: Optional (N, 3+) array of points in radar metric frame.

        Returns:
            cam_img_norm: (3, H_cam, W_cam) float32 array in [0, 1] (rectified RGB).
            camera_bev: (3, H_bev, W_bev) float32 array in [0, 1] (RGB color in BEV).
            camera_bev_mask: (1, H_bev, W_bev) float32 array (1.0 where camera observes BEV).
        """
        if not os.path.exists(cam_path):
            raise FileNotFoundError(f"Camera file not found: {cam_path}")

        bgr = cv2.imread(cam_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to decode camera image from {cam_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Rectify raw unrectified camera image using official RADIATE stereo rectification
        if hasattr(self.calib, 'rectify_left_image'):
            rgb_rect = self.calib.rectify_left_image(rgb)
        else:
            rgb_rect = rgb
        return self.process_image(rgb_rect, pts_radar, is_rectified=True)

    def process_image(
        self,
        cam_rgb: np.ndarray,
        pts_radar: Optional[np.ndarray] = None,
        is_rectified: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct normalized front camera tensor and LiDAR-assisted BEV projection.

        Args:
            cam_rgb: uint8 array of shape (H_cam, W_cam, 3) in RGB order (rectified).
            pts_radar: Optional (N, 3+) array of points in radar frame.
            is_rectified: bool indicating if cam_rgb is already rectified (default True).

        Returns:
            cam_img_norm: (3, H_cam, W_cam) float32 array.
            camera_bev: (3, H_bev, W_bev) float32 array.
            camera_bev_mask: (1, H_bev, W_bev) float32 array.
        """
        H_cam, W_cam = cam_rgb.shape[:2]
        H_bev, W_bev = self.bev_cfg.bev_height, self.bev_cfg.bev_width

        # 1. Front Camera Normalized Tensor (3, H_cam, W_cam)
        cam_img_norm = (cam_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

        camera_bev = np.zeros((3, H_bev, W_bev), dtype=np.float32)
        camera_bev_mask = np.zeros((1, H_bev, W_bev), dtype=np.float32)

        if pts_radar is None or len(pts_radar) == 0:
            return cam_img_norm, camera_bev, camera_bev_mask

        # 2. Project LiDAR points to Left Camera optical frame
        xyz_radar = pts_radar[:, :3]
        pts_cam = self.calib.radar_3d_to_cam_left(xyz_radar)
        if is_rectified and hasattr(self.calib, 'project_to_cam_left_rect'):
            uvs, in_front = self.calib.project_to_cam_left_rect(pts_cam)
        else:
            uvs, in_front = self.calib.project_to_cam_left(pts_cam)

        if not np.any(in_front) or len(uvs) == 0:
            return cam_img_norm, camera_bev, camera_bev_mask

        # Find points inside camera image boundaries
        u = uvs[:, 0]
        v = uvs[:, 1]
        in_image = (u >= 0) & (u < W_cam) & (v >= 0) & (v < H_cam)

        pts_valid_radar = pts_radar[in_front][in_image]
        u_valid = u[in_image].astype(np.int32)
        v_valid = v[in_image].astype(np.int32)

        if len(pts_valid_radar) == 0:
            return cam_img_norm, camera_bev, camera_bev_mask

        # 3. Sample RGB color from camera image at valid projected pixel coordinates
        sampled_rgb = (cam_rgb[v_valid, u_valid, :].astype(np.float32) / 255.0)

        # 4. Map points into the standardized BEV raster
        cfg = self.bev_cfg
        col_bev = ((pts_valid_radar[:, 0] - cfg.x_min) / cfg.dx).astype(np.int32)
        row_bev = ((cfg.y_max - pts_valid_radar[:, 1]) / cfg.dy).astype(np.int32)

        in_bev = (col_bev >= 0) & (col_bev < W_bev) & (row_bev >= 0) & (row_bev < H_bev)
        col_bev = col_bev[in_bev]
        row_bev = row_bev[in_bev]
        sampled_rgb = sampled_rgb[in_bev]

        if len(col_bev) == 0:
            return cam_img_norm, camera_bev, camera_bev_mask

        # Deterministic accumulation by unique BEV cell
        linear_idx = row_bev * W_bev + col_bev
        sort_order = np.argsort(linear_idx)
        sorted_idx = linear_idx[sort_order]
        sorted_rgb = sampled_rgb[sort_order]

        unique_cells, split_idx, counts = np.unique(
            sorted_idx, return_index=True, return_counts=True
        )

        u_rows = unique_cells // W_bev
        u_cols = unique_cells % W_bev

        # Average RGB per cell
        r_sum = np.add.reduceat(sorted_rgb[:, 0], split_idx)
        g_sum = np.add.reduceat(sorted_rgb[:, 1], split_idx)
        b_sum = np.add.reduceat(sorted_rgb[:, 2], split_idx)

        counts_f = counts.astype(np.float32)
        camera_bev[0, u_rows, u_cols] = r_sum / counts_f
        camera_bev[1, u_rows, u_cols] = g_sum / counts_f
        camera_bev[2, u_rows, u_cols] = b_sum / counts_f
        camera_bev_mask[0, u_rows, u_cols] = 1.0

        # 5. Optional morphological dilation to densify scanline gaps
        if self.dilation_kernel is not None:
            dilated_mask = cv2.dilate(camera_bev_mask[0], self.dilation_kernel)
            for c in range(3):
                camera_bev[c] = cv2.dilate(camera_bev[c], self.dilation_kernel)
            camera_bev_mask[0] = dilated_mask

        return cam_img_norm, camera_bev, camera_bev_mask
