"""
src/preprocessing/lidar.py
--------------------------
Deterministic preprocessing for Velodyne LiDAR point clouds into
multi-channel Bird's-Eye View (BEV) raster representations.
"""

import os
from typing import Optional, Tuple
import numpy as np

from .bev_grid import BEVGridConfig
from radiate_fusion import RadiateCalib, load_lidar_csv


class LiDARPreprocessor:
    """Preprocesses raw Velodyne LiDAR CSV point clouds into 3-channel BEV rasters."""

    def __init__(
        self,
        calib: RadiateCalib,
        bev_cfg: Optional[BEVGridConfig] = None,
        max_density_points: int = 16
    ):
        self.calib = calib
        self.bev_cfg = bev_cfg or BEVGridConfig()
        self.max_density_points = max_density_points
        self.log_max_density = np.log(1.0 + float(self.max_density_points))

    def process_file(self, lidar_csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a LiDAR CSV, transform points into radar frame, and rasterize to BEV.

        Args:
            lidar_csv_path: Path to velo_lidar CSV file.

        Returns:
            bev_tensor: Array of shape (3, H_bev, W_bev), float32:
                - Channel 0: Normalized maximum height in cell [0, 1]
                - Channel 1: Normalized point density (log-scaled) [0, 1]
                - Channel 2: Mean intensity in cell [0, 1]
            pts_radar_filtered: Array of shape (M, 5) [x, y, z, intensity, ring] in radar frame.
        """
        if not os.path.exists(lidar_csv_path):
            raise FileNotFoundError(f"LiDAR file not found: {lidar_csv_path}")

        raw_pts = load_lidar_csv(lidar_csv_path)
        return self.process_points(raw_pts)

    def process_points(self, raw_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform point cloud to radar frame and rasterize to standardized BEV grid.

        Args:
            raw_pts: Array of shape (N, 3+) [x, y, z, intensity, ring] in LiDAR frame.

        Returns:
            bev_tensor: (3, H_bev, W_bev) float32 array.
            pts_radar_filtered: (M, 5) float32 array in radar frame.
        """
        H, W = self.bev_cfg.bev_height, self.bev_cfg.bev_width
        bev_tensor = np.zeros((3, H, W), dtype=np.float32)

        if len(raw_pts) == 0:
            return bev_tensor, np.zeros((0, 5), dtype=np.float32)

        # 1. Transform points from LiDAR frame to Radar reference frame
        # P_radar = R_lidar @ P_lidar + T_lidar
        xyz_lidar = raw_pts[:, :3]
        xyz_radar = self.calib.lidar_to_radar(xyz_lidar)

        intensity = raw_pts[:, 3:4] if raw_pts.shape[1] > 3 else np.zeros((len(raw_pts), 1))
        ring = raw_pts[:, 4:5] if raw_pts.shape[1] > 4 else np.zeros((len(raw_pts), 1))
        pts_radar = np.hstack([xyz_radar, intensity, ring])

        # 2. Filter within configured spatial bounds
        cfg = self.bev_cfg
        in_bounds = (
            (pts_radar[:, 0] >= cfg.x_min) & (pts_radar[:, 0] < cfg.x_max) &
            (pts_radar[:, 1] >= cfg.y_min) & (pts_radar[:, 1] < cfg.y_max) &
            (pts_radar[:, 2] >= cfg.z_min) & (pts_radar[:, 2] <= cfg.z_max)
        )
        pts_filt = pts_radar[in_bounds]

        if len(pts_filt) == 0:
            return bev_tensor, pts_filt

        # 3. Compute 2D pixel coordinates in BEV grid
        cols = ((pts_filt[:, 0] - cfg.x_min) / cfg.dx).astype(np.int32)
        rows = ((cfg.y_max - pts_filt[:, 1]) / cfg.dy).astype(np.int32)

        cols = np.clip(cols, 0, W - 1)
        rows = np.clip(rows, 0, H - 1)

        # 4. Deterministic rasterization via 1D linear binning
        linear_indices = rows * W + cols
        sort_order = np.argsort(linear_indices)
        sorted_indices = linear_indices[sort_order]
        sorted_pts = pts_filt[sort_order]

        # Find unique cell boundaries
        unique_cells, split_idx, counts = np.unique(
            sorted_indices, return_index=True, return_counts=True
        )

        u_rows = unique_cells // W
        u_cols = unique_cells % W

        # Channel 0: Normalized max height
        # Channel 1: Normalized log point density
        # Channel 2: Mean intensity
        z_norm = (sorted_pts[:, 2] - cfg.z_min) / (cfg.z_max - cfg.z_min)
        z_norm = np.clip(z_norm, 0.0, 1.0)
        intensities = np.clip(sorted_pts[:, 3] / 255.0, 0.0, 1.0)

        # Vectorized aggregation using np.maximum.reduceat and np.add.reduceat
        max_heights = np.maximum.reduceat(z_norm, split_idx)
        sum_intensities = np.add.reduceat(intensities, split_idx)
        mean_intensities = sum_intensities / counts.astype(np.float32)

        density_norm = np.clip(np.log(1.0 + counts.astype(np.float32)) / self.log_max_density, 0.0, 1.0)

        bev_tensor[0, u_rows, u_cols] = max_heights
        bev_tensor[1, u_rows, u_cols] = density_norm
        bev_tensor[2, u_rows, u_cols] = mean_intensities

        return bev_tensor, pts_filt
