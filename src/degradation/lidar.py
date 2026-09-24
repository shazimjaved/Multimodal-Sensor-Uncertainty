"""
src/degradation/lidar.py
------------------------
Physically-grounded LiDAR point cloud degradation using Beer-Lambert
atmospheric extinction, range cutoff, radial range jitter, intensity
attenuation, and near-field backscatter clutter.

Formula:
    P_drop(r) = 1 - exp(-alpha_ext * r)

Source of Truth:
    configs/experiment_protocol.yaml
"""

import os
from typing import Dict, Any, Optional, Tuple
import yaml
import numpy as np


class LiDARDegradation:
    """
    Applies atmospheric extinction and range degradation to raw LiDAR point clouds.
    Operates on raw points BEFORE BEV rasterization.
    """

    DEFAULT_LEVELS = {
        0: {"name": "Clean", "alpha_ext": 0.0, "max_range_m": 85.0, "range_jitter_std_m": 0.0, "clutter_count": 0},
        1: {"name": "Light Attenuation", "alpha_ext": 0.010, "max_range_m": 65.0, "range_jitter_std_m": 0.02, "clutter_count": 150},
        2: {"name": "Moderate Attenuation", "alpha_ext": 0.025, "max_range_m": 45.0, "range_jitter_std_m": 0.05, "clutter_count": 350},
        3: {"name": "Severe Fog Extinction", "alpha_ext": 0.050, "max_range_m": 25.0, "range_jitter_std_m": 0.08, "clutter_count": 600},
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LiDAR degradation engine with protocol parameters.
        """
        self.levels = self.DEFAULT_LEVELS.copy()

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            lidar_cfg = cfg.get("degradation", {}).get("lidar", {}).get("levels", {})
            for k, v in lidar_cfg.items():
                if k.startswith("level_"):
                    lvl_idx = int(k.split("_")[1])
                    self.levels[lvl_idx] = {
                        "name": v.get("name", f"Level {lvl_idx}"),
                        "alpha_ext": float(v.get("alpha_ext", 0.0)),
                        "max_range_m": float(v.get("max_range_m", 85.0)),
                        "range_jitter_std_m": float(v.get("range_jitter_std_m", 0.0)),
                        "clutter_count": int(v.get("clutter_count", lvl_idx * 150)),
                    }

    def degrade_points(
        self,
        raw_pts: np.ndarray,
        level: int = 0,
        seed: Optional[int] = 42
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Beer-Lambert extinction and range corruption to raw LiDAR point cloud.

        Args:
            raw_pts: (N, 5) array of points [x, y, z, intensity, ring] in LiDAR frame.
            level: Degradation level (0=Clean, 1=Light, 2=Moderate, 3=Severe)
            seed: Deterministic random seed.

        Returns:
            degraded_pts: (M, 5) array of retained + clutter points in same format.
            metadata: Dict with point count statistics and degradation parameters.
        """
        if level not in self.levels:
            raise ValueError(f"Unknown LiDAR degradation level: {level}. Valid levels: {list(self.levels.keys())}")

        orig_count = len(raw_pts)
        if orig_count == 0:
            return raw_pts.copy(), {
                "modality": "lidar", "level": level, "original_point_count": 0,
                "retained_point_count": 0, "retention_pct": 100.0, "max_retained_range": 0.0
            }

        params = self.levels[level]
        alpha = params["alpha_ext"]
        max_range = params["max_range_m"]
        jitter_std = params["range_jitter_std_m"]
        clutter_count = params.get("clutter_count", 0)

        # Level 0 is exact bitwise identity
        if level == 0:
            r_orig = np.linalg.norm(raw_pts[:, :3], axis=1)
            return raw_pts.copy(), {
                "modality": "lidar",
                "level": level,
                "level_name": params["name"],
                "alpha_ext": alpha,
                "max_range_m": max_range,
                "range_jitter_std_m": jitter_std,
                "original_point_count": orig_count,
                "retained_point_count": orig_count,
                "retention_pct": 100.0,
                "max_retained_range": float(r_orig.max()) if orig_count > 0 else 0.0,
                "clutter_points_added": 0,
            }

        # Initialize deterministic RNG
        rng = np.random.RandomState(seed)

        # 1. Radial distance from sensor origin
        r = np.linalg.norm(raw_pts[:, :3], axis=1)

        # 2. Maximum effective range cutoff
        mask_range = r <= max_range

        # 3. Beer-Lambert atmospheric extinction dropout
        # P_drop(r) = 1 - exp(-alpha * r)
        p_drop = 1.0 - np.exp(-alpha * r)
        u = rng.uniform(0.0, 1.0, size=orig_count)
        mask_survive = mask_range & (u >= p_drop)

        pts_retained = raw_pts[mask_survive].copy()
        retained_count = len(pts_retained)

        if retained_count > 0:
            r_retained = np.linalg.norm(pts_retained[:, :3], axis=1)

            # 4. Range measurement jitter along radial beam
            if jitter_std > 0:
                jitter = rng.normal(0.0, jitter_std, size=retained_count)
                scale = 1.0 + jitter / np.maximum(r_retained, 0.1)
                pts_retained[:, :3] *= scale[:, None]

            # 5. Two-way atmospheric intensity attenuation
            pts_retained[:, 3] *= np.exp(-2.0 * alpha * r_retained)

            # 6. Near-field backscatter clutter (water droplet reflection)
            if clutter_count > 0:
                # Droplets scatter close to sensor: range 1.0m to 6.0m
                clutter_r = rng.uniform(1.0, min(6.0, max_range), size=clutter_count)
                clutter_theta = rng.uniform(-np.pi, np.pi, size=clutter_count)
                clutter_phi = rng.uniform(-0.15, 0.15, size=clutter_count)  # vertical angle

                cx = clutter_r * np.cos(clutter_phi) * np.cos(clutter_theta)
                cy = clutter_r * np.cos(clutter_phi) * np.sin(clutter_theta)
                cz = clutter_r * np.sin(clutter_phi)
                c_intensity = rng.uniform(10.0, 35.0, size=clutter_count)
                c_ring = rng.randint(0, 32, size=clutter_count)

                clutter_pts = np.column_stack([cx, cy, cz, c_intensity, c_ring])
                pts_final = np.vstack([pts_retained, clutter_pts])
            else:
                pts_final = pts_retained
        else:
            pts_final = pts_retained

        final_count = len(pts_final)
        retention_pct = (retained_count / orig_count) * 100.0
        r_final = np.linalg.norm(pts_final[:, :3], axis=1) if final_count > 0 else np.array([0.0])

        metadata = {
            "modality": "lidar",
            "level": level,
            "level_name": params["name"],
            "alpha_ext": alpha,
            "max_range_m": max_range,
            "range_jitter_std_m": jitter_std,
            "original_point_count": orig_count,
            "retained_point_count": retained_count,
            "final_point_count": final_count,
            "retention_pct": float(retention_pct),
            "max_retained_range": float(r_final.max()),
            "clutter_points_added": clutter_count if retained_count > 0 else 0,
        }

        return pts_final, metadata
