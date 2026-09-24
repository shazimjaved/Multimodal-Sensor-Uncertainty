"""
src/degradation/conditions.py
-----------------------------
Condition API and unified degradation engine implementing experimental
conditions C0 through C5 and branch ablation C2a.

Conditions:
    - C0: Clean baseline (all modalities L0)
    - C1: Camera optical corruption with clean LiDAR geometry (L1-L3), LiDAR L0, Radar L0
    - C2: Raw LiDAR degradation (L1-L3), which naturally propagates into the LiDAR-assisted camera representation
    - C2a: LiDAR branch ablation (lidar_bev zeroed at model input stage; architectural ablation, NOT sensor degradation)
    - C3: Independent radar corruption (L1-L3), Camera L0, LiDAR L0
    - C4: Fog-matched synthetic degradation (Camera + LiDAR at matched severity, Radar L0)
    - C5: Catastrophic all-modality degradation (Camera + LiDAR + Radar at matched severity)

Terminology Note:
    The camera BEV representation is 'LiDAR-assisted camera RGB projection into common BEV'
    rather than canonical PointPainting (no semantic segmentation scores are painted).

Source of Truth:
    configs/experiment_protocol.yaml
"""

import os
from typing import Dict, Any, Optional, Tuple, Union
import copy
import re
import numpy as np
import torch

from radiate_fusion import RadiateCalib, load_lidar_csv
from src.preprocessing.bev_grid import BEVGridConfig
from src.preprocessing.lidar import LiDARPreprocessor
from src.preprocessing.camera import CameraPreprocessor
from src.preprocessing.radar import RadarPreprocessor
from .camera import CameraDegradation
from .lidar import LiDARDegradation
from .radar import RadarDegradation


class DegradationEngine:
    """
    Unified multimodal degradation engine managing sensor transforms
    and condition dispatch.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        calib_path: Optional[str] = None,
        bev_cfg: Optional[BEVGridConfig] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.config_path = config_path or os.path.join(base_dir, "configs", "experiment_protocol.yaml")
        self.calib_path = calib_path or os.path.join(base_dir, "config", "default-calib.yaml")
        self.bev_cfg = bev_cfg or BEVGridConfig()

        self.calib = RadiateCalib(self.calib_path)
        self.cam_deg = CameraDegradation(config_path=self.config_path, calib=self.calib)
        self.lidar_deg = LiDARDegradation(config_path=self.config_path)
        self.radar_deg = RadarDegradation(config_path=self.config_path)

        # Preprocessors for re-rasterization
        self.lidar_proc = LiDARPreprocessor(calib=self.calib, bev_cfg=self.bev_cfg)
        self.cam_proc = CameraPreprocessor(calib=self.calib, bev_cfg=self.bev_cfg)
        self.radar_proc = RadarPreprocessor(bev_cfg=self.bev_cfg)

    def parse_condition(
        self,
        condition: str,
        severity: Optional[int] = None
    ) -> Tuple[str, int]:
        """
        Parse condition identifier and severity level.

        Supports formats like:
            - ("C0", None) -> ("C0", 0)
            - ("C1", 2) -> ("C1", 2)
            - ("C1_L2", None) -> ("C1", 2)
            - ("C1_2", None) -> ("C1", 2)
            - ("C2a", None) -> ("C2a", 0)
        """
        cond = condition.strip().upper()

        if cond in ("C0", "C0_CLEAN_BASELINE"):
            return "C0", 0

        if cond in ("C2A", "C2_A", "C2A_ABLATION"):
            return "C2a", 0

        m = re.match(r"^(C[1-5])(?:_L?([0-3]))?$", cond)
        if m:
            base_cond = m.group(1)
            level_str = m.group(2)
            if level_str is not None:
                lvl = int(level_str)
            elif severity is not None:
                lvl = int(severity)
            else:
                lvl = 1  # default to level 1 if unspecified
            if lvl not in (0, 1, 2, 3):
                raise ValueError(f"Severity level must be 0, 1, 2, or 3, got {lvl}")
            return base_cond, lvl

        raise ValueError(
            f"Unrecognized condition format: '{condition}'. "
            "Expected C0, C1, C2, C2a, C3, C4, C5 (optionally suffixed with _L1, _L2, _L3)."
        )

    def degrade(
        self,
        sample: Dict[str, Any],
        condition: str,
        severity: Optional[int] = None,
        seed: Optional[int] = 42,
        decouple_camera_lidar: bool = False
    ) -> Dict[str, Any]:
        """
        Apply specified multimodal degradation condition to a sample.

        Args:
            sample: Sample dictionary from RadiateMultimodalDataset.
            condition: Condition identifier (C0, C1, C2, C2a, C3, C4, C5).
            severity: Severity level (1, 2, 3), optional if encoded in condition string.
            seed: Deterministic random seed.
            decouple_camera_lidar: If True in C2, preserves clean LiDAR geometry for
                camera projection (Strategy A / branch ablation style).
                If False (default), reflects physical cascading degradation (Strategy D).

        Returns:
            degraded_sample: New dictionary containing degraded sensor rasters
                and full degradation metadata. Original sample is unchanged.
        """
        base_cond, level = self.parse_condition(condition, severity)

        # Create output sample (shallow copy dict, copy metadata)
        degraded = dict(sample)
        meta = copy.deepcopy(sample.get("metadata", {}))
        degraded["metadata"] = meta

        deg_meta = {
            "condition": base_cond,
            "condition_requested": condition,
            "severity_level": level,
            "seed": seed,
            "decouple_camera_lidar": decouple_camera_lidar,
            "modalities_degraded": [],
            "sensor_statistics": {},
        }

        # -------------------------------------------------------------
        # Condition C2a: LiDAR branch ablation (Feature-level zeroing)
        # -------------------------------------------------------------
        if base_cond == "C2a":
            degraded["lidar_bev"] = torch.zeros_like(sample["lidar_bev"])
            deg_meta["description"] = "LiDAR branch ablation: lidar_bev zeroed at model input stage."
            deg_meta["is_ablation"] = True
            degraded["degradation_metadata"] = deg_meta
            return degraded

        # -------------------------------------------------------------
        # Condition C0: Clean baseline
        # -------------------------------------------------------------
        if base_cond == "C0" or level == 0:
            deg_meta["description"] = "Pristine clean baseline (L0)."
            degraded["degradation_metadata"] = deg_meta
            return degraded

        # Determine active modality degradation levels
        cam_lvl = level if base_cond in ("C1", "C4", "C5") else 0
        lidar_lvl = level if base_cond in ("C2", "C4", "C5") else 0
        radar_lvl = level if base_cond in ("C3", "C5") else 0

        # Retrieve raw points for LiDAR/Camera operations
        pts_radar_clean = meta.get("pts_radar_filtered", None)
        lidar_path = meta.get("lidar_path", None)

        if pts_radar_clean is None and lidar_path and os.path.exists(lidar_path):
            raw_lidar_pts = load_lidar_csv(lidar_path)
            _, pts_radar_clean = self.lidar_proc.process_points(raw_lidar_pts)
        elif pts_radar_clean is None:
            # Fallback empty points if not available
            pts_radar_clean = np.zeros((0, 5), dtype=np.float32)

        # -------------------------------------------------------------
        # 1. LiDAR Degradation (C2, C4, C5)
        # -------------------------------------------------------------
        pts_radar_for_camera = pts_radar_clean
        if lidar_lvl > 0:
            if lidar_path and os.path.exists(lidar_path):
                raw_lidar_pts = load_lidar_csv(lidar_path)
            else:
                raw_lidar_pts = np.zeros((0, 5), dtype=np.float32)

            deg_pts_lidar, lidar_stats = self.lidar_deg.degrade_points(
                raw_lidar_pts, level=lidar_lvl, seed=seed
            )
            new_lidar_bev, pts_radar_deg = self.lidar_proc.process_points(deg_pts_lidar)

            degraded["lidar_bev"] = torch.from_numpy(new_lidar_bev).to(
                dtype=sample["lidar_bev"].dtype, device=sample["lidar_bev"].device
            )
            deg_meta["modalities_degraded"].append("lidar")
            deg_meta["sensor_statistics"]["lidar"] = lidar_stats

            if not decouple_camera_lidar:
                pts_radar_for_camera = pts_radar_deg

        # -------------------------------------------------------------
        # 2. Camera Degradation (C1, C4, C5)
        # -------------------------------------------------------------
        if cam_lvl > 0:
            cam_img = sample["camera_image"]
            if cam_img is None:
                raise ValueError("Camera degradation requires camera_image in sample.")

            # Compute depth map using clean LiDAR geometry (Koschmieder transmission)
            depth_map = self.cam_deg.compute_depth_map(pts_radar_clean)
            deg_cam_img, cam_stats = self.cam_deg.degrade_image(
                cam_img, depth_map=depth_map, level=cam_lvl, seed=seed
            )
            degraded["camera_image"] = deg_cam_img
            deg_meta["modalities_degraded"].append("camera")
            deg_meta["sensor_statistics"]["camera"] = cam_stats

            # Re-project camera BEV feature map using degraded camera image
            # cam_img is (3, H, W) in [0, 1] float32 -> uint8 RGB (H, W, 3)
            cam_np = deg_cam_img.detach().cpu().numpy() if isinstance(deg_cam_img, torch.Tensor) else deg_cam_img
            cam_rgb_uint8 = np.clip(cam_np * 255.0, 0.0, 255.0).transpose(1, 2, 0).astype(np.uint8)

            _, new_cam_bev, new_cam_mask = self.cam_proc.process_image(
                cam_rgb_uint8, pts_radar=pts_radar_for_camera
            )
            degraded["camera_bev"] = torch.from_numpy(new_cam_bev).to(
                dtype=sample["camera_bev"].dtype, device=sample["camera_bev"].device
            )
            degraded["camera_bev_mask"] = torch.from_numpy(new_cam_mask).to(
                dtype=sample["camera_bev_mask"].dtype, device=sample["camera_bev_mask"].device
            )
        elif lidar_lvl > 0 and not decouple_camera_lidar:
            # Camera optics are clean, but LiDAR points used for LiDAR-assisted camera projection into BEV were degraded
            cam_img = sample["camera_image"]
            if cam_img is not None:
                cam_np = cam_img.detach().cpu().numpy() if isinstance(cam_img, torch.Tensor) else cam_img
                cam_rgb_uint8 = np.clip(cam_np * 255.0, 0.0, 255.0).transpose(1, 2, 0).astype(np.uint8)
                _, new_cam_bev, new_cam_mask = self.cam_proc.process_image(
                    cam_rgb_uint8, pts_radar=pts_radar_for_camera
                )
                degraded["camera_bev"] = torch.from_numpy(new_cam_bev).to(
                    dtype=sample["camera_bev"].dtype, device=sample["camera_bev"].device
                )
                degraded["camera_bev_mask"] = torch.from_numpy(new_cam_mask).to(
                    dtype=sample["camera_bev_mask"].dtype, device=sample["camera_bev_mask"].device
                )

        # -------------------------------------------------------------
        # 3. Radar Degradation (C3, C5)
        # -------------------------------------------------------------
        if radar_lvl > 0:
            deg_radar, radar_stats = self.radar_deg.corrupt_raster(
                sample["radar_bev"], level=radar_lvl, seed=seed
            )
            degraded["radar_bev"] = deg_radar
            deg_meta["modalities_degraded"].append("radar")
            deg_meta["sensor_statistics"]["radar"] = radar_stats

        degraded["degradation_metadata"] = deg_meta
        return degraded


# Global default engine instance for convenience
_DEFAULT_ENGINE: Optional[DegradationEngine] = None


def get_default_engine() -> DegradationEngine:
    """Return singleton instance of DegradationEngine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = DegradationEngine()
    return _DEFAULT_ENGINE


def degrade_sample(
    sample: Dict[str, Any],
    condition: str,
    severity: Optional[int] = None,
    seed: Optional[int] = 42,
    engine: Optional[DegradationEngine] = None,
    decouple_camera_lidar: bool = False
) -> Dict[str, Any]:
    """
    Top-level functional API for deterministic multimodal degradation.

    Args:
        sample: Input sample dictionary from RadiateMultimodalDataset.
        condition: Condition identifier (C0, C1, C2, C2a, C3, C4, C5).
        severity: Severity level (1, 2, 3), optional if encoded in condition string.
        seed: Random seed for stochastic components (speckle, dropout, clutter).
        engine: Optional custom DegradationEngine instance.
        decouple_camera_lidar: If True, preserves clean LiDAR geometry for camera BEV.

    Returns:
        New degraded sample dictionary with updated sensor tensors.
    """
    eng = engine or get_default_engine()
    return eng.degrade(
        sample=sample,
        condition=condition,
        severity=severity,
        seed=seed,
        decouple_camera_lidar=decouple_camera_lidar
    )
