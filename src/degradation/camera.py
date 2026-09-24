"""
src/degradation/camera.py
-------------------------
Physically-grounded optical degradation using Koschmieder atmospheric
scattering model with depth-dependent attenuation, forward scattering blur,
and contrast reduction.

Formula:
    I(x) = J_contrast(x) * exp(-beta * d(x)) + A * (1 - exp(-beta * d(x)))

Source of Truth:
    configs/experiment_protocol.yaml
"""

import os
from typing import Dict, Any, Optional, Tuple, Union
import yaml
import cv2
import numpy as np
import torch

from radiate_fusion import RadiateCalib


class CameraDegradation:
    """
    Applies depth-dependent atmospheric scattering degradation to camera images.
    """

    # Nominal visibility parameterization:
    # Koschmieder's law with standard 5% visual contrast threshold (C_T = 0.05, -ln(0.05) ~= 3.0):
    # V_5% = 3.0 / beta. (Under WMO 2% threshold, V_2% = 3.912 / beta).
    DEFAULT_LEVELS = {
        0: {"name": "Clean", "beta": 0.0, "blur_sigma": 0.0, "contrast_scale": 1.0, "visibility_m": 300.0},
        1: {"name": "Light Fog", "beta": 0.02, "blur_sigma": 1.0, "contrast_scale": 0.85, "visibility_m": 150.0},
        2: {"name": "Moderate Fog", "beta": 0.05, "blur_sigma": 2.0, "contrast_scale": 0.65, "visibility_m": 60.0},
        3: {"name": "Dense Fog", "beta": 0.10, "blur_sigma": 3.5, "contrast_scale": 0.45, "visibility_m": 30.0},
    }
    DEFAULT_AIRLIGHT = [0.82, 0.82, 0.84]  # Grayish daylight fog luminance

    def __init__(
        self,
        config_path: Optional[str] = None,
        calib: Optional[RadiateCalib] = None,
        calib_path: Optional[str] = None
    ):
        """
        Initialize camera degradation engine with protocol parameters.
        """
        self.levels = self.DEFAULT_LEVELS.copy()
        self.airlight = np.array(self.DEFAULT_AIRLIGHT, dtype=np.float32)

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            cam_cfg = cfg.get("degradation", {}).get("camera", {})
            if "airlight_A" in cam_cfg:
                self.airlight = np.array(cam_cfg["airlight_A"], dtype=np.float32)
            lvl_cfg = cam_cfg.get("levels", {})
            for k, v in lvl_cfg.items():
                if k.startswith("level_"):
                    lvl_idx = int(k.split("_")[1])
                    self.levels[lvl_idx] = {
                        "name": v.get("name", f"Level {lvl_idx}"),
                        "beta": float(v.get("beta", 0.0)),
                        "blur_sigma": float(v.get("blur_sigma", 0.0)),
                        "contrast_scale": float(v.get("contrast_scale", 1.0)),
                        "visibility_m": float(v.get("visibility_m", 300.0)),
                    }

        # Calibration
        if calib is not None:
            self.calib = calib
        elif calib_path and os.path.exists(calib_path):
            self.calib = RadiateCalib(calib_path)
        else:
            default_calib = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config", "default-calib.yaml"
            )
            if os.path.exists(default_calib):
                self.calib = RadiateCalib(default_calib)
            else:
                self.calib = None

    def compute_depth_map(
        self,
        pts_radar: np.ndarray,
        img_shape: Tuple[int, int] = (376, 672),
        use_rectified: bool = True
    ) -> np.ndarray:
        """
        Construct a dense depth map for the left camera image from projected LiDAR points.

        Args:
            pts_radar: (N, 3+) array of points in radar frame.
            img_shape: (H, W) of the camera image.
            use_rectified: Whether to project onto rectified image plane (default True).

        Returns:
            dense_depth: (H, W) float32 array of per-pixel depth in meters in [1.0, 150.0].
        """
        if self.calib is None:
            raise ValueError("RadiateCalib is required to compute depth map from LiDAR points.")

        H, W = img_shape
        pts_cam = self.calib.radar_3d_to_cam_left(pts_radar[:, :3])
        if use_rectified and hasattr(self.calib, 'project_to_cam_left_rect'):
            uvs, mask = self.calib.project_to_cam_left_rect(pts_cam)
        else:
            uvs, mask = self.calib.project_to_cam_left(pts_cam)
        depths = pts_cam[mask, 2]

        # Filter finite coordinates and valid depth
        finite = np.isfinite(uvs[:, 0]) & np.isfinite(uvs[:, 1]) & (depths > 0.5)
        uvs_f = uvs[finite]
        depths_f = depths[finite]

        u = np.round(uvs_f[:, 0]).astype(int)
        v = np.round(uvs_f[:, 1]).astype(int)
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        u_v = u[valid]
        v_v = v[valid]
        d_v = depths_f[valid]

        sparse_depth = np.zeros((H, W), dtype=np.float32)
        if len(d_v) > 0:
            order = np.argsort(-d_v)  # nearest overwrites farther
            sparse_depth[v_v[order], u_v[order]] = d_v[order]

        # Morphological closing (wide horizontal kernel to connect LiDAR scanlines)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
        valid_mask = (sparse_depth > 0).astype(np.uint8)
        dilated = cv2.dilate(sparse_depth, k_close)
        dil_mask = cv2.dilate(valid_mask, k_close)

        dense_depth = np.where(dil_mask > 0, dilated, 0.0)

        # Sky region: pixels above the highest projected point are set to optical horizon (150m)
        min_v = int(np.min(v_v)) if len(v_v) > 0 else int(H * 0.45)
        dense_depth[:min_v, :] = 150.0

        # Fill remaining zeros (e.g. road foreground) using nearest-neighbor / inpainting
        invalid_mask = (dense_depth == 0).astype(np.uint8)
        if np.any(invalid_mask):
            dense_depth = cv2.inpaint(dense_depth, invalid_mask, 5, cv2.INPAINT_TELEA)

        dense_depth = np.clip(dense_depth, 1.0, 150.0)
        return dense_depth

    def degrade_image(
        self,
        img: Union[np.ndarray, torch.Tensor],
        depth_map: np.ndarray,
        level: int = 0,
        seed: Optional[int] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Apply Koschmieder optical degradation to an image using per-pixel depth.

        Args:
            img: (3, H, W) float32 in [0, 1] (or torch.Tensor)
            depth_map: (H, W) float32 in meters
            level: Degradation level (0=Clean, 1=Light, 2=Moderate, 3=Dense)
            seed: Random seed (unused for deterministic Koschmieder, reserved for noise extensions)

        Returns:
            degraded_img: Image of same type and shape as input, float32 in [0, 1].
            metadata: Dict with degradation parameters and transmission statistics.
        """
        if level not in self.levels:
            raise ValueError(f"Unknown camera degradation level: {level}. Valid levels: {list(self.levels.keys())}")

        is_torch = isinstance(img, torch.Tensor)
        if is_torch:
            img_np = img.detach().cpu().numpy()
        else:
            img_np = img.copy()

        # Handle channel order: expect (3, H, W)
        if img_np.ndim != 3 or img_np.shape[0] != 3:
            raise ValueError(f"Expected image of shape (3, H, W), got {img_np.shape}")

        params = self.levels[level]
        beta = params["beta"]
        blur_sigma = params["blur_sigma"]
        contrast_scale = params["contrast_scale"]

        metadata = {
            "modality": "camera",
            "level": level,
            "level_name": params["name"],
            "beta": beta,
            "blur_sigma": blur_sigma,
            "contrast_scale": contrast_scale,
            "visibility_m": params["visibility_m"],
            "airlight_A": self.airlight.tolist(),
        }

        # Level 0 is bitwise identity
        if level == 0:
            metadata.update({
                "mean_transmission": 1.0,
                "min_transmission": 1.0,
                "max_transmission": 1.0,
            })
            return img.clone() if is_torch else img_np, metadata

        # 1. Contrast reduction: J_contrast = clip(0.5 + contrast_scale * (J - 0.5), 0, 1)
        c_img = np.clip(0.5 + contrast_scale * (img_np - 0.5), 0.0, 1.0)

        # 2. Transmission map: t(x) = exp(-beta * d(x))
        transmission = np.exp(-beta * depth_map).astype(np.float32)  # (H, W)

        # 3. Koschmieder atmospheric scattering: I(x) = J_contrast(x) * t(x) + A * (1 - t(x))
        A = self.airlight[:, None, None]  # (3, 1, 1)
        t_3ch = transmission[None, :, :]  # (1, H, W)
        scat_img = c_img * t_3ch + A * (1.0 - t_3ch)

        # 4. Forward-scattering blur (Gaussian blur)
        if blur_sigma > 0:
            ksize = int(2 * np.ceil(3 * blur_sigma) + 1)
            # Apply blur to each channel
            blurred_channels = []
            for ch in range(3):
                blurred_ch = cv2.GaussianBlur(scat_img[ch], (ksize, ksize), blur_sigma)
                blurred_channels.append(blurred_ch)
            scat_img = np.stack(blurred_channels, axis=0)

        degraded = np.clip(scat_img, 0.0, 1.0).astype(np.float32)

        metadata.update({
            "mean_transmission": float(transmission.mean()),
            "min_transmission": float(transmission.min()),
            "max_transmission": float(transmission.max()),
            "mean_intensity": float(degraded.mean()),
        })

        if is_torch:
            return torch.from_numpy(degraded).to(dtype=img.dtype, device=img.device), metadata
        return degraded, metadata
