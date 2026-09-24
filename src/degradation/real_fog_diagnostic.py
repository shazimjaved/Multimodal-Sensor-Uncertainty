"""
src/degradation/real_fog_diagnostic.py
--------------------------------------
Non-learning physical diagnostic comparing synthetic degradation statistics
(from city_3_0 under C4/L1-L3) against real fog statistics from held-out fog_6_0.

IMPORTANT:
    This is an observational diagnostic tool only.
    - Do NOT tune synthetic degradation parameters using fog_6_0 test labels.
    - Do NOT use fog_6_0 to optimize or train models.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from radiate_fusion import RadiateCalib, load_lidar_csv


def compute_camera_diagnostics(img_rgb_norm: np.ndarray) -> Dict[str, float]:
    """
    Compute non-learning photometric diagnostics for a camera image.

    Args:
        img_rgb_norm: (3, H, W) float32 in [0, 1] RGB.

    Returns:
        Dict containing contrast, sharpness (Laplacian variance), mean luminance,
        and dynamic range.
    """
    # Convert to grayscale luminance: Y = 0.299*R + 0.587*G + 0.114*B
    gray = 0.299 * img_rgb_norm[0] + 0.587 * img_rgb_norm[1] + 0.114 * img_rgb_norm[2]

    # 1. RMS Contrast (standard deviation of luminance)
    rms_contrast = float(np.std(gray))

    # 2. Perceptual sharpness via Laplacian variance (high-frequency edge energy)
    # Scaled to 0-255 uint8 equivalent for standard interpretation
    gray_255 = (gray * 255.0).astype(np.float32)
    lap = cv2.Laplacian(gray_255, cv2.CV_32F)
    lap_var = float(np.var(lap))

    # 3. Mean luminance / haze floor
    mean_lum = float(np.mean(gray))

    # 4. Dynamic range (5th to 95th percentile)
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    dyn_range = p95 - p5

    return {
        "rms_contrast": rms_contrast,
        "laplacian_variance": lap_var,
        "mean_luminance": mean_lum,
        "dynamic_range": dyn_range,
    }


def compute_lidar_diagnostics(pts: np.ndarray) -> Dict[str, float]:
    """
    Compute non-learning geometric diagnostics for a LiDAR point cloud.

    Args:
        pts: (N, 3+) array of LiDAR points [x, y, z, ...].

    Returns:
        Dict containing point count, effective 95th percentile range, max range,
        and mean intensity.
    """
    count = len(pts)
    if count == 0:
        return {
            "point_count": 0, "p95_range_m": 0.0, "max_range_m": 0.0,
            "mean_intensity": 0.0, "near_field_pct": 0.0
        }

    r = np.linalg.norm(pts[:, :3], axis=1)
    p95_r = float(np.percentile(r, 95))
    max_r = float(np.max(r))
    mean_int = float(np.mean(pts[:, 3])) if pts.shape[1] > 3 else 0.0

    # Near-field backscatter concentration (r <= 6m)
    near_pct = float(np.mean(r <= 6.0) * 100.0)

    return {
        "point_count": count,
        "p95_range_m": p95_r,
        "max_range_m": max_r,
        "mean_intensity": mean_int,
        "near_field_pct": near_pct,
    }


def compare_synthetic_vs_real_fog(
    clean_sample: Dict[str, Any],
    degraded_samples_c4: Dict[int, Dict[str, Any]],
    fog_sample: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare camera and LiDAR diagnostics across:
    - Clean reference (city_3_0)
    - Synthetic fog C4 at levels 1, 2, 3
    - Real fog (fog_6_0)

    Args:
        clean_sample: Clean city_3_0 sample.
        degraded_samples_c4: Dict mapping level (1, 2, 3) to degraded sample.
        fog_sample: Real fog_6_0 sample.

    Returns:
        Diagnostic comparison report.
    """
    report = {
        "camera_comparison": {},
        "lidar_comparison": {}
    }

    # Camera diagnostics
    if "camera_image" in clean_sample and clean_sample["camera_image"] is not None:
        c_clean = clean_sample["camera_image"].detach().cpu().numpy()
        report["camera_comparison"]["clean_city3_0"] = compute_camera_diagnostics(c_clean)

        for lvl, d_samp in degraded_samples_c4.items():
            c_deg = d_samp["camera_image"].detach().cpu().numpy()
            report["camera_comparison"][f"synthetic_c4_lvl{lvl}"] = compute_camera_diagnostics(c_deg)

        if "camera_image" in fog_sample and fog_sample["camera_image"] is not None:
            c_fog = fog_sample["camera_image"].detach().cpu().numpy()
            report["camera_comparison"]["real_fog6_0"] = compute_camera_diagnostics(c_fog)

    # LiDAR diagnostics
    calib = RadiateCalib("config/default-calib.yaml")
    clean_pts_path = clean_sample.get("metadata", {}).get("lidar_path", None)
    fog_pts_path = fog_sample.get("metadata", {}).get("lidar_path", None)

    if clean_pts_path and os.path.exists(clean_pts_path):
        pts_clean = load_lidar_csv(clean_pts_path)
        report["lidar_comparison"]["clean_city3_0"] = compute_lidar_diagnostics(pts_clean)

        for lvl, d_samp in degraded_samples_c4.items():
            lidar_stats = d_samp.get("degradation_metadata", {}).get("sensor_statistics", {}).get("lidar", {})
            report["lidar_comparison"][f"synthetic_c4_lvl{lvl}"] = {
                "point_count": lidar_stats.get("final_point_count", 0),
                "retention_pct": lidar_stats.get("retention_pct", 0.0),
                "max_range_m": lidar_stats.get("max_retained_range", 0.0),
            }

    if fog_pts_path and os.path.exists(fog_pts_path):
        pts_fog = load_lidar_csv(fog_pts_path)
        report["lidar_comparison"]["real_fog6_0"] = compute_lidar_diagnostics(pts_fog)

    return report
