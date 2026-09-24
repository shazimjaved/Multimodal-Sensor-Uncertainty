"""
src/degradation/radar.py
------------------------
Controlled radar corruption model for sensor stress testing and fusion
uncertainty evaluation (radome wetting, receiver noise, backscatter clutter).

Formula:
    S_deg = S_clean * 10^(-loss_db / 10) * R(sigma_s) + clutter

Note:
    Navtech radar PNG values represent quantized received power.
    Power loss is thus applied as a power ratio: 10^(-loss_db / 10).
    This model represents controlled radar corruption to stress-test fusion.
    It is NOT claimed as a validated physical model of atmospheric fog-induced
    radar attenuation, as 77GHz millimeter waves penetrate fog with negligible loss.

Source of Truth:
    configs/experiment_protocol.yaml
"""

import os
from typing import Dict, Any, Optional, Tuple, Union
import yaml
import numpy as np
import torch


class RadarDegradation:
    """
    Applies power attenuation, multiplicative speckle noise, and additive
    clutter spikes to radar BEV representations.
    """

    DEFAULT_LEVELS = {
        0: {"name": "Clean", "power_loss_db": 0.0, "speckle_sigma": 0.0, "clutter_density": 0.0},
        1: {"name": "Mild Clutter / Rain Spray", "power_loss_db": 1.5, "speckle_sigma": 0.15, "clutter_density": 0.001},
        2: {"name": "Moderate Radome Attenuation", "power_loss_db": 3.5, "speckle_sigma": 0.30, "clutter_density": 0.005},
        3: {"name": "Severe Hardware Degradation", "power_loss_db": 6.0, "speckle_sigma": 0.50, "clutter_density": 0.015},
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize radar degradation engine with protocol parameters.
        """
        self.levels = self.DEFAULT_LEVELS.copy()

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            radar_cfg = cfg.get("degradation", {}).get("radar", {}).get("levels", {})
            for k, v in radar_cfg.items():
                if k.startswith("level_"):
                    lvl_idx = int(k.split("_")[1])
                    self.levels[lvl_idx] = {
                        "name": v.get("name", f"Level {lvl_idx}"),
                        "power_loss_db": float(v.get("power_loss_db", 0.0)),
                        "speckle_sigma": float(v.get("speckle_sigma", 0.0)),
                        "clutter_density": float(v.get("clutter_density", 0.0)),
                    }

    def corrupt_raster(
        self,
        radar_bev: Union[np.ndarray, torch.Tensor],
        level: int = 0,
        seed: Optional[int] = 42
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, Any]]:
        """
        Apply controlled power loss, speckle fading, and clutter spikes to radar BEV.

        Args:
            radar_bev: (1, H, W) float32 in [0, 1] (numpy or torch.Tensor)
            level: Degradation level (0=Clean, 1=Mild, 2=Moderate, 3=Severe)
            seed: Deterministic random seed.

        Returns:
            degraded_radar: Tensor of same shape and type, clipped to [0, 1].
            metadata: Dict comparing clean vs degraded signal statistics.
        """
        if level not in self.levels:
            raise ValueError(f"Unknown radar degradation level: {level}. Valid levels: {list(self.levels.keys())}")

        is_torch = isinstance(radar_bev, torch.Tensor)
        if is_torch:
            r_np = radar_bev.detach().cpu().numpy()
        else:
            r_np = radar_bev.copy()

        if r_np.ndim != 3 or r_np.shape[0] != 1:
            raise ValueError(f"Expected radar tensor of shape (1, H, W), got {r_np.shape}")

        params = self.levels[level]
        loss_db = params["power_loss_db"]
        speckle_sigma = params["speckle_sigma"]
        clutter_density = params["clutter_density"]

        mean_clean = float(r_np.mean())
        std_clean = float(r_np.std())
        max_clean = float(r_np.max())

        metadata = {
            "modality": "radar",
            "level": level,
            "level_name": params["name"],
            "power_loss_db": loss_db,
            "speckle_sigma": speckle_sigma,
            "clutter_density": clutter_density,
            "mean_clean": mean_clean,
            "std_clean": std_clean,
            "max_clean": max_clean,
            "physical_model_note": "Controlled corruption for fusion stress-testing; not a model of fog propagation."
        }

        # Level 0 is bitwise identity
        if level == 0:
            metadata.update({
                "mean_degraded": mean_clean,
                "std_degraded": std_clean,
                "max_degraded": max_clean,
                "relative_power_ratio": 1.0,
            })
            return radar_bev.clone() if is_torch else r_np, metadata

        # Deterministic RNG
        rng = np.random.RandomState(seed)

        # 1. Attenuation factor from power loss in dB
        # Linear power ratio for received power = 10^(-loss_db / 10)
        att_factor = float(10.0 ** (-loss_db / 10.0))

        # 2. Multiplicative Rayleigh speckle fading centered around 1.0
        if speckle_sigma > 0:
            # Standard Rayleigh has mean sqrt(pi / 2) ~ 1.2533
            ray = rng.rayleigh(scale=1.0, size=r_np.shape).astype(np.float32)
            ray_centered = ray - float(np.sqrt(np.pi / 2.0))
            speckle = np.maximum(0.0, 1.0 + speckle_sigma * ray_centered)
        else:
            speckle = 1.0

        degraded = r_np * att_factor * speckle

        # 3. Additive clutter spikes (radome reflections, false target returns)
        if clutter_density > 0:
            clutter_mask = rng.uniform(0.0, 1.0, size=r_np.shape) < clutter_density
            clutter_vals = rng.uniform(0.15, 0.85, size=r_np.shape).astype(np.float32)
            degraded = np.where(clutter_mask, np.maximum(degraded, clutter_vals), degraded)

        # 4. Consistent normalization clipped to [0, 1]
        degraded = np.clip(degraded, 0.0, 1.0).astype(np.float32)

        mean_deg = float(degraded.mean())
        std_deg = float(degraded.std())
        max_deg = float(degraded.max())

        metadata.update({
            "mean_degraded": mean_deg,
            "std_degraded": std_deg,
            "max_degraded": max_deg,
            "relative_power_ratio": float(mean_deg / max(mean_clean, 1e-8)),
        })

        if is_torch:
            return torch.from_numpy(degraded).to(dtype=radar_bev.dtype, device=radar_bev.device), metadata
        return degraded, metadata
