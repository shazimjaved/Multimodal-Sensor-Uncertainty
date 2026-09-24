"""
src/models/multimodal_detector.py
---------------------------------
Baseline BEV Multimodal Fusion Detector.
Combines:
- 3 ResNet-18 FPN backbones (Radar 1ch, LiDAR 3ch, Camera-assisted 3ch)
- Channel-wise concatenation + 1x1 Conv fusion neck
- CenterNet anchor-free detection head predicting 3 classes (car, van, bus)

Ablation Support:
1. Radar-only ("radar")
2. LiDAR-only ("lidar")
3. Camera-assisted branch ("camera") [LiDAR-assisted camera RGB projection into BEV]
4. Full multimodal ("radar", "lidar", "camera")
5. C2a LiDAR branch ablation (lidar_bev zeroed)
"""

from typing import Dict, List, Optional, Tuple, Union, Sequence, Any
import torch
import torch.nn as nn

from .backbones import ResNet18BEVBackbone
from .fusion import MultimodalBEVFusion
from .centernet_head import CenterNetBEVHead


class BEVMultimodalDetector(nn.Module):
    """
    Baseline BEV Multimodal Fusion Detector for 3D/BEV Object Detection.
    """

    SUPPORTED_MODALITIES = ("radar", "lidar", "camera")

    def __init__(
        self,
        branch_channels: int = 64,
        fusion_channels: int = 128,
        num_classes: int = 3,
        seed: Optional[int] = None
    ):
        """
        Args:
            branch_channels: Feature channel output of each ResNet-18 backbone (default 64).
            fusion_channels: Channels in fusion neck and input to detection head (default 128).
            num_classes: Number of detection target classes (default 3: car, van, bus).
            seed: Deterministic random initialization seed.
        """
        super().__init__()
        self.branch_channels = branch_channels
        self.fusion_channels = fusion_channels
        self.num_classes = num_classes

        # Deterministic seeding if requested
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Backbones
        # Radar: 1-channel Cartesian power raster
        self.radar_backbone = ResNet18BEVBackbone(in_channels=1, out_channels=branch_channels)
        # LiDAR: 3-channel BEV raster (height, log density, mean intensity)
        self.lidar_backbone = ResNet18BEVBackbone(in_channels=3, out_channels=branch_channels)
        # Camera-assisted: 3-channel PointPainted BEV raster (LiDAR-assisted RGB)
        self.camera_backbone = ResNet18BEVBackbone(in_channels=3, out_channels=branch_channels)

        # 2. Multimodal Fusion Neck
        self.fusion = MultimodalBEVFusion(
            branch_channels=branch_channels,
            fusion_channels=fusion_channels,
            active_modalities=self.SUPPORTED_MODALITIES,
            use_fixed_slot_fusion=True
        )

        # 3. CenterNet Detection Head
        self.head = CenterNetBEVHead(
            in_channels=fusion_channels,
            head_conv=64,
            num_classes=num_classes
        )

    def extract_features(
        self,
        radar_bev: Optional[torch.Tensor] = None,
        lidar_bev: Optional[torch.Tensor] = None,
        camera_bev: Optional[torch.Tensor] = None,
        active_modalities: Optional[Sequence[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract BEV feature maps for all active sensor branches.
        """
        active = [m.lower() for m in (active_modalities or self.SUPPORTED_MODALITIES)]
        features = {}

        if "radar" in active and radar_bev is not None:
            features["radar"] = self.radar_backbone(radar_bev)

        if "lidar" in active and lidar_bev is not None:
            features["lidar"] = self.lidar_backbone(lidar_bev)

        if "camera" in active and camera_bev is not None:
            features["camera"] = self.camera_backbone(camera_bev)

        return features

    def forward(
        self,
        sample_or_radar: Union[Dict[str, Any], torch.Tensor],
        lidar_bev: Optional[torch.Tensor] = None,
        camera_bev: Optional[torch.Tensor] = None,
        active_modalities: Optional[Sequence[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single-sample, batched tensor inputs, or dataset sample dictionaries.

        Args:
            sample_or_radar: Either a dataset sample dictionary containing "radar_bev",
                "lidar_bev", and "camera_bev", or a raw radar tensor (B, 1, 512, 512).
            lidar_bev: Tensor of shape (B, 3, 512, 512) if passing raw tensors.
            camera_bev: Tensor of shape (B, 3, 512, 512) if passing raw tensors.
            active_modalities: Optional list of active modalities to execute ablation:
                - ["radar", "lidar", "camera"]: Full multimodal
                - ["radar"]: Radar-only
                - ["lidar"]: LiDAR-only
                - ["camera"]: Camera-assisted branch
                - ["radar", "camera"]: C2a LiDAR branch ablation

        Returns:
            Dict containing:
                - "heatmap": (B, num_classes, 128, 128) class center probabilities
                - "heatmap_logits": (B, num_classes, 128, 128) raw logits
                - "offset": (B, 2, 128, 128) sub-pixel center offsets
                - "size": (B, 2, 128, 128) box width and length
                - "rot": (B, 2, 128, 128) orientation unit vector (sin theta, cos theta)
                - "fused_features": (B, fusion_channels, 128, 128) intermediate BEV features
        """
        # Parse inputs
        if isinstance(sample_or_radar, dict):
            r_bev = sample_or_radar.get("radar_bev")
            l_bev = sample_or_radar.get("lidar_bev")
            c_bev = sample_or_radar.get("camera_bev")
        else:
            r_bev = sample_or_radar
            l_bev = lidar_bev
            c_bev = camera_bev

        # Ensure batch dimension: if single sample (C, H, W), unsqueeze to (1, C, H, W)
        if r_bev is not None and r_bev.ndim == 3:
            r_bev = r_bev.unsqueeze(0)
        if l_bev is not None and l_bev.ndim == 3:
            l_bev = l_bev.unsqueeze(0)
        if c_bev is not None and c_bev.ndim == 3:
            c_bev = c_bev.unsqueeze(0)

        active = [m.lower() for m in (active_modalities or self.SUPPORTED_MODALITIES)]

        # 1. Branch Feature Extraction
        branch_feats = self.extract_features(
            radar_bev=r_bev,
            lidar_bev=l_bev,
            camera_bev=c_bev,
            active_modalities=active
        )

        # 2. Multimodal Fusion
        fused = self.fusion(branch_feats, active_modalities=active)

        # 3. CenterNet Head
        preds = self.head(fused)
        preds["fused_features"] = fused

        return preds
