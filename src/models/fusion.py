"""
src/models/fusion.py
--------------------
Multimodal BEV feature fusion neck.
Concatenates projected branch feature maps and applies a 1x1 convolution
fusion neck with batch normalization and ReLU activation.
Contains no self-attention or transformer layers.
"""

from typing import Dict, List, Optional, Tuple, Sequence
import torch
import torch.nn as nn


class MultimodalBEVFusion(nn.Module):
    """
    Channel-wise concatenation + 1x1 convolution fusion neck.

    Supports:
    - Fixed order concatenation: ("radar", "lidar", "camera")
    - Zero-masking ablation: Inactive modalities are zeroed out while preserving
      trained neck weight compatibility (e.g. C2a ablation or modality dropout).
    - Standalone single-modality configuration.
    """

    ALL_MODALITIES = ("radar", "lidar", "camera")

    def __init__(
        self,
        branch_channels: int = 64,
        fusion_channels: int = 128,
        active_modalities: Optional[Sequence[str]] = None,
        use_fixed_slot_fusion: bool = True
    ):
        """
        Args:
            branch_channels: Number of channels per sensor branch (default 64).
            fusion_channels: Output channel dimension of the fusion neck (default 128).
            active_modalities: Sequence of active modalities (e.g. ["radar", "lidar"]).
            use_fixed_slot_fusion: If True, fusion neck always expects slots for all
                modalities (3 * branch_channels) and zeroes out inactive modalities.
                This enables ablating trained multimodal models seamlessly.
                If False, in_channels = len(active_modalities) * branch_channels.
        """
        super().__init__()
        self.branch_channels = branch_channels
        self.fusion_channels = fusion_channels
        self.use_fixed_slot_fusion = use_fixed_slot_fusion

        if active_modalities is None:
            self.active_modalities = list(self.ALL_MODALITIES)
        else:
            self.active_modalities = [m.lower() for m in active_modalities]

        if self.use_fixed_slot_fusion:
            in_channels = len(self.ALL_MODALITIES) * branch_channels  # 3 * 64 = 192
        else:
            in_channels = len(self.active_modalities) * branch_channels

        # 1x1 Convolution Fusion Neck
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, fusion_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            # Lightweight 3x3 depthwise-separable or standard conv for local spatial context
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        branch_features: Dict[str, torch.Tensor],
        active_modalities: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        """
        Fuse branch features into a unified BEV representation.

        Args:
            branch_features: Dict mapping modality name ("radar", "lidar", "camera")
                to feature tensor of shape (B, branch_channels, H, W).
            active_modalities: Optional override of active modalities for dynamic ablation.

        Returns:
            fused_feat: Tensor of shape (B, fusion_channels, H, W).
        """
        active = active_modalities if active_modalities is not None else self.active_modalities
        active_set = set(m.lower() for m in active)

        # Determine reference shape (B, C, H, W) from any available active feature
        ref_feat = None
        for mod in self.ALL_MODALITIES:
            if mod in branch_features and branch_features[mod] is not None:
                ref_feat = branch_features[mod]
                break

        if ref_feat is None:
            raise ValueError("No valid branch features provided to fusion neck.")

        B, _, H, W = ref_feat.shape

        if self.use_fixed_slot_fusion:
            # Fixed order: ("radar", "lidar", "camera")
            feature_list = []
            for mod in self.ALL_MODALITIES:
                if mod in active_set and mod in branch_features and branch_features[mod] is not None:
                    feature_list.append(branch_features[mod])
                else:
                    # Inactive modality zeroed out (ablation mode)
                    zero_feat = torch.zeros(
                        (B, self.branch_channels, H, W),
                        dtype=ref_feat.dtype,
                        device=ref_feat.device
                    )
                    feature_list.append(zero_feat)

            cat_features = torch.cat(feature_list, dim=1)  # (B, 192, H, W)
        else:
            feature_list = []
            for mod in active:
                if mod in branch_features and branch_features[mod] is not None:
                    feature_list.append(branch_features[mod])
                else:
                    raise KeyError(f"Required active modality '{mod}' missing from branch_features.")
            cat_features = torch.cat(feature_list, dim=1)

        fused = self.neck(cat_features)  # (B, fusion_channels, H, W)
        return fused
