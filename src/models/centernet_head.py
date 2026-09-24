"""
src/models/centernet_head.py
----------------------------
CenterNet-style anchor-free 2D BEV detection head.
Predicts:
1. Object center class heatmap (3 classes: car, van, bus)
2. Sub-pixel center offset (dx, dy)
3. Object bounding box size (width, length)
4. Continuous 2D orientation representation (sin theta, cos theta)
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetBEVHead(nn.Module):
    """
    Anchor-free CenterNet detection head operating on fused BEV features.
    """

    def __init__(
        self,
        in_channels: int = 128,
        head_conv: int = 64,
        num_classes: int = 3,
        init_bias: float = -2.19  # log(0.1 / 0.9) to stabilize initial heatmap training
    ):
        super().__init__()
        self.in_channels = in_channels
        self.head_conv = head_conv
        self.num_classes = num_classes

        # 1. Heatmap Head (Object centers for car, van, bus)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True),
        )

        # 2. Offset Head (Sub-pixel grid offset dx, dy)
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
        )

        # 3. Size Head (Width, Length in BEV grid / metric units)
        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),  # Positive dimensions
        )

        # 4. Orientation Head (sin(theta), cos(theta) continuous vector)
        self.rot_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
        )

        # Initialize heatmap bias to prior probability (0.1)
        self.heatmap_head[-1].bias.data.fill_(init_bias)

    def forward(self, fused_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass predicting detection maps.

        Args:
            fused_feat: Tensor of shape (B, in_channels, H, W)

        Returns:
            Dict containing:
                - "heatmap_logits": (B, num_classes, H, W) raw class logits
                - "heatmap": (B, num_classes, H, W) class probabilities in (0, 1)
                - "offset": (B, 2, H, W) sub-pixel offsets (dx, dy)
                - "size": (B, 2, H, W) box dimensions (w, l)
                - "rot": (B, 2, H, W) normalized orientation (sin, cos)
        """
        hm_logits = self.heatmap_head(fused_feat)
        hm = torch.sigmoid(hm_logits)

        offset = self.offset_head(fused_feat)
        size = self.size_head(fused_feat)
        rot_raw = self.rot_head(fused_feat)

        # Normalize rotation vector (sin, cos) to unit length
        rot_norm = F.normalize(rot_raw, p=2, dim=1, eps=1e-6)

        return {
            "heatmap_logits": hm_logits,
            "heatmap": hm,
            "offset": offset,
            "size": size,
            "rot": rot_norm,
        }

    @staticmethod
    def decode_detections(
        predictions: Dict[str, torch.Tensor],
        k: int = 50,
        score_threshold: float = 0.1,
        downsample_stride: int = 4
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Decode CenterNet heatmaps and regressions into oriented bounding boxes.

        Args:
            predictions: Output dictionary from forward().
            k: Maximum number of detections to extract per sample.
            score_threshold: Minimum confidence score filter.
            downsample_stride: Downsampling factor from input to heatmap (default 4).

        Returns:
            List of length B, where each element is a dict with:
                - "boxes": (M, 5) [col, row, width, length, angle_rad]
                - "scores": (M,) confidence scores
                - "labels": (M,) class IDs in {0: car, 1: van, 2: bus}
        """
        hm = predictions["heatmap"]
        offset = predictions["offset"]
        size = predictions["size"]
        rot = predictions["rot"]

        B, C, H, W = hm.shape

        # 3x3 Maxpool NMS to find local peaks
        hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        keep = (hmax == hm).float()
        hm_peaks = hm * keep

        results = []
        for b in range(B):
            # Flatten spatial dimensions: (C, H * W)
            hm_b = hm_peaks[b].view(C, -1)
            scores_c, inds_c = torch.topk(hm_b, k=min(k, H * W), dim=1)

            # Find top-k across all classes
            scores_flat = scores_c.view(-1)
            top_scores, top_idx = torch.topk(scores_flat, k=min(k, len(scores_flat)))

            # Filter by score threshold
            valid = top_scores >= score_threshold
            top_scores = top_scores[valid]
            top_idx = top_idx[valid]

            if len(top_scores) == 0:
                results.append({
                    "boxes": torch.empty((0, 5), dtype=torch.float32, device=hm.device),
                    "scores": torch.empty((0,), dtype=torch.float32, device=hm.device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=hm.device),
                })
                continue

            classes = top_idx // min(k, H * W)
            spatial_inds = inds_c.view(-1)[top_idx]

            ys = (spatial_inds // W).float()
            xs = (spatial_inds % W).float()

            # Gather offsets, sizes, rotations
            dx = offset[b, 0, ys.long(), xs.long()]
            dy = offset[b, 1, ys.long(), xs.long()]
            w = size[b, 0, ys.long(), xs.long()]
            l = size[b, 1, ys.long(), xs.long()]
            sin_theta = rot[b, 0, ys.long(), xs.long()]
            cos_theta = rot[b, 1, ys.long(), xs.long()]

            # Decode center in original 512x512 BEV pixel grid
            col_bev = (xs + dx) * downsample_stride
            row_bev = (ys + dy) * downsample_stride
            theta = torch.atan2(sin_theta, cos_theta)

            boxes = torch.stack([col_bev, row_bev, w, l, theta], dim=-1)

            results.append({
                "boxes": boxes,
                "scores": top_scores,
                "labels": classes.long(),
            })

        return results
