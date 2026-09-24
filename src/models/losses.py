"""
src/models/losses.py
--------------------
CenterNet detection losses with explicit ignore-region masking.
Implements:
1. GaussianFocalLoss: Modified focal loss for center heatmaps where pixels
   overlapping annotated non-target objects (ignore mask) contribute zero
   negative penalty.
2. RegL1Loss: Masked L1 regression loss for offsets, sizes, and orientation.
3. CenterNetLoss: Combined multi-task detection loss.
4. Target Generator: Builds training ground-truth heatmaps, regression targets,
   and downsampled ignore masks.
"""

from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFocalLoss(nn.Module):
    """
    CenterNet modified focal loss with ignore mask support.

    Formula:
        L_pos = - (1 - Y_hat)^alpha * log(Y_hat)                  for Y == 1
        L_neg = - (1 - Y)^beta * Y_hat^alpha * log(1 - Y_hat) * (1 - M_ignore) for Y < 1

    The (1 - M_ignore) term completely suppresses negative penalties for
    pixels overlapping annotated non-target objects (pedestrian, truck, bike).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_heatmap: torch.Tensor,
        ignore_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute focal loss with optional ignore masking.

        Args:
            pred_logits: (B, C, H, W) raw class logits.
            target_heatmap: (B, C, H, W) float in [0, 1] with Gaussian peaks at centers.
            ignore_mask: (B, 1, H, W) or (B, C, H, W) binary mask where 1 = ignore region.

        Returns:
            loss: Scalar tensor.
        """
        pred = torch.sigmoid(pred_logits)
        pred = torch.clamp(pred, min=self.eps, max=1.0 - self.eps)

        pos_mask = target_heatmap.eq(1.0).float()
        neg_mask = target_heatmap.lt(1.0).float()

        # Weighting factors
        pos_weights = torch.pow(1.0 - pred, self.alpha)
        neg_weights = torch.pow(1.0 - target_heatmap, self.beta) * torch.pow(pred, self.alpha)

        # Apply ignore mask to negative locations
        if ignore_mask is not None:
            # Broadcast ignore_mask if (B, 1, H, W)
            if ignore_mask.shape[1] == 1 and target_heatmap.shape[1] > 1:
                ignore_mask = ignore_mask.expand(-1, target_heatmap.shape[1], -1, -1)
            # Detections overlapping ignore regions contribute zero penalty
            neg_weights = neg_weights * (1.0 - ignore_mask.float())

        pos_loss = -pos_weights * torch.log(pred) * pos_mask
        neg_loss = -neg_weights * torch.log(1.0 - pred) * neg_mask

        num_pos = pos_mask.sum()
        if num_pos > 0:
            total_loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        else:
            total_loss = neg_loss.sum()

        return total_loss


class RegL1Loss(nn.Module):
    """
    Masked L1 regression loss evaluated strictly at positive center locations.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, K, H, W) predicted regression map.
            target: (B, K, H, W) target values.
            mask: (B, 1, H, W) binary indicator of object centers (1 where object center exists).

        Returns:
            loss: Scalar tensor.
        """
        # Expand mask across regression channels
        mask_exp = mask.expand_as(pred).float()
        num_pos = mask_exp.sum()

        if num_pos == 0:
            return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

        diff = torch.abs(pred - target) * mask_exp
        return diff.sum() / num_pos


class CenterNetLoss(nn.Module):
    """
    Multi-task loss for CenterNet BEV detector combining heatmap focal loss
    and bounding box regression losses.
    """

    def __init__(
        self,
        weight_hm: float = 1.0,
        weight_offset: float = 1.0,
        weight_size: float = 0.1,
        weight_rot: float = 0.1
    ):
        super().__init__()
        self.focal_loss = GaussianFocalLoss()
        self.reg_loss = RegL1Loss()

        self.weight_hm = weight_hm
        self.weight_offset = weight_offset
        self.weight_size = weight_size
        self.weight_rot = weight_rot

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Dict from CenterNetBEVHead with keys:
                - heatmap_logits, offset, size, rot
            targets: Dict with keys:
                - heatmap, offset, size, rot, reg_mask, ignore_mask

        Returns:
            loss_dict: Dict with individual and total loss components.
        """
        hm_loss = self.focal_loss(
            predictions["heatmap_logits"],
            targets["heatmap"],
            ignore_mask=targets.get("ignore_mask")
        )

        offset_loss = self.reg_loss(
            predictions["offset"],
            targets["offset"],
            targets["reg_mask"]
        )

        size_loss = self.reg_loss(
            predictions["size"],
            targets["size"],
            targets["reg_mask"]
        )

        rot_loss = self.reg_loss(
            predictions["rot"],
            targets["rot"],
            targets["reg_mask"]
        )

        total_loss = (
            self.weight_hm * hm_loss
            + self.weight_offset * offset_loss
            + self.weight_size * size_loss
            + self.weight_rot * rot_loss
        )

        return {
            "loss": total_loss,
            "loss_heatmap": hm_loss,
            "loss_offset": offset_loss,
            "loss_size": size_loss,
            "loss_rot": rot_loss,
        }


# ------------------------------------------------------------------------------
# Target Generation Helpers
# ------------------------------------------------------------------------------

def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.5) -> float:
    """Compute Gaussian splat radius based on box dimensions and IoU overlap."""
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(max(0, b1 ** 2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(max(0, b2 ** 2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(max(0, b3 ** 2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2
    return max(0.0, min(r1, r2, r3))


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int):
    """Draw 2D Gaussian kernel on heatmap in-place."""
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    x, y = center
    H, W = heatmap.shape

    left, right = min(x, radius), min(W - x - 1, radius)
    top, bottom = min(y, radius), min(H - y - 1, radius)

    if left + right < 0 or top + bottom < 0:
        return

    masked_heatmap = heatmap[y - top : y + bottom + 1, x - left : x + right + 1]

    # Generate 2D Gaussian patch
    y_coords, x_coords = np.ogrid[-top : bottom + 1, -left : right + 1]
    patch = np.exp(-(x_coords ** 2 + y_coords ** 2) / (2 * sigma ** 2))
    patch[patch < np.finfo(patch.dtype).eps * patch.max()] = 0

    np.maximum(masked_heatmap, patch, out=masked_heatmap)


def build_centernet_targets(
    target_boxes_bev: torch.Tensor,
    target_labels: torch.Tensor,
    ignore_mask: Optional[torch.Tensor] = None,
    target_boxes_metric: Optional[torch.Tensor] = None,
    grid_size: Tuple[int, int] = (128, 128),
    stride: int = 4,
    num_classes: int = 3,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Build CenterNet ground-truth targets for a single sample or batch.

    Args:
        target_boxes_bev: (N, 4) or (B, N, 4) [col, row, width, length] in 512x512 BEV pixels.
        target_labels: (N,) or (B, N) class IDs in {0, 1, 2}.
        ignore_mask: (1, 512, 512) or (B, 1, 512, 512) binary ignore mask.
        target_boxes_metric: Optional metric boxes containing rotation [x, y, w, l, rot_rad].
        grid_size: (H_feat, W_feat) output resolution (default 128x128).
        stride: Downsampling factor (default 4).
        num_classes: Number of target classes (default 3).

    Returns:
        Dict with keys: heatmap, offset, size, rot, reg_mask, ignore_mask.
    """
    dev = device or target_boxes_bev.device
    H_g, W_g = grid_size

    # Handle batched vs unbatched
    if target_boxes_bev.ndim == 2:
        boxes_list = [target_boxes_bev]
        labels_list = [target_labels]
        metric_list = [target_boxes_metric] if target_boxes_metric is not None else [None]
        if ignore_mask is not None:
            if ignore_mask.ndim == 3:
                ignore_mask = ignore_mask.unsqueeze(0)  # (1, 1, 512, 512)
        batch_size = 1
    else:
        boxes_list = list(target_boxes_bev)
        labels_list = list(target_labels)
        metric_list = list(target_boxes_metric) if target_boxes_metric is not None else [None] * len(boxes_list)
        batch_size = len(boxes_list)

    # Downsample ignore mask using max pooling
    if ignore_mask is not None:
        ignore_down = F.max_pool2d(ignore_mask.float(), kernel_size=stride, stride=stride)
    else:
        ignore_down = torch.zeros((batch_size, 1, H_g, W_g), dtype=torch.float32, device=dev)

    hm = np.zeros((batch_size, num_classes, H_g, W_g), dtype=np.float32)
    offset = np.zeros((batch_size, 2, H_g, W_g), dtype=np.float32)
    size = np.zeros((batch_size, 2, H_g, W_g), dtype=np.float32)
    rot = np.zeros((batch_size, 2, H_g, W_g), dtype=np.float32)
    reg_mask = np.zeros((batch_size, 1, H_g, W_g), dtype=np.float32)

    for b in range(batch_size):
        b_boxes = boxes_list[b]
        b_labels = labels_list[b]
        b_metric = metric_list[b]

        if len(b_boxes) == 0:
            continue

        b_boxes_np = b_boxes.detach().cpu().numpy()
        b_labels_np = b_labels.detach().cpu().numpy()
        b_metric_np = b_metric.detach().cpu().numpy() if b_metric is not None else None

        for i in range(len(b_boxes_np)):
            col_px, row_px, w_px, l_px = b_boxes_np[i][:4]
            cls_id = int(b_labels_np[i])

            if cls_id < 0 or cls_id >= num_classes:
                continue

            # Grid coordinates at stride 4
            col_g = col_px / float(stride)
            row_g = row_px / float(stride)
            cx_int = int(math.floor(col_g))
            cy_int = int(math.floor(row_g))

            if not (0 <= cx_int < W_g and 0 <= cy_int < H_g):
                continue

            # Box dimensions at stride
            w_g = w_px / float(stride)
            l_g = l_px / float(stride)
            radius = gaussian_radius((l_g, w_g))
            radius = max(0, int(radius))

            # Draw Gaussian on heatmap
            draw_gaussian(hm[b, cls_id], (cx_int, cy_int), radius)

            # Sub-pixel offsets
            offset[b, 0, cy_int, cx_int] = col_g - cx_int
            offset[b, 1, cy_int, cx_int] = row_g - cy_int

            # Bounding box size (width, length)
            size[b, 0, cy_int, cx_int] = w_px
            size[b, 1, cy_int, cx_int] = l_px

            # Orientation
            if b_metric_np is not None and b_metric_np.shape[1] >= 5:
                theta = float(b_metric_np[i, 4])
                rot[b, 0, cy_int, cx_int] = math.sin(theta)
                rot[b, 1, cy_int, cx_int] = math.cos(theta)
            else:
                rot[b, 0, cy_int, cx_int] = 0.0
                rot[b, 1, cy_int, cx_int] = 1.0

            reg_mask[b, 0, cy_int, cx_int] = 1.0

    return {
        "heatmap": torch.from_numpy(hm).to(dev),
        "offset": torch.from_numpy(offset).to(dev),
        "size": torch.from_numpy(size).to(dev),
        "rot": torch.from_numpy(rot).to(dev),
        "reg_mask": torch.from_numpy(reg_mask).to(dev),
        "ignore_mask": ignore_down.to(dev),
    }
