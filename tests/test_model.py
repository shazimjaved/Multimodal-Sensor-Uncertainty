"""
tests/test_model.py
-------------------
Unit tests for Phase 3D Baseline BEV Multimodal Fusion Detector.

Covers:
1. Model initialization and deterministic seeding
2. Single-sample forward pass
3. Batched tensor forward pass
4. Output tensor shapes and 3-class heatmap
5. Branch ablation support (Radar-only, LiDAR-only, Camera-assisted, Full, C2a)
6. Ignore-mask handling in loss computation
7. Regression loss computation
8. Detection decoding (peak extraction)
9. CPU execution performance and memory footprint
"""

import os
import sys
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from radiate_fusion import RadiateCalib
from dataset import RadiateIndexer, RadiateMultimodalDataset
from models import (
    ResNet18BEVBackbone,
    MultimodalBEVFusion,
    CenterNetBEVHead,
    BEVMultimodalDetector,
    GaussianFocalLoss,
    CenterNetLoss,
    build_centernet_targets,
)


@pytest.fixture(scope="module")
def calib():
    return RadiateCalib("config/default-calib.yaml")


@pytest.fixture(scope="module")
def dataset(calib):
    indexer = RadiateIndexer("city_3_0", split="train")
    return RadiateMultimodalDataset(indexer=indexer, calib=calib)


@pytest.fixture(scope="module")
def sample(dataset):
    return dataset[0]


# ------------------------------------------------------------------------------
# 1. Initialization and Determinism Tests
# ------------------------------------------------------------------------------
class TestModelInitialization:
    def test_deterministic_initialization(self):
        m1 = BEVMultimodalDetector(seed=123)
        m2 = BEVMultimodalDetector(seed=123)
        m3 = BEVMultimodalDetector(seed=456)

        # Check weights identical for same seed
        p1 = list(m1.parameters())[0]
        p2 = list(m2.parameters())[0]
        p3 = list(m3.parameters())[0]

        assert torch.equal(p1, p2), "Models initialized with same seed must have identical parameters"
        assert not torch.equal(p1, p3), "Models initialized with different seeds must have different parameters"

    def test_parameter_count(self):
        model = BEVMultimodalDetector()
        num_params = sum(p.numel() for p in model.parameters())
        # ResNet18 * 3 (~33.8M) + Fusion (~0.17M) + CenterNet Head (~0.3M) ~ 34.3M
        assert 30_000_000 < num_params < 40_000_000, f"Expected ~34M parameters, got {num_params}"


# ------------------------------------------------------------------------------
# 2. Forward Pass & Output Shape Tests
# ------------------------------------------------------------------------------
class TestForwardPass:
    def test_single_sample_forward(self, sample):
        model = BEVMultimodalDetector(seed=42)
        model.eval()

        with torch.no_grad():
            out = model(sample)

        assert "heatmap" in out
        assert "offset" in out
        assert "size" in out
        assert "rot" in out
        assert "fused_features" in out

        # Check exact tensor shapes
        assert out["heatmap"].shape == (1, 3, 128, 128)
        assert out["offset"].shape == (1, 2, 128, 128)
        assert out["size"].shape == (1, 2, 128, 128)
        assert out["rot"].shape == (1, 2, 128, 128)
        assert out["fused_features"].shape == (1, 128, 128, 128)

        # Value range checks
        assert (out["heatmap"] >= 0.0).all() and (out["heatmap"] <= 1.0).all()
        assert (out["size"] >= 0.0).all()  # non-negative sizes
        rot_norms = torch.norm(out["rot"], p=2, dim=1)
        assert torch.allclose(rot_norms, torch.ones_like(rot_norms), atol=1e-5)  # unit vectors

    def test_batched_forward(self, sample):
        model = BEVMultimodalDetector(seed=42)
        model.eval()

        B = 2
        r_batch = torch.stack([sample["radar_bev"]] * B)
        l_batch = torch.stack([sample["lidar_bev"]] * B)
        c_batch = torch.stack([sample["camera_bev"]] * B)

        with torch.no_grad():
            out = model(r_batch, l_batch, c_batch)

        assert out["heatmap"].shape == (B, 3, 128, 128)
        assert out["offset"].shape == (B, 2, 128, 128)
        assert out["size"].shape == (B, 2, 128, 128)
        assert out["rot"].shape == (B, 2, 128, 128)


# ------------------------------------------------------------------------------
# 3. Branch Ablation Tests
# ------------------------------------------------------------------------------
class TestBranchAblation:
    def test_radar_only_ablation(self, sample):
        model = BEVMultimodalDetector(seed=42)
        model.eval()
        with torch.no_grad():
            out = model(sample, active_modalities=["radar"])
        assert out["heatmap"].shape == (1, 3, 128, 128)

    def test_lidar_only_ablation(self, sample):
        model = BEVMultimodalDetector(seed=42)
        model.eval()
        with torch.no_grad():
            out = model(sample, active_modalities=["lidar"])
        assert out["heatmap"].shape == (1, 3, 128, 128)

    def test_camera_assisted_branch_ablation(self, sample):
        model = BEVMultimodalDetector(seed=42)
        model.eval()
        with torch.no_grad():
            out = model(sample, active_modalities=["camera"])
        assert out["heatmap"].shape == (1, 3, 128, 128)

    def test_c2a_lidar_branch_ablation(self, sample):
        model = BEVMultimodalDetector(seed=42)
        model.eval()
        with torch.no_grad():
            # C2a: LiDAR branch zeroed/excluded, Radar + Camera active
            out = model(sample, active_modalities=["radar", "camera"])
        assert out["heatmap"].shape == (1, 3, 128, 128)


# ------------------------------------------------------------------------------
# 4. Ignore-Mask and Loss Tests
# ------------------------------------------------------------------------------
class TestLossAndIgnoreMask:
    def test_ignore_mask_suppresses_negative_penalty(self):
        loss_fn = GaussianFocalLoss()

        # Simulated predictions (some false alarm predictions)
        logits = torch.full((1, 3, 128, 128), -1.0)  # low confidence
        # Add high confidence false positive at location (50, 50)
        logits[0, 0, 50, 50] = 5.0  # high confidence ~0.99

        # Target heatmap is completely empty (no true targets)
        target_hm = torch.zeros((1, 3, 128, 128))

        # Case A: Without ignore mask
        loss_without_ignore = loss_fn(logits, target_hm, ignore_mask=None)

        # Case B: With ignore mask active at (50, 50)
        ignore_mask = torch.zeros((1, 1, 128, 128))
        ignore_mask[0, 0, 50, 50] = 1.0  # masked as ignore region
        loss_with_ignore = loss_fn(logits, target_hm, ignore_mask=ignore_mask)

        # The loss with ignore mask MUST be strictly lower because false alarm on ignore region is suppressed!
        assert loss_with_ignore < loss_without_ignore, (
            f"Expected ignore mask to suppress penalty: {loss_with_ignore} vs {loss_without_ignore}"
        )

    def test_full_centernet_loss_computation(self, sample):
        model = BEVMultimodalDetector(seed=42)
        model.train()

        out = model(sample)
        targets = build_centernet_targets(
            target_boxes_bev=sample["target_boxes_bev"],
            target_labels=sample["target_labels"],
            ignore_mask=sample["ignore_mask"],
            target_boxes_metric=sample["target_boxes_metric"]
        )

        criterion = CenterNetLoss()
        losses = criterion(out, targets)

        assert "loss" in losses
        assert "loss_heatmap" in losses
        assert "loss_offset" in losses
        assert "loss_size" in losses
        assert "loss_rot" in losses

        assert torch.isfinite(losses["loss"]).all()
        assert losses["loss"].item() > 0.0


# ------------------------------------------------------------------------------
# 5. Detection Decoding Tests
# ------------------------------------------------------------------------------
class TestDetectionDecoder:
    def test_decode_detections_format(self, sample):
        model = BEVMultimodalDetector(seed=42)
        model.eval()

        with torch.no_grad():
            preds = model(sample)

        decoded = CenterNetBEVHead.decode_detections(preds, k=20, score_threshold=0.01)
        assert len(decoded) == 1
        d0 = decoded[0]
        assert "boxes" in d0
        assert "scores" in d0
        assert "labels" in d0

        if len(d0["boxes"]) > 0:
            assert d0["boxes"].shape[1] == 5  # [col, row, width, length, theta]
            assert (d0["labels"] >= 0).all() and (d0["labels"] < 3).all()
