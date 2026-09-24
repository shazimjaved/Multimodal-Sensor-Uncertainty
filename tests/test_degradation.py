"""
tests/test_degradation.py
-------------------------
Unit test suite for Phase 3C Synthetic Sensor Degradation Engine.

Covers:
1. Parameter loading from configs/experiment_protocol.yaml
2. L0 identity (bitwise identical)
3. Deterministic seeds (reproducibility and divergence)
4. Camera depth dependence (Koschmieder physics)
5. LiDAR dropout monotonicity and range cutoff
6. Radar attenuation direction and power loss
7. Output shape preservation
8. Numerical validity (No NaN/Inf)
9. Raw dataset immutability
10. Multimodal condition dispatch (C0, C1, C2, C2a, C3, C4, C5)
"""

import os
import sys
import copy
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from radiate_fusion import RadiateCalib
from dataset import RadiateIndexer, RadiateMultimodalDataset
from degradation import (
    CameraDegradation,
    LiDARDegradation,
    RadarDegradation,
    DegradationEngine,
    degrade_sample,
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
# 1. Parameter loading tests
# ------------------------------------------------------------------------------
class TestParameterLoading:
    def test_camera_parameters_loaded(self):
        cam_deg = CameraDegradation("configs/experiment_protocol.yaml")
        assert 0 in cam_deg.levels
        assert 3 in cam_deg.levels
        assert cam_deg.levels[1]["beta"] == pytest.approx(0.02)
        assert cam_deg.levels[2]["beta"] == pytest.approx(0.05)
        assert cam_deg.levels[3]["beta"] == pytest.approx(0.10)
        assert len(cam_deg.airlight) == 3

    def test_lidar_parameters_loaded(self):
        lidar_deg = LiDARDegradation("configs/experiment_protocol.yaml")
        assert lidar_deg.levels[0]["alpha_ext"] == 0.0
        assert lidar_deg.levels[1]["alpha_ext"] == pytest.approx(0.010)
        assert lidar_deg.levels[2]["alpha_ext"] == pytest.approx(0.025)
        assert lidar_deg.levels[3]["alpha_ext"] == pytest.approx(0.050)
        assert lidar_deg.levels[3]["max_range_m"] == pytest.approx(25.0)

    def test_radar_parameters_loaded(self):
        radar_deg = RadarDegradation("configs/experiment_protocol.yaml")
        assert radar_deg.levels[0]["power_loss_db"] == 0.0
        assert radar_deg.levels[1]["power_loss_db"] == pytest.approx(1.5)
        assert radar_deg.levels[2]["power_loss_db"] == pytest.approx(3.5)
        assert radar_deg.levels[3]["power_loss_db"] == pytest.approx(6.0)


# ------------------------------------------------------------------------------
# 2. L0 Identity tests
# ------------------------------------------------------------------------------
class TestL0Identity:
    def test_camera_l0_bitwise_identical(self, sample):
        cam_deg = CameraDegradation()
        depth_dummy = np.full((376, 672), 20.0, dtype=np.float32)
        clean_img = sample["camera_image"]
        deg_img, meta = cam_deg.degrade_image(clean_img, depth_dummy, level=0)
        assert torch.equal(clean_img, deg_img)
        assert meta["mean_transmission"] == 1.0

    def test_lidar_l0_bitwise_identical(self, sample):
        lidar_deg = LiDARDegradation()
        pts_path = sample["metadata"]["lidar_path"]
        from radiate_fusion import load_lidar_csv
        raw_pts = load_lidar_csv(pts_path)
        deg_pts, meta = lidar_deg.degrade_points(raw_pts, level=0)
        assert np.array_equal(raw_pts, deg_pts)
        assert meta["retention_pct"] == 100.0

    def test_radar_l0_bitwise_identical(self, sample):
        radar_deg = RadarDegradation()
        clean_radar = sample["radar_bev"]
        deg_radar, meta = radar_deg.corrupt_raster(clean_radar, level=0)
        assert torch.equal(clean_radar, deg_radar)
        assert meta["power_loss_db"] == 0.0

    def test_sample_c0_bitwise_identical(self, sample):
        c0 = degrade_sample(sample, "C0")
        assert torch.equal(c0["radar_bev"], sample["radar_bev"])
        assert torch.equal(c0["lidar_bev"], sample["lidar_bev"])
        assert torch.equal(c0["camera_bev"], sample["camera_bev"])
        assert torch.equal(c0["camera_image"], sample["camera_image"])


# ------------------------------------------------------------------------------
# 3. Determinism and Seed tests
# ------------------------------------------------------------------------------
class TestDeterministicSeeds:
    def test_same_seed_produces_identical_output(self, sample):
        deg1 = degrade_sample(sample, "C5_L2", seed=100)
        deg2 = degrade_sample(sample, "C5_L2", seed=100)
        assert torch.equal(deg1["radar_bev"], deg2["radar_bev"])
        assert torch.equal(deg1["lidar_bev"], deg2["lidar_bev"])
        assert torch.equal(deg1["camera_bev"], deg2["camera_bev"])
        assert torch.equal(deg1["camera_image"], deg2["camera_image"])

    def test_different_seeds_produce_different_output(self, sample):
        deg1 = degrade_sample(sample, "C5_L2", seed=100)
        deg2 = degrade_sample(sample, "C5_L2", seed=200)
        # Radar speckle/clutter will differ
        assert not torch.equal(deg1["radar_bev"], deg2["radar_bev"])
        # LiDAR dropout/jitter will differ
        assert not torch.equal(deg1["lidar_bev"], deg2["lidar_bev"])


# ------------------------------------------------------------------------------
# 4. Camera depth dependence tests
# ------------------------------------------------------------------------------
class TestCameraDepthDependence:
    def test_farther_depth_has_lower_transmission(self):
        cam_deg = CameraDegradation()
        depth_map = np.zeros((376, 672), dtype=np.float32)
        depth_map[:, :336] = 5.0    # Near region: 5m
        depth_map[:, 336:] = 50.0   # Far region: 50m

        img = np.full((3, 376, 672), 0.2, dtype=np.float32)
        deg_img, _ = cam_deg.degrade_image(img, depth_map, level=2)

        # In fog, distant objects become lighter (closer to airlight ~0.82)
        mean_near = float(deg_img[:, :, :336].mean())
        mean_far = float(deg_img[:, :, 336:].mean())
        assert mean_far > mean_near, "Distant region should be more obscured by fog airlight"


# ------------------------------------------------------------------------------
# 5. LiDAR dropout monotonicity tests
# ------------------------------------------------------------------------------
class TestLiDARDropoutMonotonicity:
    def test_dropout_increases_monotonically(self, sample):
        lidar_deg = LiDARDegradation()
        pts_path = sample["metadata"]["lidar_path"]
        from radiate_fusion import load_lidar_csv
        raw_pts = load_lidar_csv(pts_path)

        pts0, meta0 = lidar_deg.degrade_points(raw_pts, level=0, seed=42)
        pts1, meta1 = lidar_deg.degrade_points(raw_pts, level=1, seed=42)
        pts2, meta2 = lidar_deg.degrade_points(raw_pts, level=2, seed=42)
        pts3, meta3 = lidar_deg.degrade_points(raw_pts, level=3, seed=42)

        assert meta0["retained_point_count"] >= meta1["retained_point_count"]
        assert meta1["retained_point_count"] > meta2["retained_point_count"]
        assert meta2["retained_point_count"] > meta3["retained_point_count"]

    def test_max_range_cutoff_strictly_enforced(self, sample):
        lidar_deg = LiDARDegradation()
        pts_path = sample["metadata"]["lidar_path"]
        from radiate_fusion import load_lidar_csv
        raw_pts = load_lidar_csv(pts_path)

        for lvl, max_r in [(1, 65.0), (2, 45.0), (3, 25.0)]:
            pts, meta = lidar_deg.degrade_points(raw_pts, level=lvl, seed=42)
            r = np.linalg.norm(pts[:, :3], axis=1)
            # With jitter (+-0.08m max), points should not exceed max_r + 1.0m
            assert np.max(r) <= max_r + 1.0, f"Points at level {lvl} exceeded max range {max_r}"


# ------------------------------------------------------------------------------
# 6. Radar attenuation tests
# ------------------------------------------------------------------------------
class TestRadarAttenuation:
    def test_power_loss_monotonically_reduces_mean_signal(self, sample):
        radar_deg = RadarDegradation()
        clean = sample["radar_bev"]

        # Disable clutter spikes to test pure attenuation + speckle direction
        r0, m0 = radar_deg.corrupt_raster(clean, level=0)
        r1, m1 = radar_deg.corrupt_raster(clean, level=1)
        r2, m2 = radar_deg.corrupt_raster(clean, level=2)
        r3, m3 = radar_deg.corrupt_raster(clean, level=3)

        assert m0["mean_degraded"] > m1["mean_degraded"]
        assert m1["mean_degraded"] > m2["mean_degraded"]
        assert m2["mean_degraded"] > m3["mean_degraded"]


# ------------------------------------------------------------------------------
# 7. Output shape preservation
# ------------------------------------------------------------------------------
class TestShapePreservation:
    @pytest.mark.parametrize("condition", ["C0", "C1_L2", "C2_L2", "C2a", "C3_L2", "C4_L3", "C5_L3"])
    def test_tensor_shapes_preserved(self, sample, condition):
        deg = degrade_sample(sample, condition, seed=42)
        assert deg["radar_bev"].shape == (1, 512, 512)
        assert deg["lidar_bev"].shape == (3, 512, 512)
        assert deg["camera_bev"].shape == (3, 512, 512)
        assert deg["camera_bev_mask"].shape == (1, 512, 512)
        if deg["camera_image"] is not None:
            assert deg["camera_image"].shape == (3, 376, 672)


# ------------------------------------------------------------------------------
# 8. Numerical validity (No NaN/Inf)
# ------------------------------------------------------------------------------
class TestNumericalValidity:
    @pytest.mark.parametrize("condition", ["C0", "C1_L3", "C2_L3", "C2a", "C3_L3", "C4_L3", "C5_L3"])
    def test_no_nan_or_inf(self, sample, condition):
        deg = degrade_sample(sample, condition, seed=42)
        for key in ["radar_bev", "lidar_bev", "camera_bev", "camera_bev_mask"]:
            tensor = deg[key]
            assert torch.isfinite(tensor).all(), f"NaN or Inf found in {key} under {condition}"
            assert (tensor >= 0.0).all() and (tensor <= 1.0).all(), f"{key} out of [0, 1] range"

        if deg["camera_image"] is not None:
            c_img = deg["camera_image"]
            assert torch.isfinite(c_img).all()
            assert (c_img >= 0.0).all() and (c_img <= 1.0).all()


# ------------------------------------------------------------------------------
# 9. Raw dataset immutability
# ------------------------------------------------------------------------------
class TestImmutability:
    def test_original_sample_untouched_after_degradation(self, sample):
        radar_orig = sample["radar_bev"].clone()
        lidar_orig = sample["lidar_bev"].clone()
        cam_orig = sample["camera_bev"].clone()
        img_orig = sample["camera_image"].clone()

        # Run severe catastrophic degradation
        _ = degrade_sample(sample, "C5_L3", seed=42)

        # Check that original sample values are 100% unchanged
        assert torch.equal(sample["radar_bev"], radar_orig)
        assert torch.equal(sample["lidar_bev"], lidar_orig)
        assert torch.equal(sample["camera_bev"], cam_orig)
        assert torch.equal(sample["camera_image"], img_orig)


# ------------------------------------------------------------------------------
# 10. Condition dispatch tests
# ------------------------------------------------------------------------------
class TestConditionDispatch:
    def test_c1_affects_only_camera(self, sample):
        deg = degrade_sample(sample, "C1_L2")
        assert torch.equal(deg["radar_bev"], sample["radar_bev"])
        assert torch.equal(deg["lidar_bev"], sample["lidar_bev"])
        assert not torch.equal(deg["camera_image"], sample["camera_image"])

    def test_c2_affects_lidar_and_cascades_to_camera_bev(self, sample):
        deg = degrade_sample(sample, "C2_L2")
        assert torch.equal(deg["radar_bev"], sample["radar_bev"])
        assert torch.equal(deg["camera_image"], sample["camera_image"])  # optical image clean
        assert not torch.equal(deg["lidar_bev"], sample["lidar_bev"])

    def test_c2a_zeros_lidar_bev(self, sample):
        deg = degrade_sample(sample, "C2a")
        assert torch.max(deg["lidar_bev"]).item() == 0.0
        assert torch.equal(deg["radar_bev"], sample["radar_bev"])
        assert torch.equal(deg["camera_image"], sample["camera_image"])
        assert deg["degradation_metadata"]["is_ablation"] is True

    def test_c3_affects_only_radar(self, sample):
        deg = degrade_sample(sample, "C3_L2")
        assert not torch.equal(deg["radar_bev"], sample["radar_bev"])
        assert torch.equal(deg["lidar_bev"], sample["lidar_bev"])
        assert torch.equal(deg["camera_bev"], sample["camera_bev"])
        assert torch.equal(deg["camera_image"], sample["camera_image"])

    def test_c4_affects_camera_and_lidar_keeps_radar_clean(self, sample):
        deg = degrade_sample(sample, "C4_L3")
        assert torch.equal(deg["radar_bev"], sample["radar_bev"])  # radar clean
        assert not torch.equal(deg["lidar_bev"], sample["lidar_bev"])
        assert not torch.equal(deg["camera_image"], sample["camera_image"])

    def test_c5_affects_all_modalities(self, sample):
        deg = degrade_sample(sample, "C5_L3")
        assert not torch.equal(deg["radar_bev"], sample["radar_bev"])
        assert not torch.equal(deg["lidar_bev"], sample["lidar_bev"])
        assert not torch.equal(deg["camera_image"], sample["camera_image"])
        assert not torch.equal(deg["camera_bev"], sample["camera_bev"])
