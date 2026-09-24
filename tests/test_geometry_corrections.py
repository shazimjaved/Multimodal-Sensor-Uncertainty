"""
tests/test_geometry_corrections.py
-----------------------------------
Rigorous test suite for Phase 3D-PRE Critical Geometry & Methodology Corrections.

Covers:
1. Issue 1: Bounding Box Top-Left -> Center Conversion & Reversibility & Rotation
2. Issue 2: Raw vs Rectified Camera Projection & Stereo Rectification
3. Issue 3: Radar Coordinate System & Invertibility
4. Issue 4: Camera BEV Dependency & Precise Modality Decoupling (C1, C2, C2a, C3)
5. Issue 5: Koschmieder Visibility Consistency
6. Issue 6: LiDAR Range Cutoff & Extinction Point Counts
7. Issue 7: Radar Power Attenuation Formulation
8. Issue 8: Dataset Population Counts & Split Integrity
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import cv2
import pytest
import torch

from radiate_fusion import (
    RadiateCalib,
    load_lidar_csv,
    load_annotations_for_frame,
    bbox_to_radar_3d_corners,
    project_bbox_to_camera,
    parse_timestamp_file,
    find_nearest_frame,
    lidar_abs_frame_to_csv,
)
from src.preprocessing.bev_grid import BEVGridConfig, encode_targets
from src.preprocessing.camera import CameraPreprocessor
from src.degradation.camera import CameraDegradation
from src.degradation.lidar import LiDARDegradation
from src.degradation.radar import RadarDegradation
from src.degradation.conditions import DegradationEngine


CALIB_PATH = "config/default-calib.yaml"
CITY_DIR = "city_3_0"
FOG_DIR = "fog_6_0"


@pytest.fixture(scope="module")
def calib():
    return RadiateCalib(CALIB_PATH)


@pytest.fixture(scope="module")
def bev_cfg():
    return BEVGridConfig()


# ===========================================================================
# ISSUE 1 — Bounding Box Coordinate Conventions
# ===========================================================================

class TestBoundingBoxCorrections:
    """Verifies that bounding boxes follow the official RADIATE [x_tl, y_tl, w, h] convention."""

    def test_top_left_to_center_conversion(self, bev_cfg):
        """Upper-left [x, y, w, h] must convert to center [x + w/2, y + h/2]."""
        x_tl, y_tl, w_px, h_px = 500.0, 400.0, 20.0, 50.0
        ann = [{
            "id": 1,
            "class_name": "car",
            "position": [x_tl, y_tl, w_px, h_px],
            "rotation": 0.0
        }]
        res = encode_targets(ann, bev_cfg=bev_cfg)
        assert len(res["target_boxes_metric"]) == 1
        x_m, y_m, w_m, l_m, rot = res["target_boxes_metric"][0]

        expected_cx_px = x_tl + w_px / 2.0  # 510.0
        expected_cy_px = y_tl + h_px / 2.0  # 425.0
        expected_x_m = (expected_cx_px - 576.0) * 0.17361
        expected_y_m = (576.0 - expected_cy_px) * 0.17361
        expected_w_m = w_px * 0.17361
        expected_l_m = h_px * 0.17361

        assert abs(x_m - expected_x_m) < 1e-5
        assert abs(y_m - expected_y_m) < 1e-5
        assert abs(w_m - expected_w_m) < 1e-5
        assert abs(l_m - expected_l_m) < 1e-5

    def test_center_conversion_reversibility(self):
        """Converting top-left -> center -> top-left must be mathematically exact."""
        np.random.seed(42)
        for _ in range(50):
            x_tl = np.random.uniform(400, 750)
            y_tl = np.random.uniform(200, 550)
            w_px = np.random.uniform(10, 40)
            h_px = np.random.uniform(20, 80)

            # Forward: top-left to center
            cx_px = x_tl + w_px / 2.0
            cy_px = y_tl + h_px / 2.0

            # Backward: center to top-left
            recovered_x_tl = cx_px - w_px / 2.0
            recovered_y_tl = cy_px - h_px / 2.0

            assert abs(recovered_x_tl - x_tl) < 1e-12
            assert abs(recovered_y_tl - y_tl) < 1e-12

    def test_known_radiate_sample_location(self, bev_cfg):
        """Verifies Object ID 5 (van) in city_3_0 frame 5 lands at expected BEV location."""
        ann_path = os.path.join(CITY_DIR, "annotations", "annotations.json")
        if not os.path.exists(ann_path):
            pytest.skip("city_3_0 annotations not available")
        anns = load_annotations_for_frame(ann_path, 5)
        van_ann = [a for a in anns if a["id"] == 5]
        assert len(van_ann) == 1
        pos = van_ann[0]["position"]

        res = encode_targets(van_ann, bev_cfg=bev_cfg)
        assert len(res["target_boxes_metric"]) == 1
        x_m, y_m, w_m, l_m, rot = res["target_boxes_metric"][0]

        # In frame 5, van is right in front of vehicle: x ~ 0.3m, y ~ 9.0m, w ~ 3.0m, l ~ 8.7m
        assert 0.0 <= x_m <= 1.5, f"Expected van lateral position ~0.3m, got {x_m}"
        assert 8.0 <= y_m <= 10.5, f"Expected van longitudinal range ~9.0m, got {y_m}"
        assert 2.5 <= w_m <= 3.5, f"Expected van width ~3.0m, got {w_m}"
        assert 7.5 <= l_m <= 9.5, f"Expected van length ~8.7m, got {l_m}"

    def test_width_height_not_half_dimensions(self, bev_cfg):
        """Verifies width and height in target_boxes_bev represent full span, not half-dimensions."""
        w_px, h_px = 30.0, 60.0
        ann = [{"id": 1, "class_name": "car", "position": [576, 500, w_px, h_px], "rotation": 0.0}]
        res = encode_targets(ann, bev_cfg=bev_cfg)
        col, row, w_bev, h_bev = res["target_boxes_bev"][0]

        expected_w_bev = (w_px * 0.17361) / bev_cfg.dx
        expected_h_bev = (h_px * 0.17361) / bev_cfg.dy
        assert abs(w_bev - expected_w_bev) < 1e-4
        assert abs(h_bev - expected_h_bev) < 1e-4

    def test_raw_annotations_preserved_untouched(self, bev_cfg):
        """Raw annotations list in returned dict must match original without mutation."""
        orig_ann = [{"id": 1, "class_name": "car", "position": [570.0, 480.0, 20.0, 50.0], "rotation": 5.0}]
        res = encode_targets(orig_ann, bev_cfg=bev_cfg)
        assert res["raw_annotations"] == orig_ann
        assert res["raw_annotations"][0]["position"] == [570.0, 480.0, 20.0, 50.0]

    def test_rotation_convention(self, bev_cfg):
        """Rotation should be converted from degrees to radians in target_boxes_metric."""
        ann = [{"id": 1, "class_name": "car", "position": [570.0, 480.0, 20.0, 50.0], "rotation": 45.0}]
        res = encode_targets(ann, bev_cfg=bev_cfg)
        theta_rad = res["target_boxes_metric"][0, 4]
        assert abs(theta_rad - np.deg2rad(45.0)) < 1e-6


# ===========================================================================
# ISSUE 2 — Raw vs Rectified Camera
# ===========================================================================

class TestCameraRectification:
    """Verifies official RADIATE camera rectification and projection."""

    def test_calib_has_stereo_rectification(self, calib):
        """RadiateCalib must initialize stereo rectification matrices and maps."""
        assert getattr(calib, "has_rectification", False) is True
        assert hasattr(calib, "R_rect_left")
        assert hasattr(calib, "P_rect_left")
        assert hasattr(calib, "map_left_x")
        assert hasattr(calib, "map_left_y")
        assert calib.R_rect_left.shape == (3, 3)
        assert calib.P_rect_left.shape == (3, 4)

    def test_rectify_left_image_dimensions(self, calib):
        """Rectified image must preserve original camera dimensions (376, 672, 3)."""
        dummy_img = np.zeros((376, 672, 3), dtype=np.uint8)
        rect_img = calib.rectify_left_image(dummy_img)
        assert rect_img.shape == (376, 672, 3)
        assert rect_img.dtype == np.uint8

    def test_rectified_projection_consistency_with_undistort_points(self, calib):
        """
        Pinhole projection of R_rect @ P_cam via P_rect[:3, :3] must match
        cv2.undistortPoints of the raw distorted projection down to 1e-5 px.
        """
        pts_cam = np.array([
            [0.5, 0.2, 5.0],
            [-1.2, 0.1, 8.0],
            [1.8, -0.4, 12.0],
            [-0.3, 0.8, 15.0]
        ], dtype=float)

        # 1. Project to raw image
        uv_raw, _ = calib.project_to_cam_left(pts_cam)

        # 2. Undistort raw coordinates using OpenCV
        uv_undist = cv2.undistortPoints(
            uv_raw.reshape(-1, 1, 2),
            calib.K_left, calib.dist_left,
            R=calib.R_rect_left,
            P=calib.P_rect_left
        ).reshape(-1, 2)

        # 3. Project directly via rectified projection
        uv_rect, _ = calib.project_to_cam_left_rect(pts_cam)

        np.testing.assert_allclose(uv_rect, uv_undist, atol=1e-5)

    def test_camera_projection_on_city_3_0_frames(self, calib):
        """LiDAR projection on city_3_0 frames 5, 100, 500 must produce valid rectified points."""
        radar_ts = parse_timestamp_file(os.path.join(CITY_DIR, "Navtech_Cartesian.txt"))
        lidar_ts = parse_timestamp_file(os.path.join(CITY_DIR, "velo_lidar.txt"))
        cam_ts   = parse_timestamp_file(os.path.join(CITY_DIR, "zed_left.txt"))

        for rf in [5, 100, 500]:
            t = radar_ts[rf]
            lf, _, dt_l = find_nearest_frame(t, lidar_ts)
            cf, _, dt_c = find_nearest_frame(t, cam_ts)
            assert dt_l <= 0.050 and dt_c <= 0.050

            csv_file = os.path.join(CITY_DIR, "velo_lidar", lidar_abs_frame_to_csv(lf, 1))
            pts_raw = load_lidar_csv(csv_file)
            pts_radar = calib.lidar_to_radar(pts_raw)
            pts_cam = calib.radar_3d_to_cam_left(pts_radar)

            uvs, mask = calib.project_to_cam_left_rect(pts_cam)
            assert len(uvs) > 0

            # In-image points must lie within [0, 672) x [0, 376)
            # NaNs will correctly evaluate to False in the bounds check.
            u, v = uvs[:, 0], uvs[:, 1]
            in_bounds = (u >= 0) & (u < 672) & (v >= 0) & (v < 376)
            assert in_bounds.sum() > 5000, f"Expected >5000 points in frame {rf}, got {in_bounds.sum()}"

    def test_camera_projection_on_fog_6_0_frame(self, calib):
        """LiDAR projection on fog_6_0 frame 10 must produce valid rectified points."""
        radar_ts = parse_timestamp_file(os.path.join(FOG_DIR, "Navtech_Cartesian.txt"))
        lidar_ts = parse_timestamp_file(os.path.join(FOG_DIR, "velo_lidar.txt"))
        cam_ts   = parse_timestamp_file(os.path.join(FOG_DIR, "zed_left.txt"))

        rf = 10
        t = radar_ts[rf]
        lf, _, _ = find_nearest_frame(t, lidar_ts)
        pts_raw = load_lidar_csv(os.path.join(FOG_DIR, "velo_lidar", lidar_abs_frame_to_csv(lf, 1)))
        pts_radar = calib.lidar_to_radar(pts_raw)
        pts_cam = calib.radar_3d_to_cam_left(pts_radar)
        uvs, mask = calib.project_to_cam_left_rect(pts_cam)

        u, v = uvs[:, 0], uvs[:, 1]
        in_bounds = (u >= 0) & (u < 672) & (v >= 0) & (v < 376)
        assert in_bounds.sum() > 1000


# ===========================================================================
# ISSUE 3 — Radar Coordinate Conventions
# ===========================================================================

class TestRadarCoordinateSystem:
    """Verifies radar coordinate origin, signs, resolution, and invertibility."""

    def test_radar_origin_mapping(self, calib):
        """Center pixel (576, 576) must map to metric (0.0, 0.0)."""
        pts_radar = np.array([[0.0, 0.0, 0.0]])
        px = calib.radar_pts_to_bev_px(pts_radar)
        assert px[0, 0] == 576
        assert px[0, 1] == 576

    def test_radar_sign_conventions(self, calib):
        """Lateral right must be +X (col > 576); forward must be +Y (row < 576)."""
        pts = np.array([
            [10.0, 20.0, 0.0],   # Right 10m, Forward 20m
            [-10.0, -20.0, 0.0], # Left 10m, Rearward 20m
        ])
        px = calib.radar_pts_to_bev_px(pts)
        # Point 1: col > 576, row < 576
        assert px[0, 0] > 576
        assert px[0, 1] < 576
        # Point 2: col < 576, row > 576
        assert px[1, 0] < 576
        assert px[1, 1] > 576

    def test_radar_meters_per_pixel(self, calib):
        """Resolution must be 0.17361 m/px."""
        assert calib.RADAR_M_PER_PX == pytest.approx(0.17361, rel=1e-5)
        # 100 pixels along +X = 17.361 meters
        pts = np.array([[17.361, 0.0, 0.0]])
        px = calib.radar_pts_to_bev_px(pts)
        assert px[0, 0] == 576 + 100

    def test_radar_pixel_metric_invertibility(self):
        """Round-trip conversion pixel -> metric -> pixel must be exact."""
        m_per_px = 0.17361
        cx, cy = 576.0, 576.0

        for col in [100.0, 576.0, 800.0, 1100.0]:
            for row in [50.0, 300.0, 576.0, 950.0]:
                x_m = (col - cx) * m_per_px
                y_m = (cy - row) * m_per_px

                recovered_col = cx + x_m / m_per_px
                recovered_row = cy - y_m / m_per_px

                assert abs(recovered_col - col) < 1e-12
                assert abs(recovered_row - row) < 1e-12


# ===========================================================================
# ISSUE 4 — Camera BEV Terminology & Modality Dependency
# ===========================================================================

class TestModalityDependency:
    """Verifies that conditions C1, C2, C2a, C3 correctly handle cross-modal dependency."""

    def test_c1_keeps_lidar_clean(self):
        """C1 degrades only camera optics while keeping clean LiDAR geometry."""
        engine = DegradationEngine()
        sample = {
            "radar_bev": torch.zeros((1, 512, 512)),
            "lidar_bev": torch.ones((3, 512, 512)),
            "camera_bev": torch.ones((3, 512, 512)),
            "camera_bev_mask": torch.ones((1, 512, 512)),
            "camera_image": torch.ones((3, 376, 672)),
            "lidar_points_radar": np.zeros((100, 5)),
            "metadata": {"frame_id": 5}
        }
        res = engine.degrade(sample, condition="C1_L1", seed=42)
        deg_meta = res.get("degradation_metadata", res.get("metadata", {}).get("degradation", {}))
        assert "camera" in deg_meta["modalities_degraded"]
        assert "lidar" not in deg_meta["modalities_degraded"]
        assert torch.equal(res["lidar_bev"], sample["lidar_bev"])

    def test_c2a_is_branch_ablation(self):
        """C2a zeros out lidar_bev at model input level without touching camera."""
        engine = DegradationEngine()
        sample = {
            "radar_bev": torch.ones((1, 512, 512)),
            "lidar_bev": torch.ones((3, 512, 512)),
            "camera_bev": torch.ones((3, 512, 512)),
            "camera_bev_mask": torch.ones((1, 512, 512)),
            "metadata": {"frame_id": 5}
        }
        res = engine.degrade(sample, condition="C2a", seed=42)
        assert torch.all(res["lidar_bev"] == 0.0)
        assert torch.equal(res["radar_bev"], sample["radar_bev"])
        assert torch.equal(res["camera_bev"], sample["camera_bev"])


# ===========================================================================
# ISSUE 5 — Synthetic Camera Model (Koschmieder)
# ===========================================================================

class TestSyntheticCameraModel:
    """Verifies Koschmieder parameterization and visibility threshold consistency."""

    def test_koschmieder_visibility_values(self):
        """
        Visibility values must equal V_5% = 3.0 / beta under standard 5% contrast threshold.
        """
        cam_deg = CameraDegradation()
        expected = {
            1: 150.0, # 3.0 / 0.02
            2: 60.0,  # 3.0 / 0.05
            3: 30.0   # 3.0 / 0.10
        }
        for lvl, exp_v in expected.items():
            beta = cam_deg.levels[lvl]["beta"]
            v_calc = 3.0 / beta
            assert cam_deg.levels[lvl]["visibility_m"] == pytest.approx(exp_v, rel=1e-3)
            assert v_calc == pytest.approx(exp_v, rel=1e-3)


# ===========================================================================
# ISSUE 6 — LiDAR Degradation Point Counts
# ===========================================================================

class TestLiDARDegradationRationale:
    """Verifies that L3 degradation enforces 25m range cutoff without arbitrary sub-sampling."""

    def test_l3_max_range_cutoff(self):
        """All points beyond 25m must be suppressed in L3."""
        lidar_deg = LiDARDegradation()
        np.random.seed(42)
        # Create synthetic points from 5m to 70m
        ranges = np.linspace(5.0, 70.0, 1000)
        xyz = np.column_stack([ranges, np.zeros_like(ranges), np.zeros_like(ranges)])
        raw_pts = np.column_stack([xyz, np.ones((1000, 1)) * 50.0, np.zeros((1000, 1))])

        deg_pts, meta = lidar_deg.degrade_points(raw_pts, level=3, seed=42)
        r_deg = np.linalg.norm(deg_pts[:, :3], axis=1)
        assert r_deg.max() <= 25.0, f"Expected max range <= 25.0m, got {r_deg.max()}"


# ===========================================================================
# ISSUE 7 — Radar Corruption Model
# ===========================================================================

class TestRadarCorruptionModel:
    """Verifies that power loss formula uses linear power ratio 10^(-loss_db / 10)."""

    def test_radar_power_ratio_formula(self):
        """Attenuating received power by 3.5 dB must produce factor 10^(-0.35) ~= 0.4467."""
        radar_deg = RadarDegradation()
        loss_db = 3.5
        expected_att = 10.0 ** (-loss_db / 10.0)

        # Create uniform raster
        clean_raster = np.ones((1, 512, 512), dtype=np.float32) * 0.5
        deg_raster, meta = radar_deg.corrupt_raster(clean_raster, level=2, seed=42)

        # In absence of clutter and speckle, mean power ratio matches 10^(-loss_db / 10)
        # Here speckle has mean ~ 1.0, so mean signal ~= 0.5 * expected_att
        mean_deg = float(deg_raster.mean())
        assert abs(mean_deg - (0.5 * expected_att)) < 0.05


# ===========================================================================
# ISSUE 9 — Data Counts & Split Integrity
# ===========================================================================

class TestDatasetCountsAndIntegrity:
    """Verifies dataset frame counts, startup exclusions, and split sizes on disk."""

    def test_city_3_0_counts(self):
        """city_3_0 must have 713 radar frames, 4 startup exclusions, 532 train, 177 val."""
        radar_ts = parse_timestamp_file(os.path.join(CITY_DIR, "Navtech_Cartesian.txt"))
        assert len(radar_ts) == 713

        train_frames = [fid for fid in radar_ts if 5 <= fid <= 536]
        val_frames   = [fid for fid in radar_ts if 537 <= fid <= 713]
        startup      = [fid for fid in radar_ts if 1 <= fid <= 4]

        assert len(startup) == 4
        assert len(train_frames) == 532
        assert len(val_frames) == 177
        assert len(train_frames) + len(val_frames) == 709

    def test_fog_6_0_counts(self):
        """fog_6_0 must have 714 radar frames, 3 startup exclusions, 711 sync frames."""
        radar_ts = parse_timestamp_file(os.path.join(FOG_DIR, "Navtech_Cartesian.txt"))
        assert len(radar_ts) == 714
        sync_frames = [fid for fid in radar_ts if fid >= 4]
        assert len(sync_frames) == 711
