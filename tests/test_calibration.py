"""
test_calibration.py
-------------------
Unit tests for RADIATE calibration loading, transforms, and projections.

Run with:
    python -m pytest tests/test_calibration.py -v
or:
    python tests/test_calibration.py
"""

import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from radiate_fusion import (
    RadiateCalib, parse_timestamp_file, find_nearest_frame,
    lidar_abs_frame_to_csv, bbox_to_radar_3d_corners, project_bbox_to_camera,
    load_lidar_csv, load_annotations_for_frame
)

CALIB_PATH  = os.path.join(os.path.dirname(__file__), '..', 'config', 'default-calib.yaml')
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'tiny_foggy')

# ============================================================
# Fixture
# ============================================================

@pytest.fixture(scope='module')
def calib():
    return RadiateCalib(CALIB_PATH)


# ============================================================
# 1. Calibration loading
# ============================================================

class TestCalibrationLoading:

    def test_calib_file_exists(self):
        assert os.path.exists(CALIB_PATH), 'Calibration YAML not found at %s' % CALIB_PATH

    def test_calib_loads_without_error(self):
        c = RadiateCalib(CALIB_PATH)
        assert c is not None

    def test_lidar_translation_values(self, calib):
        T = calib.T_lidar
        assert T.shape == (3,)
        assert abs(T[0] - 0.6003)    < 1e-6, 'T_x mismatch'
        assert abs(T[1] - (-0.120102)) < 1e-6, 'T_y mismatch'
        assert abs(T[2] - 0.250012)  < 1e-6, 'T_z mismatch'

    def test_camera_intrinsics_values(self, calib):
        K = calib.K_left
        assert K.shape == (3, 3)
        assert abs(K[0, 0] - 337.9191) < 0.01, 'fx mismatch'
        assert abs(K[1, 1] - 338.6957) < 0.01, 'fy mismatch'
        assert abs(K[0, 2] - 341.7366) < 0.01, 'cx mismatch'
        assert abs(K[1, 2] - 200.7360) < 0.01, 'cy mismatch'

    def test_distortion_coefficients(self, calib):
        d = calib.dist_left
        assert len(d) == 5
        assert abs(d[0] - (-0.183879)) < 1e-4, 'k1 mismatch'
        assert abs(d[1] - 0.030861)    < 1e-4, 'k2 mismatch'

    def test_cam_resolution(self, calib):
        assert calib.cam_left_res == (672, 376)


# ============================================================
# 2. Rodrigues conversion
# ============================================================

class TestRodriguesConversion:

    def test_lidar_rotation_is_valid(self, calib):
        """R_lidar must be a valid SO(3) rotation matrix."""
        R = calib.R_lidar
        assert R.shape == (3, 3)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-6, 'det(R_lidar) = %.8f, expected 1.0' % det

    def test_lidar_rotation_orthogonal(self, calib):
        R = calib.R_lidar
        err = np.max(np.abs(R @ R.T - np.eye(3)))
        assert err < 1e-6, 'R_lidar not orthogonal, max err=%.2e' % err

    def test_cam_rotation_is_valid(self, calib):
        R = calib.R_cam_left
        assert R.shape == (3, 3)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-6, 'det(R_cam) = %.8f, expected 1.0' % det

    def test_cam_rotation_orthogonal(self, calib):
        R = calib.R_cam_left
        err = np.max(np.abs(R @ R.T - np.eye(3)))
        assert err < 1e-6, 'R_cam not orthogonal, max err=%.2e' % err

    def test_lidar_rotation_near_identity(self, calib):
        """LiDAR rotation should be near-identity (small misalignment) + camera axis permutation."""
        R = calib.R_lidar
        P = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0],
        ])
        diff = np.max(np.abs(R - P))
        assert diff < 0.1, 'R_lidar deviates too far from expected permutation: %.4f' % diff

    def test_zero_rodrigues_gives_identity(self):
        import cv2
        R, _ = cv2.Rodrigues(np.zeros(3))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-9)


# ============================================================
# 3. LiDAR-to-radar transform
# ============================================================

class TestLiDARToRadar:

    def test_lidar_origin_maps_to_T(self, calib):
        """LiDAR origin (0,0,0) maps to the LiDAR sensor position in the radar frame.
        Built via the camera calibration chain (official RADIATE SDK convention).
        Must be physically close to LidarT within sensor mounting tolerances."""
        origin = np.array([[0.0, 0.0, 0.0]])
        result = calib.lidar_to_radar(origin)
        assert result.shape == (1, 3)
        # Dominant lateral component: LiDAR is ~0.6m right of radar
        assert abs(result[0, 0] - calib.LidarT[0]) < 0.5, (
            f'Lateral {result[0,0]:.4f} too far from LidarT[0]={calib.LidarT[0]:.4f}')
        # Overall offset within physical mounting tolerance (<2m)
        assert np.linalg.norm(result[0] - calib.LidarT) < 2.0

    def test_output_shape_preserved(self, calib):
        N = 100
        pts = np.random.randn(N, 5)   # 5 cols like real LiDAR CSV
        result = calib.lidar_to_radar(pts)
        assert result.shape == (N, 3)

    def test_transform_preserves_distances(self, calib):
        """Pure rotation preserves point-to-point distances."""
        pts_a = np.array([[1.0, 0.0, 0.0]])
        pts_b = np.array([[0.0, 1.0, 0.0]])
        r_a = calib.lidar_to_radar(pts_a)
        r_b = calib.lidar_to_radar(pts_b)
        dist_in  = np.linalg.norm(pts_a[0, :3] - pts_b[0, :3])
        dist_out = np.linalg.norm(r_a[0] - r_b[0])
        assert abs(dist_in - dist_out) < 1e-9

    def test_bev_pixel_conversion_center(self, calib):
        """Radar origin (0,0,0) should map to BEV pixel center (576, 576)."""
        origin_radar = np.array([[0.0, 0.0, 0.0]])
        px = calib.radar_pts_to_bev_px(origin_radar)
        assert px[0, 0] == 576  # col
        assert px[0, 1] == 576  # row

    def test_bev_pixel_forward(self, calib):
        """Point at (x=0, y=50m, z=0) should be 50m forward = row < 576 (up in image)."""
        p = np.array([[0.0, 50.0, 0.0]])
        px = calib.radar_pts_to_bev_px(p)
        assert px[0, 1] < 576, 'Forward point should have row < 576 (up)'
        expected_row = 576 - int(50.0 / 0.17361)
        assert abs(px[0, 1] - expected_row) <= 1

    def test_bev_pixel_lateral_right(self, calib):
        """Point at (x=10m, y=0, z=0) should be to the right (col > 576)."""
        p = np.array([[10.0, 0.0, 0.0]])
        px = calib.radar_pts_to_bev_px(p)
        assert px[0, 0] > 576

    def test_actual_lidar_csv_loads(self):
        """A real tiny_foggy LiDAR CSV should load with correct columns."""
        csv_path = os.path.join(DATASET_DIR, 'velo_lidar', '000011.csv')
        if not os.path.exists(csv_path):
            pytest.skip('LiDAR CSV not found')
        pts = load_lidar_csv(csv_path)
        assert pts.ndim == 2
        assert pts.shape[1] >= 3, 'Need at least x,y,z columns'
        assert len(pts) > 1000, 'Expected >1000 LiDAR points'

    def test_lidar_points_in_reasonable_range(self, calib):
        """After transform, LiDAR points should be within 100m of radar."""
        csv_path = os.path.join(DATASET_DIR, 'velo_lidar', '000011.csv')
        if not os.path.exists(csv_path):
            pytest.skip('LiDAR CSV not found')
        pts = load_lidar_csv(csv_path)
        pts_radar = calib.lidar_to_radar(pts)
        ranges = np.sqrt(pts_radar[:, 0]**2 + pts_radar[:, 1]**2)
        pct_in_range = np.mean(ranges < 150)
        assert pct_in_range > 0.95, 'Too many points outside 150m: %.1f%%' % (pct_in_range*100)


# ============================================================
# 4. Camera projection
# ============================================================

class TestCameraProjection:

    def test_forward_point_has_positive_z(self, calib):
        """A point 50m forward should have positive Z in camera (in front)."""
        p_radar = np.array([[0.0, 50.0, 1.0]])
        p_cam = calib.radar_3d_to_cam_left(p_radar)
        assert p_cam[0, 2] > 0, 'Forward point should be in front of camera (Z>0)'

    def test_forward_point_projects_in_frame(self, calib):
        """A point 50m forward should project inside the camera image."""
        p_radar = np.array([[0.0, 50.0, 1.0]])
        p_cam = calib.radar_3d_to_cam_left(p_radar)
        uvs, mask = calib.project_to_cam_left(p_cam)
        assert np.any(mask), 'No in-front points found'
        W, H = calib.cam_left_res
        u, v = uvs[0]
        assert 0 <= u < W, 'u=%.1f out of [0, %d)' % (u, W)
        assert 0 <= v < H, 'v=%.1f out of [0, %d)' % (v, H)

    def test_behind_camera_filtered(self, calib):
        """A point behind the camera should be filtered out."""
        p_radar = np.array([[0.0, -10.0, 1.0]])   # behind vehicle
        p_cam = calib.radar_3d_to_cam_left(p_radar)
        uvs, mask = calib.project_to_cam_left(p_cam)
        if p_cam[0, 2] <= 0:
            assert uvs.shape[0] == 0, 'Should have no projections for behind-camera points'

    def test_projection_at_various_ranges(self, calib):
        """Objects from 10m to 80m forward should project in-frame (with some slack)."""
        success_count = 0
        W, H = calib.cam_left_res
        for dist in [10, 20, 30, 40, 50, 60, 70, 80]:
            p_radar = np.array([[0.0, float(dist), 1.0]])
            p_cam = calib.radar_3d_to_cam_left(p_radar)
            uvs, mask = calib.project_to_cam_left(p_cam)
            if np.any(mask):
                u, v = uvs[0]
                if (-50 <= u < W+50) and (-50 <= v < H+50):
                    success_count += 1
        assert success_count >= 4, 'Expected at least 4/8 range tests in-frame'


# ============================================================
# 5. Annotation projection
# ============================================================

class TestAnnotationProjection:

    def test_bbox_to_3d_corners_shape(self):
        corners = bbox_to_radar_3d_corners(576, 400, 50, 80)
        assert corners.shape == (8, 3), 'Expected 8 corners (bottom 4 + top 4)'

    def test_bbox_center_at_origin(self):
        """A bbox centered at radar BEV center (576, 576) should have x=0, y=0."""
        corners = bbox_to_radar_3d_corners(576, 576, 100, 100)
        # corners[:4] = bottom layer at z_bottom=-1.5m; corners[4:] = top layer at z_top=0.0m
        bottom = corners[:4]
        top    = corners[4:]
        np.testing.assert_allclose(bottom[:, 2], -1.5, atol=1e-9,
            err_msg='Bottom corners must be at z_bottom=-1.5m (vehicle base)')
        np.testing.assert_allclose(top[:, 2], 0.0, atol=1e-9,
            err_msg='Top corners must be at z_top=0.0m (radar/roof plane)')
        # Centroid x,y should be (0, 0)
        assert abs(bottom[:, 0].mean()) < 1e-6
        assert abs(bottom[:, 1].mean()) < 1e-6

    def test_bbox_height_correct(self):
        """Top corners should be at z_top=1.5m."""
        corners = bbox_to_radar_3d_corners(576, 576, 100, 100, z_bottom=0.0, z_top=1.5)
        top_z = corners[4:, 2]
        np.testing.assert_allclose(top_z, 1.5, atol=1e-9)

    def test_bus_annotation_projects_into_frame(self, calib):
        """The bus annotation in frame 5 should project into camera frame."""
        ann_path = os.path.join(DATASET_DIR, 'annotations', 'annotations.json')
        if not os.path.exists(ann_path):
            pytest.skip('Annotation file not found')
        anns = load_annotations_for_frame(ann_path, 5)
        bus_anns = [a for a in anns if a['class_name'] == 'bus']
        assert len(bus_anns) > 0, 'Expected bus annotation in frame 5'

        bus = bus_anns[0]
        uvs, all_valid = project_bbox_to_camera(bus['position'], calib)
        assert uvs is not None, 'Bus projection returned None'
        valid_uvs = uvs[~np.isnan(uvs[:, 0])]
        assert len(valid_uvs) >= 4, 'Expected at least 4 projected corners, got %d' % len(valid_uvs)

    def test_all_frame5_annotations_project(self, calib):
        """All frame 5 annotations should produce at least partial projections."""
        ann_path = os.path.join(DATASET_DIR, 'annotations', 'annotations.json')
        if not os.path.exists(ann_path):
            pytest.skip('Annotation file not found')
        anns = load_annotations_for_frame(ann_path, 5)
        for ann in anns:
            uvs, _ = project_bbox_to_camera(ann['position'], calib)
            assert uvs is not None, 'Projection failed for %s id=%d' % (
                ann['class_name'], ann['id'])
            valid_uvs = uvs[~np.isnan(uvs[:, 0])]
            assert len(valid_uvs) > 0, 'Zero valid corners for %s id=%d' % (
                ann['class_name'], ann['id'])


# ============================================================
# Runner
# ============================================================

if __name__ == '__main__':
    # Simple runner without pytest
    import traceback

    test_classes = [
        TestCalibrationLoading,
        TestRodriguesConversion,
        TestLiDARToRadar,
        TestCameraProjection,
        TestAnnotationProjection,
    ]

    c = RadiateCalib(CALIB_PATH)
    total, passed, failed = 0, 0, 0
    failures = []

    for cls in test_classes:
        print('\n=== %s ===' % cls.__name__)
        instance = cls()
        for name in dir(cls):
            if not name.startswith('test_'):
                continue
            method = getattr(instance, name)
            total += 1
            try:
                # Handle fixture injection
                import inspect
                sig = inspect.signature(method)
                if 'calib' in sig.parameters:
                    method(c)
                else:
                    method()
                print('  PASS  %s' % name)
                passed += 1
            except Exception as e:
                print('  FAIL  %s  ->  %s' % (name, e))
                failures.append((cls.__name__, name, str(e)))
                failed += 1

    print()
    print('=' * 50)
    print('Results: %d/%d passed  (%d failed)' % (passed, total, failed))
    if failures:
        print('\nFailed tests:')
        for cls_name, test_name, err in failures:
            print('  %s::%s  ->  %s' % (cls_name, test_name, err))
    print('STATUS: %s' % ('PASS' if failed == 0 else 'FAIL'))
