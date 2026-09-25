"""
tests/test_dataset_pipeline.py
------------------------------
Unit tests for the Phase 3B multimodal dataset loader and BEV preprocessor.
Tests indexing, synchronization, preprocessing, target encoding, and split isolation.

Run with:
    python -m pytest tests/test_dataset_pipeline.py -v
"""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from radiate_fusion import RadiateCalib
from preprocessing.bev_grid import BEVGridConfig, encode_bev_boxes
from preprocessing.radar import RadarPreprocessor
from preprocessing.lidar import LiDARPreprocessor
from preprocessing.camera import CameraPreprocessor
from dataset.indexer import RadiateIndexer
from dataset.radiate_dataset import RadiateMultimodalDataset

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CALIB_PATH = os.path.join(BASE_DIR, "config", "default-calib.yaml")
CITY_DIR = os.path.join(BASE_DIR, "city_3_0")
FOG_DIR = os.path.join(BASE_DIR, "fog_6_0")


@pytest.fixture(scope="module")
def calib():
    return RadiateCalib(CALIB_PATH)


@pytest.fixture(scope="module")
def bev_cfg():
    return BEVGridConfig(bev_height=512, bev_width=512)


@pytest.fixture(scope="module")
def train_indexer():
    return RadiateIndexer(
        dataset_dir=CITY_DIR,
        split_range=(5, 626),
        exclude_frames=[1, 2, 3, 4],
        max_dt_sec=0.050
    )


@pytest.fixture(scope="module")
def val_indexer():
    return RadiateIndexer(
        dataset_dir=CITY_DIR,
        split_range=(627, 713),
        exclude_frames=[1, 2, 3, 4],
        max_dt_sec=0.050
    )


# ==============================================================================
# 1. Dataset Indexing & Filtering Tests
# ==============================================================================

class TestDatasetIndexing:

    def test_train_split_exact_count(self, train_indexer):
        assert len(train_indexer) == 622, f"Expected 622 train frames, got {len(train_indexer)}"

    def test_val_split_exact_count(self, val_indexer):
        assert len(val_indexer) == 87, f"Expected 87 val frames, got {len(val_indexer)}"

    def test_combined_usable_count(self, train_indexer, val_indexer):
        assert len(train_indexer) + len(val_indexer) == 709

    def test_frames_1_to_4_properly_excluded(self):
        indexer = RadiateIndexer(CITY_DIR, split_range=(1, 10), exclude_frames=[1, 2, 3, 4])
        for f in [1, 2, 3, 4]:
            assert f in indexer.rejection_log
            assert indexer.rejection_log[f] == "explicitly_excluded"

    def test_train_val_zero_frame_overlap(self, train_indexer, val_indexer):
        train_frames = set(r["radar_frame"] for r in train_indexer)
        val_frames = set(r["radar_frame"] for r in val_indexer)
        overlap = train_frames.intersection(val_frames)
        assert len(overlap) == 0, f"Found overlapping frames: {overlap}"


# ==============================================================================
# 2. Synchronization Integrity Tests
# ==============================================================================

class TestSynchronization:

    def test_all_train_frames_under_sync_threshold(self, train_indexer):
        for r in train_indexer:
            assert r["lidar_dt_ms"] <= 50.0, f"Frame {r['radar_frame']} LiDAR dt={r['lidar_dt_ms']}ms > 50ms"
            assert r["cam_left_dt_ms"] <= 50.0, f"Frame {r['radar_frame']} Cam dt={r['cam_left_dt_ms']}ms > 50ms"

    def test_all_val_frames_under_sync_threshold(self, val_indexer):
        for r in val_indexer:
            assert r["lidar_dt_ms"] <= 50.0, f"Frame {r['radar_frame']} LiDAR dt={r['lidar_dt_ms']}ms > 50ms"
            assert r["cam_left_dt_ms"] <= 50.0, f"Frame {r['radar_frame']} Cam dt={r['cam_left_dt_ms']}ms > 50ms"

    def test_chronological_time_ordering(self, train_indexer, val_indexer):
        max_train_t = max(r["radar_time"] for r in train_indexer)
        min_val_t = min(r["radar_time"] for r in val_indexer)
        assert max_train_t < min_val_t, f"Train time ({max_train_t}) >= Val time ({min_val_t})"


# ==============================================================================
# 3. Sensor Preprocessing Tests
# ==============================================================================

class TestSensorPreprocessing:

    def test_radar_preprocessing_shape_and_range(self, bev_cfg):
        radar_proc = RadarPreprocessor(bev_cfg)
        path = os.path.join(CITY_DIR, "Navtech_Cartesian", "000005.png")
        bev = radar_proc.process_file(path)
        assert bev.shape == (1, 512, 512)
        assert bev.dtype == np.float32
        assert bev.min() >= 0.0 and bev.max() <= 1.0

    def test_lidar_preprocessing_shape_and_channels(self, calib, bev_cfg):
        lidar_proc = LiDARPreprocessor(calib, bev_cfg)
        path = os.path.join(CITY_DIR, "velo_lidar", "000029.csv")
        bev, pts_filt = lidar_proc.process_file(path)
        assert bev.shape == (3, 512, 512)
        assert bev.dtype == np.float32
        assert bev.min() >= 0.0 and bev.max() <= 1.0
        assert len(pts_filt) > 0

    def test_camera_point_painting_shape_and_mask(self, calib, bev_cfg):
        lidar_proc = LiDARPreprocessor(calib, bev_cfg)
        cam_proc = CameraPreprocessor(calib, bev_cfg)

        _, pts_filt = lidar_proc.process_file(os.path.join(CITY_DIR, "velo_lidar", "000029.csv"))
        cam_norm, cam_bev, cam_mask = cam_proc.process_file(
            os.path.join(CITY_DIR, "zed_left", "000002.png"), pts_radar=pts_filt
        )

        assert cam_norm.shape == (3, 376, 672)
        assert cam_bev.shape == (3, 512, 512)
        assert cam_mask.shape == (1, 512, 512)
        assert cam_bev.dtype == np.float32
        assert cam_mask.max() == 1.0


# ==============================================================================
# 4. Target Encoding and Ignore Regions Tests
# ==============================================================================

class TestTargetEncoding:

    def test_target_classes_only_car_van_bus(self, train_indexer, calib, bev_cfg):
        ds = RadiateMultimodalDataset(train_indexer, calib, bev_cfg)
        sample = ds[0] # Frame 5 contains van, bus, and pedestrians

        labels = sample["target_labels"].numpy()
        for l in labels:
            assert l in [0, 1, 2], f"Invalid label {l} found in target_labels!"

    def test_ignore_mask_generated_for_pedestrians(self, train_indexer, calib, bev_cfg):
        ds = RadiateMultimodalDataset(train_indexer, calib, bev_cfg)
        sample = ds[0] # Frame 5 has pedestrians

        ignore_mask = sample["ignore_mask"].numpy()
        assert ignore_mask.shape == (1, 512, 512)
        assert ignore_mask.sum() > 0, "Ignore mask should be active for non-target classes (pedestrians)!"

    def test_raw_annotations_preserved(self, train_indexer, calib, bev_cfg):
        ds = RadiateMultimodalDataset(train_indexer, calib, bev_cfg)
        sample = ds[0]
        raw = sample["metadata"]["raw_annotations"]
        assert len(raw) == 6 # Frame 5 has 6 total objects in raw annotations
        class_names = [a["class_name"] for a in raw]
        assert "pedestrian" in class_names


# ==============================================================================
# 5. Dataset Lazy Loading & Determinism Tests
# ==============================================================================

class TestDatasetDeterminism:

    def test_repeated_access_bitwise_identical(self, train_indexer, calib, bev_cfg):
        ds = RadiateMultimodalDataset(train_indexer, calib, bev_cfg)
        sample1 = ds[0]
        sample2 = ds[0]

        assert torch.all(sample1["radar_bev"] == sample2["radar_bev"])
        assert torch.all(sample1["lidar_bev"] == sample2["lidar_bev"])
        assert torch.all(sample1["camera_bev"] == sample2["camera_bev"])
        assert torch.all(sample1["ignore_mask"] == sample2["ignore_mask"])


# ==============================================================================
# 6. Held-Out fog_6_0 Isolation Tests
# ==============================================================================

class TestFog6Isolation:

    def test_fog_6_0_indexer_initializes_independently(self):
        fog_idx = RadiateIndexer(FOG_DIR, split_range=None, max_dt_sec=0.050)
        assert len(fog_idx) == 711
        summary = fog_idx.get_summary()
        assert summary["total_indexed"] == 711

    def test_fog_6_0_loads_sample_without_error(self, calib, bev_cfg):
        fog_idx = RadiateIndexer(FOG_DIR, split_range=None, max_dt_sec=0.050)
        fog_ds = RadiateMultimodalDataset(fog_idx, calib, bev_cfg)
        sample = fog_ds[0]

        assert sample["radar_bev"].shape == (1, 512, 512)
        assert sample["lidar_bev"].shape == (3, 512, 512)
        assert sample["camera_bev"].shape == (3, 512, 512)
        assert "metadata" in sample
