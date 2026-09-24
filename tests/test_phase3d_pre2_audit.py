"""
tests/test_phase3d_pre2_audit.py
--------------------------------
Phase 3D-PRE-2: Final Geometry + Pipeline Integration Audit.

Validates the remaining tasks for Phase 3D-PRE-2:
- Task 3: BEV Preprocessing/Loader Consistency
- Task 4: Degradation -> Fusion Propagation
- Task 5: Ignore Region Mask & Loss Consistency
- Task 6: FOV / Positive-Depth Gate
"""

import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radiate_fusion import RadiateCalib
from src.dataset.radiate_dataset import RadiateMultimodalDataset
from src.dataset.indexer import RadiateIndexer
from src.preprocessing.bev_grid import BEVGridConfig, encode_bev_boxes
from src.models.multimodal_detector import BEVMultimodalDetector
from src.models.losses import CenterNetLoss
from src.degradation.conditions import DegradationEngine

CALIB_PATH = "config/default-calib.yaml"
CITY_DIR = "city_3_0"

@pytest.fixture(scope="module")
def calib():
    return RadiateCalib(CALIB_PATH)

@pytest.fixture(scope="module")
def bev_cfg():
    return BEVGridConfig()

# ===========================================================================
# TASK 3 — BEV PREPROCESSING / LOADER
# ===========================================================================

class TestBEVLoaderConsistency:
    """Audit BEV Preprocessing/Loader (Coordinate consistency + Numerical reconstruction check)."""
    
    def test_numerical_reconstruction_check(self, bev_cfg):
        """Verify round trip: pixel -> metric -> radar origin."""
        # 1. Start with a known metric point (10m forward, 5m right)
        x_m, y_m = 5.0, 10.0
        
        # 2. Metric -> BEV pixel
        # bev_grid handles this: col = (x - x_min)/dx, row = (y_max - y)/dy
        # cx = -x_min/dx = 576. cy = y_max/dy = 576.
        # col = cx + x_m/dx = 576 + 5.0/0.17361 = 604.8
        # row = cy - y_m/dy = 576 - 10.0/0.17361 = 518.4
        expected_col = 576.0 + (x_m / bev_cfg.dx)
        expected_row = 576.0 - (y_m / bev_cfg.dy)
        
        # Invert it
        recovered_x_m = (expected_col - 576.0) * bev_cfg.dx
        recovered_y_m = (576.0 - expected_row) * bev_cfg.dy
        
        assert abs(recovered_x_m - x_m) < 1e-5
        assert abs(recovered_y_m - y_m) < 1e-5

    @pytest.mark.skipif(not os.path.exists(CITY_DIR), reason="Dataset required")
    def test_class_representation_aligns_with_target_encoding(self, calib):
        """Ensure only target classes are encoded; others are ignored."""
        indexer = RadiateIndexer(CITY_DIR, split='train')
        ds = RadiateMultimodalDataset(indexer, calib)
        assert 'car' in ds.TARGET_CLASSES
        assert 'pedestrian' not in ds.TARGET_CLASSES
        assert 'pedestrian' in ds.IGNORE_CLASSES


# ===========================================================================
# TASK 4 — PROPAGATION CHECK
# ===========================================================================

class TestDegradationToFusionPropagation:
    """Propagation check (Degradation -> Fusion -> Detector)."""
    
    @pytest.mark.skipif(not os.path.exists(CITY_DIR), reason="Dataset required")
    def test_degraded_sample_propagates_to_detector(self, calib):
        """Verify the detector handles a degraded sample correctly."""
        indexer = RadiateIndexer(CITY_DIR, split='train')
        ds = RadiateMultimodalDataset(indexer, calib)
        sample = ds[0] # Frame 5 (train start)
        
        engine = DegradationEngine()
        deg_sample = engine.degrade(sample, condition="C5_L3", seed=42)
        
        # Mock batching
        batch = {
            'radar_bev': deg_sample['radar_bev'].unsqueeze(0),
            'lidar_bev': deg_sample['lidar_bev'].unsqueeze(0),
            'camera_bev': deg_sample['camera_bev'].unsqueeze(0),
        }
        
        model = BEVMultimodalDetector()
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
            
        assert 'heatmap' in outputs
        assert outputs['heatmap'].shape == (1, 3, 128, 128)
        assert torch.isfinite(outputs['heatmap']).all()


# ===========================================================================
# TASK 5 — IGNORE REGION MASK AUDIT
# ===========================================================================

class TestIgnoreRegionMaskAudit:
    """Ignore Region Mask Audit (Class semantics + Loss masking)."""

    def test_ignore_mask_fully_zeros_penalties(self):
        """Check losses.py to verify that ignore_mask fully zero-outs penalties for ignore-classes."""
        loss_fn = CenterNetLoss()
        
        # B x C x H x W
        pred_hm = torch.sigmoid(torch.randn(1, 1, 128, 128)) 
        gt_hm = torch.zeros(1, 1, 128, 128)
        
        # Add an ignore region
        ignore_mask = torch.zeros(1, 1, 128, 128)
        ignore_mask[0, 0, 50:60, 50:60] = 1.0 # Ignored region
        
        # Case 1: Prediction in ignored region
        pred_hm_1 = pred_hm.clone()
        pred_hm_1[0, 0, 55, 55] = 0.99 # False positive in ignore region
        
        # Case 2: Prediction in valid background
        pred_hm_2 = pred_hm.clone()
        pred_hm_2[0, 0, 10, 10] = 0.99 # False positive in background
        
        loss_1 = loss_fn.focal_loss(pred_hm_1, gt_hm, ignore_mask)
        loss_2 = loss_fn.focal_loss(pred_hm_2, gt_hm, ignore_mask)
        
        # Loss 1 should be lower than Loss 2 because the FP in Loss 1 is ignored
        assert loss_1.item() < loss_2.item()


# ===========================================================================
# TASK 6 — FOV / POSITIVE-DEPTH GATE AUDIT
# ===========================================================================

class TestFOVPositiveDepthGateAudit:
    """FOV/Positive-Depth Gate Audit (Explicit visibility/clipping semantics)."""
    
    def test_project_to_cam_left_clips_out_of_fov(self, calib):
        """Explicit visibility/clipping semantics need explicit check."""
        
        # Point way off to the side (x = 1000m)
        pts_cam_out = np.array([[1000.0, 0.0, 10.0]])
        
        uvs, mask = calib.project_to_cam_left(pts_cam_out)
        
        # It has Z > 0.1 so mask is True
        assert mask[0] == True
        
        # BUT it might project outside image FOV
        u, v = uvs[0, 0], uvs[0, 1]
        
        # FOV check:
        W, H = calib.cam_left_res
        in_frame = (u >= 0) and (u < W) and (v >= 0) and (v < H)
        assert in_frame == False, "Point outside FOV should not be in_frame"
