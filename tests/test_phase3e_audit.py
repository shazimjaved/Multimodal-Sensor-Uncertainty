import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np
from src.dataset.indexer import RadiateIndexer
from src.dataset.radiate_dataset import RadiateMultimodalDataset
from src.radiate_fusion import RadiateCalib
from src.models.losses import build_centernet_targets
from src.preprocessing.bev_grid import BEVGridConfig

def test_end_to_end_path_city3():
    calib = RadiateCalib("config/default-calib.yaml")
    indexer = RadiateIndexer("city_3_0", split="train")
    ds = RadiateMultimodalDataset(indexer, calib)
    
    sample = ds[0]
    
    assert "target_boxes_bev" in sample
    assert "target_boxes_metric" in sample
    
    targets = build_centernet_targets(
        target_boxes_bev=sample["target_boxes_bev"].unsqueeze(0),
        target_labels=sample["target_labels"].unsqueeze(0),
        ignore_mask=sample["ignore_mask"],
        target_boxes_metric=sample["target_boxes_metric"].unsqueeze(0)
    )
    
    assert "heatmap" in targets
    assert targets["heatmap"].shape == (1, 3, 128, 128)
    assert targets["offset"].shape == (1, 2, 128, 128)
    assert targets["size"].shape == (1, 2, 128, 128)
    assert targets["rot"].shape == (1, 2, 128, 128)
    assert targets["ignore_mask"].shape == (1, 128, 128) # downsampled
    assert targets["reg_mask"].shape == (1, 1, 128, 128)

def test_train_val_split_semantics():
    # Verify train and val do not overlap in indices
    train_idx = RadiateIndexer("city_3_0", split="train")
    val_idx = RadiateIndexer("city_3_0", split="val")
    
    train_set = {r["radar_frame"] for r in train_idx.index_records}
    val_set = {r["radar_frame"] for r in val_idx.index_records}
    
    assert len(train_set.intersection(val_set)) == 0, "Train and Val splits overlap!"

def test_duplicate_leakage_check():
    train_idx = RadiateIndexer("city_3_0", split="train")
    val_idx = RadiateIndexer("city_3_0", split="val")
    
    # Check for identical radar paths
    train_paths = {r["radar_path"] for r in train_idx.index_records}
    val_paths = {r["radar_path"] for r in val_idx.index_records}
    
    intersection_paths = train_paths.intersection(val_paths)
    
    assert len(intersection_paths) == 0, f"Found {len(intersection_paths)} duplicate paths between train and val!"

def test_temporal_contiguity_group_leakage_check():
    """
    Verify train/validation leakage using temporal-contiguity grouping for leakage control (temporal contiguous blocks).
    Frames within 400ms of each other are considered part of the same temporal_contiguity_group.
    """
    full_idx = RadiateIndexer("city_3_0", split=None, exclude_frames=[])
    
    # Assign group IDs based on temporal continuity (dt < 400ms)
    groups = {}
    current_group = 0
    last_time = -1
    
    # Sort by radar_time to track contiguous blocks
    sorted_records = sorted(full_idx.index_records, key=lambda x: x["radar_time"])
    
    for r in sorted_records:
        t = r["radar_time"]
        if last_time != -1 and (t - last_time) > 0.4:
            current_group += 1
            
        groups[r["radar_frame"]] = current_group
        last_time = t
        
    train_idx = RadiateIndexer("city_3_0", split="train")
    val_idx = RadiateIndexer("city_3_0", split="val")
    
    train_groups = {groups[r["radar_frame"]] for r in train_idx.index_records if r["radar_frame"] in groups}
    val_groups = {groups[r["radar_frame"]] for r in val_idx.index_records if r["radar_frame"] in groups}
    
    intersection = train_groups.intersection(val_groups)
    assert len(intersection) == 0, f"Found {len(intersection)} leaking temporal groups between train and val! Shared groups: {intersection}"
