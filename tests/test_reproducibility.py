import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import numpy as np
import random
from src.dataset.indexer import RadiateIndexer
from src.dataset.radiate_dataset import RadiateMultimodalDataset
from src.radiate_fusion import RadiateCalib
from src.models.multimodal_detector import BEVMultimodalDetector
from src.models.losses import CenterNetLoss, build_centernet_targets

def run_forward_backward():
    # 1. Deterministic Seeds
    torch.manual_seed(12345)
    np.random.seed(12345)
    random.seed(12345)
    
    # 2. Dataset
    calib = RadiateCalib("config/default-calib.yaml")
    indexer = RadiateIndexer("city_3_0", split="train")
    ds = RadiateMultimodalDataset(indexer, calib)
    
    sample = ds[0]
    
    # 3. Model
    model = BEVMultimodalDetector(seed=12345)
    model.train()
    
    batch = {
        'radar_bev': sample['radar_bev'].unsqueeze(0),
        'lidar_bev': sample['lidar_bev'].unsqueeze(0),
        'camera_bev': sample['camera_bev'].unsqueeze(0)
    }
    targets = build_centernet_targets(
        target_boxes_bev=sample["target_boxes_bev"].unsqueeze(0),
        target_labels=sample["target_labels"].unsqueeze(0),
        ignore_mask=sample["ignore_mask"],
        target_boxes_metric=sample["target_boxes_metric"].unsqueeze(0)
    )
    
    loss_fn = CenterNetLoss()
    
    # Forward
    out = model(batch)
    losses = loss_fn(out, targets)
    total_loss = losses["loss"]
    total_loss.backward()
    
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
            
    return {
        "inputs": batch,
        "targets": targets,
        "out": out,
        "loss": total_loss.item(),
        "grad_norm": grad_norm
    }
    
def test_reproducibility():
    res1 = run_forward_backward()
    res2 = run_forward_backward()
    
    # Compare Inputs
    assert torch.allclose(res1["inputs"]["radar_bev"], res2["inputs"]["radar_bev"], atol=1e-7)
    
    # Compare Targets
    assert torch.allclose(res1["targets"]["heatmap"], res2["targets"]["heatmap"], atol=1e-7)
    
    # Compare Model Output
    assert torch.allclose(res1["out"]["heatmap_logits"], res2["out"]["heatmap_logits"], atol=1e-7)
    
    # Compare Loss and Gradients
    assert abs(res1["loss"] - res2["loss"]) < 1e-7
    assert abs(res1["grad_norm"] - res2["grad_norm"]) < 1e-7
