import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
from dataset.indexer import RadiateIndexer
from radiate_fusion import RadiateCalib
from dataset.radiate_dataset import RadiateMultimodalDataset
from torch.utils.data import DataLoader
from train_c0 import collate_fn
from eval.evaluator import Evaluator


def test_batch_device_transfer():
    # 1. Create a real batch using collate_fn
    calib = RadiateCalib("config/default-calib.yaml")
    
    # We can just take the first 2 items of city_3_0
    train_idx = RadiateIndexer("city_3_0", split="train")
    
    # Check that there are items
    assert len(train_idx) > 0, "No items found in dataset"
    
    train_ds = RadiateMultimodalDataset(train_idx, calib)
    
    # Disable shuffling to make it deterministic
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    batch = next(iter(train_loader))
    
    # 2. Confirm ignore_boxes_metric is a Python list
    assert isinstance(batch['ignore_boxes_metric'], list)
    assert len(batch['ignore_boxes_metric']) == 2
    
    # Check what type the items in the list are
    assert torch.is_tensor(batch['ignore_boxes_metric'][0])
    
    # 3. Confirm every tensor in the batch can be moved to a device
    # and 4. Confirm the batch-processing logic does not attempt .to() on the list
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for k in batch:
        if torch.is_tensor(batch[k]):
            # This should succeed without crashing
            batch[k] = batch[k].to(device, non_blocking=True)
            
    # Verify the tensors moved (or didn't fail at least)
    assert batch['radar_bev'].device.type == device.type
    
    # 5. Confirm validation can reach evaluator.add_batch()
    # (Just passing the ignore boxes through the expected list comprehension)
    evaluator = Evaluator(num_classes=3, iou_threshold=0.5)
    
    batch_preds = []
    # Dummy predictions
    for i in range(2):
        batch_preds.append({
            'boxes': torch.zeros((0, 5)),
            'scores': torch.zeros((0,)),
            'labels': torch.zeros((0,))
        })
        
    batch_gts = []
    for i in range(2):
        batch_gts.append(torch.zeros((0, 6)))
        
    batch_ignore_boxes = [ib.cpu().numpy() for ib in batch['ignore_boxes_metric']]
    
    # Should not crash
    evaluator.add_batch(batch_preds, batch_gts, batch_ignore_boxes)
    assert len(evaluator.all_matches) == 0
