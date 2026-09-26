import pytest
import numpy as np
import torch
from src.preprocessing.bev_grid import BEVGridConfig
from src.eval.evaluator import Evaluator, match_predictions, calculate_iou_2d
from src.dataset.radiate_dataset import RadiateMultimodalDataset
from src.models.centernet_head import CenterNetBEVHead

def test_bev_to_metric_roundtrip():
    bev_cfg = BEVGridConfig()
    
    # Known metric coordinates
    x_m = 12.5
    y_m = 30.0
    w_m = 2.0
    l_m = 4.0
    
    # Metric -> BEV Pixel
    col, row = bev_cfg.metric_to_pixel(x_m, y_m)
    w_px = w_m / bev_cfg.dx
    l_px = l_m / bev_cfg.dy
    
    # BEV Pixel -> Metric
    x_rt, y_rt = bev_cfg.pixel_to_metric(col, row)
    w_rt = w_px * bev_cfg.dx
    l_rt = l_px * bev_cfg.dy
    
    assert np.isclose(x_m, x_rt)
    assert np.isclose(y_m, y_rt)
    assert np.isclose(w_m, w_rt)
    assert np.isclose(l_m, l_rt)

def test_prediction_gt_matching():
    bev_cfg = BEVGridConfig()
    evaluator = Evaluator(num_classes=3, iou_threshold=0.5)
    
    # Create a synthetic GT box in metric coordinates
    # [x_m, y_m, w_m, l_m, angle]
    gt_box = np.array([[10.0, 20.0, 2.0, 4.0, 0.0, 0]]) # class 0
    
    # Simulate a raw prediction in BEV pixel space that perfectly matches GT
    col, row = bev_cfg.metric_to_pixel(10.0, 20.0)
    w_px = 2.0 / bev_cfg.dx
    l_px = 4.0 / bev_cfg.dy
    
    # Mimic the decode step
    boxes_bev = np.array([[col, row, w_px, l_px, 0.0]])
    scores = np.array([0.95])
    labels = np.array([0])
    
    # Perform the conversion step from train_c0.py
    boxes_metric = np.zeros_like(boxes_bev)
    boxes_metric[:, 0], boxes_metric[:, 1] = bev_cfg.pixel_to_metric(boxes_bev[:, 0], boxes_bev[:, 1])
    boxes_metric[:, 2] = boxes_bev[:, 2] * bev_cfg.dx
    boxes_metric[:, 3] = boxes_bev[:, 3] * bev_cfg.dy
    boxes_metric[:, 4] = boxes_bev[:, 4]
    
    batch_preds = [{'boxes': boxes_metric, 'scores': scores, 'labels': labels}]
    batch_gts = [gt_box]
    
    evaluator.add_batch(batch_preds, batch_gts)
    metrics = evaluator.compute_metrics()
    
    # Because it matches perfectly, AP for class 0 should be 1.0, precision 1.0, recall 1.0
    assert metrics['AP_class_0'] == 1.0
    assert metrics['Precision_class_0'] == 1.0
    assert metrics['Recall_class_0'] == 1.0
    assert len(evaluator.all_matches) == 1
    assert evaluator.all_matches[0]['matched'] == True
    assert np.isclose(evaluator.all_matches[0]['iou'], 1.0)

def test_ignore_boxes_prevention():
    # A prediction overlapping an ignore box must not be recorded as a false-positive match
    bev_cfg = BEVGridConfig()
    evaluator = Evaluator(num_classes=3, iou_threshold=0.5)
    
    # No GT boxes (empty frame)
    batch_gts = [np.empty((0, 6))]
    
    # Synthetic ignore box at [15.0, 30.0]
    ignore_boxes = [np.array([[15.0, 30.0, 2.0, 4.0, 0.0]])]
    
    # Synthetic prediction exactly on the ignore box
    boxes_metric = np.array([[15.0, 30.0, 2.0, 4.0, 0.0]])
    scores = np.array([0.8])
    labels = np.array([1]) # class 1
    
    batch_preds = [{'boxes': boxes_metric, 'scores': scores, 'labels': labels}]
    
    evaluator.add_batch(batch_preds, batch_gts, ignore_list=ignore_boxes)
    metrics = evaluator.compute_metrics()
    
    # Normally, predicting in an empty frame yields AP=0.
    # But because it was ignored, it's not a false positive.
    # Wait, Evaluator semantics for an empty frame with 0 matched and 0 FP?
    # Let's check Evaluator: if no FP and no TP, it's essentially empty.
    # If c_matches = [], and num_gt = 0, AP = 1.0
    assert len(evaluator.all_matches) == 0 # Ignored predictions are NOT added to all_matches as false positives?
    # Wait! In evaluator.py:
    # if not ignored: matches.append({'matched': False})
    # If it IS ignored, it is simply discarded and not appended to matches!
    # So all_matches will be empty.
    assert len(evaluator.all_matches) == 0
    # Zero-GT classes are not assigned artificial AP=1.
    # They receive AP=0 and are excluded from the mAP denominator.
    assert metrics['AP_class_1'] == 0.0
