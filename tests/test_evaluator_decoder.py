import pytest
import numpy as np
import torch
import math
from src.eval.evaluator import Evaluator, calculate_iou_2d
from src.models.centernet_head import CenterNetBEVHead

class TestEvaluatorTasks:
    def test_calculate_iou_2d(self):
        # Two identical boxes
        box1 = [0, 0, 10, 10, 0]
        box2 = [0, 0, 10, 10, 0]
        assert abs(calculate_iou_2d(box1, box2) - 1.0) < 1e-5

        # Non-overlapping
        box3 = [20, 20, 10, 10, 0]
        assert calculate_iou_2d(box1, box3) == 0.0

        # Partial overlap (half)
        box4 = [5, 0, 10, 10, 0]
        iou = calculate_iou_2d(box1, box4)
        # area of intersection is 5*10=50, union is 100+100-50=150
        assert abs(iou - 50.0 / 150.0) < 1e-5

    def test_matching_perfect_prediction(self):
        evaluator = Evaluator(num_classes=3, iou_threshold=0.5)
        # pred: [x, y, w, l, rot]
        preds = {
            'boxes': torch.tensor([[0, 0, 10, 10, 0]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([0])
        }
        gts = np.array([[0, 0, 10, 10, 0, 0]]) # label 0 at the end
        evaluator.add_batch([preds], [gts])
        
        metrics = evaluator.compute_metrics()
        assert metrics['AP_class_0'] == 1.0
        assert metrics['Precision_class_0'] == 1.0
        assert metrics['Recall_class_0'] == 1.0
        
    def test_duplicate_predictions(self):
        evaluator = Evaluator(num_classes=3, iou_threshold=0.5)
        preds = {
            'boxes': torch.tensor([[0, 0, 10, 10, 0], [0, 0, 10, 10, 0]]),
            'scores': torch.tensor([0.9, 0.8]), # first one matches, second one is FP
            'labels': torch.tensor([0, 0])
        }
        gts = np.array([[0, 0, 10, 10, 0, 0]])
        evaluator.add_batch([preds], [gts])
        
        metrics = evaluator.compute_metrics()
        # AP for [TP, FP] with 1 GT is expected to be 1.0 using standard 11-point/AUC
        # Precision = 0.5, Recall = 1.0 at the end
        assert metrics['Precision_class_0'] == 0.5
        assert metrics['Recall_class_0'] == 1.0

    def test_wrong_class_prediction(self):
        evaluator = Evaluator(num_classes=3, iou_threshold=0.5)
        preds = {
            'boxes': torch.tensor([[0, 0, 10, 10, 0]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([1]) # wrong class
        }
        gts = np.array([[0, 0, 10, 10, 0, 0]])
        evaluator.add_batch([preds], [gts])
        
        metrics = evaluator.compute_metrics()
        assert metrics['AP_class_0'] == 0.0 # recall 0
        assert metrics['AP_class_1'] == 0.0 # precision 0 (FP)

    def test_ignore_region(self):
        evaluator = Evaluator(num_classes=3, iou_threshold=0.5)
        preds = {
            'boxes': torch.tensor([[0, 0, 10, 10, 0]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([0])
        }
        gts = np.empty((0, 6))
        ignores = np.array([[0, 0, 15, 15, 0]]) # overlap is high
        
        evaluator.add_batch([preds], [gts], [ignores])
        
        # the prediction should be ignored, meaning it doesn't count as FP
        metrics = evaluator.compute_metrics()
        # Prediction is ignored and therefore does not count as an FP.
        # However, with zero GT for this class AP is 0 rather than
        # an artificial perfect score of 1.
        assert metrics['AP_class_0'] == 0.0

    def test_edge_cases(self):
        # Empty everything
        evaluator = Evaluator()
        evaluator.add_batch([{'boxes': torch.empty((0,5)), 'scores': torch.empty(0), 'labels': torch.empty(0)}], [np.empty((0,6))])
        metrics = evaluator.compute_metrics()
        # With no GT-present classes, detection AP/mAP is undefined;
        # the evaluator's raw numeric representation is 0 rather than
        # an artificial perfect score.
        assert metrics['mAP@0.50'] == 0.0
        assert metrics['ECE'] == 0.0
        assert metrics['NLL'] == 0.0
        assert metrics['Brier'] == 0.0

        # GT only (False negative)
        evaluator = Evaluator(num_classes=1)
        evaluator.add_batch([{'boxes': torch.empty((0,5)), 'scores': torch.empty(0), 'labels': torch.empty(0)}], [np.array([[0,0,10,10,0,0]])])
        metrics = evaluator.compute_metrics()
        assert metrics['AP_class_0'] == 0.0
        assert metrics.get('Precision_class_0', 0.0) == 0.0
        assert metrics.get('Recall_class_0', 0.0) == 0.0
        assert metrics['ECE'] == 0.0
        assert metrics['NLL'] == 0.0
        
        # Pred only (False positive)
        evaluator = Evaluator(num_classes=1)
        preds = {
            'boxes': torch.tensor([[0, 0, 10, 10, 0]]),
            'scores': torch.tensor([0.9]),
            'labels': torch.tensor([0])
        }
        evaluator.add_batch([preds], [np.empty((0,6))])
        metrics = evaluator.compute_metrics()
        assert metrics['AP_class_0'] == 0.0
        assert metrics.get('Precision_class_0', 0.0) == 0.0
        assert metrics.get('Recall_class_0', 0.0) == 0.0
        assert metrics['ECE'] > 0.0 # Miscalibrated since pred is 0.9 but Acc is 0.0

    def test_global_aggregation_vs_per_frame_averaging(self):
        """
        Verify that aggregate mAP is computed globally, NOT by averaging per-frame AP.
        If we average per frame: Frame 1 (Empty GT, Empty Pred) -> AP=1.0. Frame 2 (1 GT, Empty Pred) -> AP=0.0. Average = 0.5.
        Global computation: 1 total GT, 0 TP -> Recall = 0.0 -> AP = 0.0.
        """
        evaluator = Evaluator(num_classes=1)
        
        # Frame 1: Empty GT, Empty Preds
        evaluator.add_batch([{'boxes': torch.empty((0,5)), 'scores': torch.empty(0), 'labels': torch.empty(0)}], [np.empty((0,6))])
        
        # Frame 2: 1 GT, Empty Preds
        evaluator.add_batch([{'boxes': torch.empty((0,5)), 'scores': torch.empty(0), 'labels': torch.empty(0)}], [np.array([[0,0,10,10,0,0]])])
        
        metrics = evaluator.compute_metrics()
        assert metrics['mAP@0.50'] == 0.0

    def test_calibration_metrics(self):
        evaluator = Evaluator(num_classes=1)
        # We need a predictable ECE/Brier
        preds = {
            'boxes': torch.tensor([[0,0,10,10,0], [20,20,10,10,0]]), # one matches, one doesn't
            'scores': torch.tensor([0.9, 0.9]),
            'labels': torch.tensor([0, 0])
        }
        gts = np.array([[0,0,10,10,0,0]])
        evaluator.add_batch([preds], [gts])
        
        metrics = evaluator.compute_metrics()
        # Acc = 0.5 for bin (0.8, 1.0], conf = 0.9. ECE = |0.5 - 0.9| = 0.4
        assert abs(metrics['ECE'] - 0.4) < 1e-4
        # Brier = ((0.9 - 1.0)^2 + (0.9 - 0.0)^2) / 2 = (0.01 + 0.81) / 2 = 0.41
        assert abs(metrics['Brier'] - 0.41) < 1e-4
        # NLL = (-1 * log(0.9) + (-1) * log(1 - 0.9)) / 2
        import math
        expected_nll = (-math.log(0.9) - math.log(0.1)) / 2.0
        assert abs(metrics['NLL'] - expected_nll) < 1e-4

class TestDecoderTask:
    def test_decoder_roundtrip(self):
        hm = torch.zeros((1, 3, 128, 128))
        offset = torch.zeros((1, 2, 128, 128))
        size = torch.zeros((1, 2, 128, 128))
        rot = torch.zeros((1, 2, 128, 128))
        
        # Place an object at grid (30, 40)
        # Class 0
        hm[0, 0, 40, 30] = 0.95
        
        # Offset
        offset[0, 0, 40, 30] = 0.2
        offset[0, 1, 40, 30] = 0.3
        
        # Size (width, length)
        size[0, 0, 40, 30] = 10.0
        size[0, 1, 40, 30] = 20.0
        
        # Rot (sin, cos) of 90 degrees (pi/2) -> (1, 0)
        rot[0, 0, 40, 30] = 1.0
        rot[0, 1, 40, 30] = 0.0
        
        preds = {
            "heatmap": hm,
            "offset": offset,
            "size": size,
            "rot": rot
        }
        
        results = CenterNetBEVHead.decode_detections(preds, downsample_stride=4)
        
        assert len(results) == 1
        res = results[0]
        
        boxes = res['boxes']
        scores = res['scores']
        labels = res['labels']
        
        assert len(boxes) == 1
        assert scores[0] == 0.95
        assert labels[0] == 0
        
        col_bev, row_bev, w, l, theta = boxes[0]
        # col_bev = (xs + dx) * stride = (30 + 0.2) * 4 = 120.8
        assert abs(col_bev.item() - 120.8) < 1e-4
        # row_bev = (ys + dy) * stride = (40 + 0.3) * 4 = 161.2
        assert abs(row_bev.item() - 161.2) < 1e-4
        # size
        assert abs(w.item() - 10.0) < 1e-4
        assert abs(l.item() - 20.0) < 1e-4
        
        # rot = atan2(1, 0) = pi / 2
        assert abs(theta.item() - math.pi/2) < 1e-4
