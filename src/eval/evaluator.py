"""
src/eval/evaluator.py
---------------------
Detection matching and metric evaluation for BEV Object Detection.

Implements class-aware matching, IoU calculation, precision, recall, mAP,
and calibration metrics (ECE, NLL, Brier score) based on prediction events.
"""

import math
from typing import Dict, List, Tuple
import numpy as np

def calculate_iou_2d(box1, box2):
    """
    Calculate 2D Intersection over Union (IoU) for axis-aligned bounding boxes.
    Assumes box = [x_center, y_center, width, length, angle]
    (Angle is ignored for simple axis-aligned approximation, which is 
     sufficient for matching if objects are mostly axis-aligned or IoU thresh is 0.5)
    """
    x1_c, y1_c, w1, l1, a1 = box1
    x2_c, y2_c, w2, l2, a2 = box2
    
    x1_min, x1_max = x1_c - w1/2, x1_c + w1/2
    y1_min, y1_max = y1_c - l1/2, y1_c + l1/2
    
    x2_min, x2_max = x2_c - w2/2, x2_c + w2/2
    y2_min, y2_max = y2_c - l2/2, y2_c + l2/2
    
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0
        
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = w1 * l1
    area2 = w2 * l2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
        
    return inter_area / union_area

def match_predictions(
    preds_boxes: np.ndarray,
    preds_scores: np.ndarray,
    preds_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
    ignore_boxes: np.ndarray = None
):
    """
    Match predictions to ground truth objects.
    Returns:
        matches: list of dicts with 'pred_idx', 'gt_idx', 'iou', 'score', 'matched', 'label'
    """
    num_preds = len(preds_boxes)
    num_gt = len(gt_boxes)
    
    # Sort predictions by confidence score descending
    sort_idx = np.argsort(-preds_scores)
    
    matches = []
    gt_matched = np.zeros(num_gt, dtype=bool)
    
    for p_idx in sort_idx:
        p_box = preds_boxes[p_idx]
        p_label = preds_labels[p_idx]
        p_score = preds_scores[p_idx]
        
        best_iou = 0.0
        best_gt_idx = -1
        
        for g_idx in range(num_gt):
            if gt_matched[g_idx] or gt_labels[g_idx] != p_label:
                continue
                
            iou = calculate_iou_2d(p_box, gt_boxes[g_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
                
        if best_iou >= iou_threshold:
            gt_matched[best_gt_idx] = True
            matches.append({
                'pred_idx': p_idx,
                'gt_idx': best_gt_idx,
                'iou': best_iou,
                'score': p_score,
                'matched': True,
                'label': p_label
            })
        else:
            # Check if it hits an ignore region
            ignored = False
            if ignore_boxes is not None and len(ignore_boxes) > 0:
                for ig_box in ignore_boxes:
                    iou = calculate_iou_2d(p_box, ig_box)
                    if iou > 0.0:  # Any overlap with ignore region
                        ignored = True
                        break
            
            if not ignored:
                matches.append({
                    'pred_idx': p_idx,
                    'gt_idx': -1,
                    'iou': best_iou,
                    'score': p_score,
                    'matched': False,
                    'label': p_label
                })
                
    return matches, gt_matched

class Evaluator:
    def __init__(self, num_classes=3, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        
        self.all_matches = []
        self.all_gt_counts = {c: 0 for c in range(num_classes)}
        
    def add_batch(self, preds_list, gt_list, ignore_list=None):
        if ignore_list is None:
            ignore_list = [None] * len(preds_list)
            
        for preds, gts, ignores in zip(preds_list, gt_list, ignore_list):
            if gts is not None and len(gts) > 0:
                gt_boxes = gts[:, :5]
                gt_labels = gts[:, 5]
                for lbl in gt_labels:
                    if lbl in self.all_gt_counts:
                        self.all_gt_counts[lbl] += 1
            else:
                gt_boxes = np.empty((0, 5))
                gt_labels = np.empty((0,))
                
            if preds is not None and len(preds['boxes']) > 0:
                p_boxes = preds['boxes'].cpu().numpy() if hasattr(preds['boxes'], 'cpu') else preds['boxes']
                p_scores = preds['scores'].cpu().numpy() if hasattr(preds['scores'], 'cpu') else preds['scores']
                p_labels = preds['labels'].cpu().numpy() if hasattr(preds['labels'], 'cpu') else preds['labels']
            else:
                p_boxes = np.empty((0, 5))
                p_scores = np.empty((0,))
                p_labels = np.empty((0,))
                
            matches, _ = match_predictions(
                p_boxes, p_scores, p_labels,
                gt_boxes, gt_labels,
                iou_threshold=self.iou_threshold,
                ignore_boxes=ignores
            )
            
            self.all_matches.extend(matches)
            
    def compute_metrics(self):
        """
        Computes AP, Precision, Recall, and Calibration metrics.
        
        Empty Frame Semantics:
        - If GT is empty and Predictions are empty (True Negative frame): AP = 1.0, Precision/Recall = 0.0.
        - If GT is empty and Predictions exist (False Positive frame): AP = 0.0, Precision/Recall = 0.0.
        - If GT exists and Predictions are empty (False Negative frame): AP = 0.0, Precision/Recall = 0.0.
        """
        # Sort all matches by confidence
        self.all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        metrics = {}
        aps = []
        
        for c in range(self.num_classes):
            c_matches = [m for m in self.all_matches if m['label'] == c]
            num_gt = self.all_gt_counts.get(c, 0)
            
            if num_gt == 0:
                if len(c_matches) == 0:
                    metrics[f'AP_class_{c}'] = 1.0 # True negative frame essentially
                    aps.append(1.0)
                else:
                    metrics[f'AP_class_{c}'] = 0.0
                    aps.append(0.0)
                continue
                
            if len(c_matches) == 0:
                metrics[f'AP_class_{c}'] = 0.0
                aps.append(0.0)
                continue
                
            tp = np.zeros(len(c_matches))
            fp = np.zeros(len(c_matches))
            
            for i, m in enumerate(c_matches):
                if m['matched']:
                    tp[i] = 1
                else:
                    fp[i] = 1
                    
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / num_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # AP calculation (11-point interpolation or AUC)
            # Standard VOC style AUC
            ap = self._calculate_ap(recalls, precisions)
            metrics[f'AP_class_{c}'] = ap
            aps.append(ap)
            
            # Save P, R, F1 at last point
            metrics[f'Precision_class_{c}'] = precisions[-1] if len(precisions) > 0 else 0.0
            metrics[f'Recall_class_{c}'] = recalls[-1] if len(recalls) > 0 else 0.0
            if precisions[-1] + recalls[-1] > 0:
                metrics[f'F1_class_{c}'] = 2 * (precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1])
            else:
                metrics[f'F1_class_{c}'] = 0.0
                
        metrics['mAP@0.50'] = np.mean(aps) if len(aps) > 0 else 0.0
        
        # Calibration Metrics (ECE, NLL, Brier)
        calib_metrics = self.compute_calibration()
        metrics.update(calib_metrics)
        
        return metrics
        
    def _calculate_ap(self, rec, prec):
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
        
    def compute_calibration(self, num_bins=10):
        """
        Computes calibration metrics (ECE, NLL, Brier score) pooled across all classes.
        
        Semantics:
        - p_hat: Max class confidence for each prediction (pooled).
        - y: 1 if matched to a ground truth (IoU >= threshold, correct class), else 0.
        - Empty set handling: If no predictions exist, returns 0.0 for all metrics to 
          prevent NaN, reflecting no measurable miscalibration.
        - Clipping: NLL applies a 1e-7 epsilon clip to avoid log(0).
        """
        if not self.all_matches:
            return {'ECE': 0.0, 'NLL': 0.0, 'Brier': 0.0}
            
        confs = np.array([m['score'] for m in self.all_matches])
        accs = np.array([1.0 if m['matched'] else 0.0 for m in self.all_matches])
        
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        bin_indices = np.digitize(confs, bins) - 1
        
        ece = 0.0
        n_total = len(confs)
        
        for i in range(num_bins):
            bin_mask = (bin_indices == i)
            if np.any(bin_mask):
                bin_acc = accs[bin_mask].mean()
                bin_conf = confs[bin_mask].mean()
                bin_count = bin_mask.sum()
                ece += (bin_count / n_total) * np.abs(bin_acc - bin_conf)
                
        # NLL
        eps = 1e-7
        confs_clip = np.clip(confs, eps, 1.0 - eps)
        nll = -np.mean(accs * np.log(confs_clip) + (1 - accs) * np.log(1 - confs_clip))
        
        # Brier
        brier = np.mean((confs - accs) ** 2)
        
        return {
            'ECE': ece,
            'NLL': nll,
            'Brier': brier
        }
