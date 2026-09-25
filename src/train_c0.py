import os
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import subprocess

from dataset.indexer import RadiateIndexer
from dataset.radiate_dataset import RadiateMultimodalDataset
from radiate_fusion import RadiateCalib
from models.multimodal_detector import BEVMultimodalDetector
from models.losses import CenterNetLoss, build_centernet_targets
from eval.evaluator import Evaluator
from models.centernet_head import CenterNetBEVHead
from preprocessing.bev_grid import BEVGridConfig

# Set fixed seed
SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch):
    out = {
        'radar_bev': torch.stack([b['radar_bev'] for b in batch]),
        'lidar_bev': torch.stack([b['lidar_bev'] for b in batch]),
        'camera_bev': torch.stack([b['camera_bev'] for b in batch]),
        'ignore_mask': torch.stack([b['ignore_mask'] for b in batch])
    }
    
    max_objs = max(b['target_labels'].shape[0] for b in batch)
    if max_objs == 0:
        max_objs = 1
        
    padded_labels = []
    padded_boxes_bev = []
    padded_boxes_metric = []
    
    for b in batch:
        num = b['target_labels'].shape[0]
        pad_len = max_objs - num
        
        if pad_len > 0:
            padded_labels.append(torch.cat([b['target_labels'], torch.full((pad_len,), -1, dtype=b['target_labels'].dtype)]))
            padded_boxes_bev.append(torch.cat([b['target_boxes_bev'], torch.zeros((pad_len, 4), dtype=b['target_boxes_bev'].dtype)]))
            padded_boxes_metric.append(torch.cat([b['target_boxes_metric'], torch.zeros((pad_len, 5), dtype=b['target_boxes_metric'].dtype)]))
        else:
            padded_labels.append(b['target_labels'])
            padded_boxes_bev.append(b['target_boxes_bev'])
            padded_boxes_metric.append(b['target_boxes_metric'])
            
    out['target_labels'] = torch.stack(padded_labels)
    out['target_boxes_bev'] = torch.stack(padded_boxes_bev)
    out['target_boxes_metric'] = torch.stack(padded_boxes_metric)
    out['ignore_boxes_metric'] = [b['ignore_boxes_metric'] for b in batch]
    
    return out

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Unknown"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run C0 Controlled Baseline Training")
    parser.add_argument("--data_dir", type=str, default="city_3_0", help="Path to city_3_0 dataset")
    parser.add_argument("--calib", type=str, default="config/default-calib.yaml", help="Path to default-calib.yaml")
    args = parser.parse_args()

    print("Initializing C0 Baseline Training...")
    
    output_dir = "outputs/c0_baseline"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Dataset Setup
    calib = RadiateCalib(args.calib)
    
    train_idx = RadiateIndexer(args.data_dir, split="train")
    val_idx = RadiateIndexer(args.data_dir, split="val")
    
    train_ds = RadiateMultimodalDataset(train_idx, calib)
    val_ds = RadiateMultimodalDataset(val_idx, calib)
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    
    # 2. Model Setup
    model = BEVMultimodalDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = CenterNetLoss()
    
    # 3. Save Configuration
    config = {
        "experiment": "C0_Controlled_Baseline",
        "dataset": "city_3_0",
        "train_split": "5-626",
        "val_split": "627-713",
        "seed": SEED,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "epochs": 10,
        "git_commit": get_git_commit()
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
        
    # 4. Training Loop
    epochs = 10
    best_map = -1.0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_map_050": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_ece": [],
        "val_nll": [],
        "val_brier": [],
        "peak_rss_mb": []
    }
    
    import psutil
    process = psutil.Process()
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        t0 = time.time()
        
        # Train phase
        model.train()
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(device)
                
            optimizer.zero_grad()
            
            outputs = model(batch)
            targets = build_centernet_targets(
                target_boxes_bev=batch["target_boxes_bev"],
                target_labels=batch["target_labels"],
                ignore_mask=batch["ignore_mask"],
                target_boxes_metric=batch["target_boxes_metric"],
                device=device
            )
            
            losses = loss_fn(outputs, targets)
            loss = losses["loss"]
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (i+1) % 5 == 0:
                print(f"  Step {i+1}/{len(train_loader)} - Loss: {loss.item():.4f}", flush=True)
                
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        evaluator = Evaluator(num_classes=3, iou_threshold=0.5)
        bev_cfg = BEVGridConfig()
        
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                    
                outputs = model(batch)
                targets = build_centernet_targets(
                    target_boxes_bev=batch["target_boxes_bev"],
                    target_labels=batch["target_labels"],
                    ignore_mask=batch["ignore_mask"],
                    target_boxes_metric=batch["target_boxes_metric"],
                    device=device
                )
                
                v_loss = loss_fn(outputs, targets)["loss"]
                val_loss += v_loss.item()
                
                # Decode detections
                decoded = CenterNetBEVHead.decode_detections(outputs, downsample_stride=4)
                
                # Extract GT targets dynamically per frame (ignoring padded -1s)
                batch_gts = []
                for b_idx in range(len(batch['target_labels'])):
                    gt_boxes_metric = batch['target_boxes_metric'][b_idx]
                    gt_labels = batch['target_labels'][b_idx]
                    
                    valid_mask = gt_labels != -1
                    if valid_mask.sum() > 0:
                        b_gt_boxes = gt_boxes_metric[valid_mask].cpu().numpy()
                        b_gt_labels = gt_labels[valid_mask].cpu().numpy()
                        b_gt_all = np.concatenate([b_gt_boxes, np.expand_dims(b_gt_labels, 1)], axis=1)
                        batch_gts.append(b_gt_all)
                    else:
                        batch_gts.append(np.empty((0, 6)))
                        
                # Format predictions
                batch_preds = []
                for det in decoded:
                    boxes_bev = det['boxes'].cpu().numpy()
                    scores = det['scores'].cpu().numpy()
                    labels = det['labels'].cpu().numpy()
                    
                    boxes_metric = np.zeros_like(boxes_bev)
                    if len(boxes_bev) > 0:
                        col = boxes_bev[:, 0]
                        row = boxes_bev[:, 1]
                        w_px = boxes_bev[:, 2]
                        l_px = boxes_bev[:, 3]
                        angle = boxes_bev[:, 4]
                        
                        # Use the evaluator's coordinate grid
                        # It's instantiated outside the loop as bev_cfg
                        x_m, y_m = bev_cfg.pixel_to_metric(col, row)
                        w_m = w_px * bev_cfg.dx
                        l_m = l_px * bev_cfg.dy
                        
                        boxes_metric[:, 0] = x_m
                        boxes_metric[:, 1] = y_m
                        boxes_metric[:, 2] = w_m
                        boxes_metric[:, 3] = l_m
                        boxes_metric[:, 4] = angle
                        
                    batch_preds.append({
                        'boxes': boxes_metric,
                        'scores': scores,
                        'labels': labels
                    })
                    
                # Extract ignore boxes
                batch_ignore_boxes = [ib.cpu().numpy() for ib in batch['ignore_boxes_metric']]
                    
                evaluator.add_batch(batch_preds, batch_gts, batch_ignore_boxes)
                
                # Diagnostic logging for the very first validation batch of the first epoch
                if epoch == 0 and not hasattr(evaluator, '_logged_diagnostics'):
                    print("\n--- VALIDATION DIAGNOSTIC AUDIT (BATCH 1) ---")
                    for b_idx in range(len(batch_preds)):
                        pb = batch_preds[b_idx]['boxes']
                        gb = batch_gts[b_idx]
                        ib = batch_ignore_boxes[b_idx]
                        print(f" Frame {b_idx}:")
                        print(f"   Predictions: {len(pb)}")
                        if len(pb) > 0:
                            print(f"     X_m range: [{pb[:,0].min():.1f}, {pb[:,0].max():.1f}]")
                            print(f"     Y_m range: [{pb[:,1].min():.1f}, {pb[:,1].max():.1f}]")
                        print(f"   GT boxes: {len(gb)}")
                        if len(gb) > 0:
                            print(f"     X_m range: [{gb[:,0].min():.1f}, {gb[:,0].max():.1f}]")
                            print(f"     Y_m range: [{gb[:,1].min():.1f}, {gb[:,1].max():.1f}]")
                        print(f"   Ignore boxes: {len(ib)}")
                    evaluator._logged_diagnostics = True
                    print("---------------------------------------------\n")
                
        avg_val_loss = val_loss / len(val_loader)
        metrics = evaluator.compute_metrics()
        
        map_050 = metrics.get('mAP@0.50', 0.0)
        precision = np.mean([metrics.get(f'Precision_class_{c}', 0.0) for c in range(3)])
        recall = np.mean([metrics.get(f'Recall_class_{c}', 0.0) for c in range(3)])
        f1 = np.mean([metrics.get(f'F1_class_{c}', 0.0) for c in range(3)])
        
        peak_rss = process.memory_info().rss / (1024 * 1024)
        
        t_epoch = time.time() - t0
        
        print(f"Epoch {epoch+1} done in {t_epoch:.1f}s - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | mAP: {map_050:.4f}")
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_map_050"].append(map_050)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)
        history["val_ece"].append(metrics.get("ECE", 0.0))
        history["val_nll"].append(metrics.get("NLL", 0.0))
        history["val_brier"].append(metrics.get("Brier", 0.0))
        history["peak_rss_mb"].append(peak_rss)
        
        # Save checkpoints
        torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_final.pth"))
        if map_050 > best_map:
            best_map = map_050
            torch.save(model.state_dict(), os.path.join(output_dir, "checkpoint_best.pth"))
            print("  -> Saved new best checkpoint!")
            
        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=4)
            
    print("C0 CONTROLLED BASELINE TRAINING COMPLETE")
    
if __name__ == "__main__":
    main()
