"""
src/generate_phase3b_debug_visuals.py
-------------------------------------
Generates 6-panel debug visualizations demonstrating:
- Radar BEV raster
- LiDAR BEV raster
- Front camera image
- Camera-projected BEV raster
- Multimodal fused alignment
- Target GT bboxes vs Ignore regions
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from radiate_fusion import RadiateCalib, project_bbox_to_camera
from src.preprocessing.bev_grid import BEVGridConfig
from src.dataset import RadiateIndexer, RadiateMultimodalDataset

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CALIB_PATH = os.path.join(BASE_DIR, 'config', 'default-calib.yaml')
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'phase3b')
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_COLORS = {
    0: {'name': 'car', 'rgb': (0.31, 0.80, 0.77), 'hex': '#4ECDC4'},
    1: {'name': 'van', 'rgb': (0.58, 0.88, 0.83), 'hex': '#95E1D3'},
    2: {'name': 'bus', 'rgb': (1.0, 0.42, 0.21), 'hex': '#FF6B35'},
}

BG_COLOR = '#0d0d1a'
FG_COLOR = '#e0e8ff'

def generate_visual(sample, out_filename, title_prefix='city_3_0'):
    meta = sample['metadata']
    rf = meta['radar_frame']
    
    radar_bev = sample['radar_bev'][0].numpy()       # (512, 512)
    lidar_bev = sample['lidar_bev'].numpy()          # (3, 512, 512): [height, density, intensity]
    camera_bev = sample['camera_bev'].numpy()        # (3, 512, 512)
    camera_mask = sample['camera_bev_mask'][0].numpy() # (512, 512)
    cam_img = sample['camera_image'].numpy().transpose(1, 2, 0) # (376, 672, 3)
    
    target_boxes = sample['target_boxes_bev'].numpy()  # (N, 4) [col, row, w, h]
    target_labels = sample['target_labels'].numpy()    # (N,)
    ignore_mask = sample['ignore_mask'][0].numpy()     # (512, 512)
    
    fig = plt.figure(figsize=(24, 15))
    fig.patch.set_facecolor(BG_COLOR)
    gs = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.15)
    
    ax_rad = fig.add_subplot(gs[0, 0])
    ax_lid = fig.add_subplot(gs[0, 1])
    ax_cam = fig.add_subplot(gs[0, 2])
    ax_cbv = fig.add_subplot(gs[1, 0])
    ax_fus = fig.add_subplot(gs[1, 1])
    ax_tgt = fig.add_subplot(gs[1, 2])
    
    # -------------------------------------------------------------
    # Panel 1: Radar BEV
    # -------------------------------------------------------------
    ax_rad.set_facecolor(BG_COLOR)
    ax_rad.imshow(radar_bev, cmap='magma', vmin=0, vmax=0.8)
    for b, l in zip(target_boxes, target_labels):
        col, row, w, h = b
        c_info = CLASS_COLORS.get(int(l), {'name': 'target', 'rgb': (1,1,0), 'hex': '#FFFF00'})
        rect = mpatches.Rectangle((col - w/2, row - h/2), w, h, linewidth=1.5,
                                  edgecolor=c_info['rgb'], facecolor='none', alpha=0.9)
        ax_rad.add_patch(rect)
    ax_rad.set_title(f'1. Radar BEV Raster (1, 512, 512)\nFrame {rf:06d} | t={meta["radar_time"]:.2f}s',
                     color=FG_COLOR, fontsize=11, pad=6)
    ax_rad.axis('off')
    
    # -------------------------------------------------------------
    # Panel 2: LiDAR BEV (Height + Density)
    # -------------------------------------------------------------
    ax_lid.set_facecolor(BG_COLOR)
    lid_rgb = np.stack([lidar_bev[0], lidar_bev[1], lidar_bev[2]], axis=-1)
    ax_lid.imshow(lid_rgb)
    for b, l in zip(target_boxes, target_labels):
        col, row, w, h = b
        c_info = CLASS_COLORS.get(int(l), {'name': 'target', 'rgb': (1,1,0), 'hex': '#FFFF00'})
        rect = mpatches.Rectangle((col - w/2, row - h/2), w, h, linewidth=1.2,
                                  edgecolor=c_info['rgb'], facecolor='none', alpha=0.8, ls='--')
        ax_lid.add_patch(rect)
    ax_lid.set_title(f'2. LiDAR BEV Raster (3, 512, 512)\nR=Height, G=Density, B=Intensity | dt={meta["lidar_dt_ms"]:.1f}ms',
                     color=FG_COLOR, fontsize=11, pad=6)
    ax_lid.axis('off')
    
    # -------------------------------------------------------------
    # Panel 3: Front Camera Perspective Image
    # -------------------------------------------------------------
    ax_cam.set_facecolor(BG_COLOR)
    ax_cam.imshow(np.clip(cam_img, 0, 1))
    ax_cam.set_title(f'3. Front ZED Camera Image (3, 376, 672)\nFrame {meta["cam_left_frame"]:06d} | dt={meta["cam_left_dt_ms"]:.1f}ms',
                     color=FG_COLOR, fontsize=11, pad=6)
    ax_cam.axis('off')
    
    # -------------------------------------------------------------
    # Panel 4: Camera-Projected BEV
    # -------------------------------------------------------------
    ax_cbv.set_facecolor(BG_COLOR)
    cam_bev_rgb = np.clip(camera_bev.transpose(1, 2, 0), 0, 1)
    ax_cbv.imshow(cam_bev_rgb)
    for b, l in zip(target_boxes, target_labels):
        col, row, w, h = b
        c_info = CLASS_COLORS.get(int(l), {'name': 'target', 'rgb': (1,1,0), 'hex': '#FFFF00'})
        rect = mpatches.Rectangle((col - w/2, row - h/2), w, h, linewidth=1.2,
                                  edgecolor=c_info['rgb'], facecolor='none', alpha=0.7)
        ax_cbv.add_patch(rect)
    ax_cbv.set_title(f'4. Camera BEV Projection (3, 512, 512)\nLiDAR Depth-Associated RGB | Active Px: {int(camera_mask.sum())}',
                     color=FG_COLOR, fontsize=11, pad=6)
    ax_cbv.axis('off')
    
    # -------------------------------------------------------------
    # Panel 5: Common Fused BEV Overlay
    # -------------------------------------------------------------
    ax_fus.set_facecolor(BG_COLOR)
    fused_rgb = np.zeros((512, 512, 3), dtype=np.float32)
    fused_rgb[:, :, 0] = radar_bev * 1.5                # Red channel = Radar
    fused_rgb[:, :, 1] = lidar_bev[1] * 1.2             # Green channel = LiDAR density
    fused_rgb[:, :, 2] = camera_bev.mean(axis=0) * 1.0  # Blue channel = Camera luminance
    ax_fus.imshow(np.clip(fused_rgb, 0, 1))
    for b, l in zip(target_boxes, target_labels):
        col, row, w, h = b
        c_info = CLASS_COLORS.get(int(l), {'name': 'target', 'rgb': (1,1,0), 'hex': '#FFFF00'})
        rect = mpatches.Rectangle((col - w/2, row - h/2), w, h, linewidth=1.5,
                                  edgecolor=c_info['rgb'], facecolor='none', alpha=0.9)
        ax_fus.add_patch(rect)
    ax_fus.set_title(f'5. Common BEV Multimodal Alignment\nR=Radar, G=LiDAR Density, B=Camera Lum',
                     color=FG_COLOR, fontsize=11, pad=6)
    ax_fus.axis('off')
    
    # -------------------------------------------------------------
    # Panel 6: Target GT BBoxes and Ignore Mask
    # -------------------------------------------------------------
    ax_tgt.set_facecolor(BG_COLOR)
    target_vis = np.zeros((512, 512, 3), dtype=np.float32)
    # Magenta overlay for ignore mask (pedestrians, bikes, trucks)
    target_vis[ignore_mask > 0, 0] = 0.8
    target_vis[ignore_mask > 0, 2] = 0.8
    ax_tgt.imshow(target_vis)
    
    for b, l in zip(target_boxes, target_labels):
        col, row, w, h = b
        c_info = CLASS_COLORS.get(int(l), {'name': 'target', 'rgb': (0,1,1), 'hex': '#00FFFF'})
        rect = mpatches.Rectangle((col - w/2, row - h/2), w, h, linewidth=2.0,
                                  edgecolor=c_info['rgb'], facecolor='none', alpha=1.0)
        ax_tgt.add_patch(rect)
        ax_tgt.text(col, row - h/2 - 4, f"{c_info['name']}",
                    color=c_info['rgb'], fontsize=8, ha='center', fontweight='bold')
        
    ax_tgt.set_title(f'6. Target BBoxes ({len(target_boxes)} cars/vans/buses)\nIgnore Regions ({int((ignore_mask>0).sum())} px) = Magenta Mask',
                     color='#00ff88', fontsize=11, pad=6)
    ax_tgt.axis('off')
    
    fig.suptitle(f'RADIATE {title_prefix} — Phase 3B Multimodal BEV Pipeline  |  Radar Frame {rf:06d}',
                 color='#a8c8ff', fontsize=14, fontweight='bold', y=0.98)
    
    save_path = os.path.join(OUT_DIR, out_filename)
    fig.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved debug visual: {save_path}')


def main():
    print('Generating Phase 3B Multimodal BEV Visualizations...')
    calib = RadiateCalib(CALIB_PATH)
    bev_cfg = BEVGridConfig(bev_height=512, bev_width=512)
    
    # 1. city_3_0 Early Train Frame 000005
    idx_train = RadiateIndexer(os.path.join(BASE_DIR, 'city_3_0'), split_range=(5, 536), exclude_frames=[1, 2, 3, 4])
    ds_train = RadiateMultimodalDataset(idx_train, calib, bev_cfg)
    generate_visual(ds_train[0], 'debug_bev_city3_0_train_frame_000005.png', 'city_3_0 Train')
    
    # 2. city_3_0 Mid Train Frame 000100
    idx_100 = next(i for i, r in enumerate(idx_train) if r['radar_frame'] == 100)
    generate_visual(ds_train[idx_100], 'debug_bev_city3_0_train_frame_000100.png', 'city_3_0 Train')
    
    # 3. city_3_0 Val Frame 000550
    idx_val = RadiateIndexer(os.path.join(BASE_DIR, 'city_3_0'), split_range=(537, 713), exclude_frames=[1, 2, 3, 4])
    ds_val = RadiateMultimodalDataset(idx_val, calib, bev_cfg)
    idx_550 = next(i for i, r in enumerate(idx_val) if r['radar_frame'] == 550)
    generate_visual(ds_val[idx_550], 'debug_bev_city3_0_val_frame_000550.png', 'city_3_0 Val')
    
    # 4. fog_6_0 Held-Out Test Frame 000010
    idx_test = RadiateIndexer(os.path.join(BASE_DIR, 'fog_6_0'), split_range=None, max_dt_sec=0.050)
    ds_test = RadiateMultimodalDataset(idx_test, calib, bev_cfg)
    idx_10 = next(i for i, r in enumerate(idx_test) if r['radar_frame'] == 10)
    generate_visual(ds_test[idx_10], 'debug_bev_fog6_0_test_frame_000010.png', 'fog_6_0 Test (Held-Out)')
    
    print('All Phase 3B debug visualizations generated successfully!')

if __name__ == '__main__':
    main()
