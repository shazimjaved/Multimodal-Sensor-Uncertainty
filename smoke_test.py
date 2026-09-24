"""
smoke_test.py
-------------
Phase 1C: RADIATE Multimodal Fusion Smoke Test

Tests calibration loading, LiDAR-to-radar transform, and camera projection
on tiny_foggy radar frames 5, 10, 16.

Usage:
    python smoke_test.py
"""

import os, sys, json, time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from radiate_fusion import (
    RadiateCalib, parse_timestamp_file, find_nearest_frame,
    lidar_abs_frame_to_csv, load_lidar_csv, load_annotations_for_frame,
    project_bbox_to_camera, bbox_to_radar_3d_corners
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, 'tiny_foggy')
CALIB_PATH  = os.path.join(BASE_DIR, 'config', 'default-calib.yaml')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs', 'smoke_test')
ANN_PATH    = os.path.join(DATASET_DIR, 'annotations', 'annotations.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Radar frames to test
TARGET_RADAR_FRAMES = [5, 10, 16]

# Class colours (BGR for OpenCV, RGB for matplotlib)
CLASS_COLORS = {
    'bus':  {'rgb': (1.0, 0.42, 0.21), 'hex': '#FF6B35'},
    'car':  {'rgb': (0.31, 0.80, 0.77), 'hex': '#4ECDC4'},
    'van':  {'rgb': (0.58, 0.88, 0.83), 'hex': '#95E1D3'},
    'default': {'rgb': (1.0, 1.0, 0.0), 'hex': '#FFFF00'},
}

VEHICLE_HEIGHT_M = 1.5   # average vehicle height for 3D bbox projection


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

def run_smoke_test():
    results = {
        'status': 'UNKNOWN',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'calibration': {},
        'frames': {}
    }

    # ------------------------------------------------------------------
    # 1. Load calibration
    # ------------------------------------------------------------------
    print('[1/5] Loading calibration...')
    calib = RadiateCalib(CALIB_PATH)

    results['calibration'] = {
        'T_lidar': calib.T_lidar.tolist(),
        'R_lidar_rodrigues': calib.R_lidar_rod.tolist(),
        'R_lidar_matrix': calib.R_lidar.tolist(),
        'T_cam_left': calib.T_cam_left.tolist(),
        'R_cam_left_rodrigues': calib.R_cam_left_rod.tolist(),
        'R_cam_left_matrix': calib.R_cam_left.tolist(),
        'K_left_fx': calib.K_left[0, 0],
        'K_left_fy': calib.K_left[1, 1],
        'K_left_cx': calib.K_left[0, 2],
        'K_left_cy': calib.K_left[1, 2],
        'R_lidar_det': float(np.linalg.det(calib.R_lidar)),
        'R_cam_det':   float(np.linalg.det(calib.R_cam_left)),
    }
    print('   R_lidar det=%.8f (should be 1.0)' % results['calibration']['R_lidar_det'])
    print('   R_cam   det=%.8f (should be 1.0)' % results['calibration']['R_cam_det'])

    # ------------------------------------------------------------------
    # 2. Load timestamp files
    # ------------------------------------------------------------------
    print('[2/5] Loading timestamps...')
    radar_ts = parse_timestamp_file(os.path.join(DATASET_DIR, 'Navtech_Cartesian.txt'))
    lidar_ts = parse_timestamp_file(os.path.join(DATASET_DIR, 'velo_lidar.txt'))
    cam_ts   = parse_timestamp_file(os.path.join(DATASET_DIR, 'zed_left.txt'))
    lidar_first_abs = min(lidar_ts.keys())
    print('   Radar frames: %d  LiDAR frames: %d  Cam frames: %d' % (
        len(radar_ts), len(lidar_ts), len(cam_ts)))

    # ------------------------------------------------------------------
    # 3. Process each frame
    # ------------------------------------------------------------------
    print('[3/5] Processing target frames...')

    all_pass = True
    for rf in TARGET_RADAR_FRAMES:
        print('\n--- Radar frame %d ---' % rf)
        frame_result = {'radar_frame': rf}

        r_ts = radar_ts[rf]

        # Find nearest LiDAR frame
        l_fid, l_ts, l_dt = find_nearest_frame(r_ts, lidar_ts)
        csv_name = lidar_abs_frame_to_csv(l_fid, lidar_first_abs)
        csv_path = os.path.join(DATASET_DIR, 'velo_lidar', csv_name)

        # Find nearest camera frame
        c_fid, c_ts, c_dt = find_nearest_frame(r_ts, cam_ts)
        cam_path = os.path.join(DATASET_DIR, 'zed_left', '%06d.png' % c_fid)

        frame_result['lidar_frame_abs']  = l_fid
        frame_result['lidar_csv_file']   = csv_name
        frame_result['lidar_dt_ms']      = round(l_dt * 1000, 2)
        frame_result['cam_frame']        = c_fid
        frame_result['cam_dt_ms']        = round(c_dt * 1000, 2)

        print('   LiDAR: abs=%d, file=%s, dt=%.1fms' % (l_fid, csv_name, l_dt*1000))
        print('   Camera: frame=%d, dt=%.1fms' % (c_fid, c_dt*1000))

        # ---- Load data -----------------------------------------------
        radar_img = cv2.imread(
            os.path.join(DATASET_DIR, 'Navtech_Cartesian', '%06d.png' % rf),
            cv2.IMREAD_GRAYSCALE)
        lidar_raw = load_lidar_csv(csv_path)
        cam_img   = cv2.cvtColor(
            cv2.imread(cam_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        anns      = load_annotations_for_frame(ANN_PATH, rf)

        frame_result['n_lidar_points_raw'] = len(lidar_raw)
        frame_result['n_annotations']      = len(anns)
        print('   LiDAR points: %d' % len(lidar_raw))
        print('   Annotations: %d' % len(anns))

        # ---- LiDAR → Radar transform ---------------------------------
        pts_radar = calib.lidar_to_radar(lidar_raw)

        # Filter to BEV visible range (±100m) and reasonable z
        mask = ((np.abs(pts_radar[:, 0]) < 100) &
                (pts_radar[:, 1] > 0) &         # forward only
                (pts_radar[:, 1] < 100) &
                (pts_radar[:, 2] > -2.0) &
                (pts_radar[:, 2] < 4.0))
        pts_radar_filtered = pts_radar[mask]
        frame_result['n_lidar_points_in_bev'] = int(np.sum(mask))
        print('   LiDAR pts in BEV after filter: %d' % np.sum(mask))

        # ---- Camera projection of annotations -----------------------
        proj_results = []
        any_proj_fail = False
        for ann in anns:
            pos = ann['position']  # [cx, cy, w, h] in radar BEV px
            uvs, all_valid = project_bbox_to_camera(pos, calib,
                                                     z_top=VEHICLE_HEIGHT_M)
            proj_status = 'FAIL'
            if uvs is not None:
                # Check if midpoint of visible corners is in image
                valid_uvs = uvs[~np.isnan(uvs[:, 0])]
                if len(valid_uvs) >= 4:
                    proj_status = 'PASS'
                elif len(valid_uvs) > 0:
                    proj_status = 'PARTIAL'
                else:
                    proj_status = 'FAIL'
                    any_proj_fail = True

            proj_results.append({
                'id': ann['id'],
                'class': ann['class_name'],
                'bbox_px': pos,
                'status': proj_status,
                'corners_projected': uvs.tolist() if uvs is not None else None
            })
            print('   Ann id=%d (%s): proj=%s' % (ann['id'], ann['class_name'], proj_status))

        frame_result['projections'] = proj_results
        frame_result['any_proj_fail'] = any_proj_fail

        if any_proj_fail:
            print('   WARNING: At least one annotation failed camera projection!')
            all_pass = False

        # ---- Generate visualization ----------------------------------
        fig = _make_frame_figure(
            rf, r_ts, radar_img, pts_radar_filtered, lidar_raw,
            cam_img, anns, proj_results, calib, l_dt, c_dt)

        out_path = os.path.join(OUTPUT_DIR, 'frame_%02d.png' % rf)
        fig.savefig(out_path, dpi=130, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print('   Saved: %s' % out_path)
        frame_result['output_file'] = out_path
        results['frames']['frame_%02d' % rf] = frame_result

    # ------------------------------------------------------------------
    # 4. Final verdict
    # ------------------------------------------------------------------
    results['status'] = 'PASS' if all_pass else 'FAIL'
    print('\n' + '=' * 50)
    print('SMOKE TEST STATUS: %s' % results['status'])
    print('=' * 50)

    # Save JSON summary
    summary_path = os.path.join(OUTPUT_DIR, 'smoke_test_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print('Summary written: %s' % summary_path)

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

BG_COLOR = '#0d0d1a'
FG_COLOR = '#e0e8ff'

def _make_frame_figure(rf, r_ts, radar_img, pts_bev, lidar_raw,
                       cam_img, anns, proj_results, calib: RadiateCalib,
                       l_dt, c_dt):
    """Generate 3-panel figure: radar BEV | LiDAR BEV | camera."""

    fig = plt.figure(figsize=(21, 7))
    fig.patch.set_facecolor(BG_COLOR)

    gs = fig.add_gridspec(1, 3, wspace=0.04)
    ax_radar = fig.add_subplot(gs[0, 0])
    ax_lidar = fig.add_subplot(gs[0, 1])
    ax_cam   = fig.add_subplot(gs[0, 2])

    # ---- Panel 1: Radar BEV + annotations -------------------------
    ax_radar.set_facecolor(BG_COLOR)
    ax_radar.imshow(radar_img, cmap='magma', vmin=0, vmax=220)

    for ann in anns:
        cx, cy, w, h = ann['position']
        rot = ann['rotation']
        cls = ann['class_name']
        color = CLASS_COLORS.get(cls, CLASS_COLORS['default'])['rgb']
        rect = mpatches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            linewidth=2.0, edgecolor=color, facecolor='none', alpha=0.9)
        ax_radar.add_patch(rect)
        ax_radar.text(cx, cy - h/2 - 6, '%s #%d' % (cls, ann['id']),
                      color=color, fontsize=7.5, ha='center', fontweight='bold')

    ax_radar.set_title('Radar BEV  ·  frame %d\n%d annotations  |  t=+%.3fs' % (
        rf, len(anns), r_ts % 100),
        color=FG_COLOR, fontsize=10, pad=5)
    ax_radar.axis('off')

    # ---- Panel 2: LiDAR in radar BEV coords -----------------------
    ax_lidar.set_facecolor(BG_COLOR)
    ax_lidar.set_xlim(0, 1151)
    ax_lidar.set_ylim(1151, 0)
    ax_lidar.set_aspect('equal')

    if len(pts_bev) > 0:
        # Convert to BEV pixels
        bev_px = calib.radar_pts_to_bev_px(pts_bev)
        # Colour by height (z)
        z_vals = pts_bev[:, 2]
        z_norm = np.clip((z_vals + 1.5) / 4.0, 0, 1)
        ax_lidar.scatter(bev_px[:, 0], bev_px[:, 1],
                         c=z_norm, cmap='plasma', s=0.4, alpha=0.6, linewidths=0)

    # Draw radar range rings
    for r_m in [25, 50, 75, 100]:
        r_px = r_m / calib.RADAR_M_PER_PX
        circle = plt.Circle((576, 576), r_px, color='#334466',
                             fill=False, lw=0.6, ls='--', alpha=0.6)
        ax_lidar.add_patch(circle)
        ax_lidar.text(576 + r_px, 576, '%dm' % r_m,
                      color='#5577aa', fontsize=5.5, va='center')

    # Overlay radar annotations on LiDAR panel too
    for ann in anns:
        cx, cy, w, h = ann['position']
        cls = ann['class_name']
        color = CLASS_COLORS.get(cls, CLASS_COLORS['default'])['rgb']
        rect = mpatches.Rectangle(
            (cx - w/2, cy - h/2), w, h,
            linewidth=1.5, edgecolor=color, facecolor='none', alpha=0.7, ls='--')
        ax_lidar.add_patch(rect)

    ax_lidar.set_title('LiDAR in Radar BEV  ·  dt=%.1fms\n%d pts (coloured by height)' % (
        l_dt*1000, len(pts_bev)),
        color=FG_COLOR, fontsize=10, pad=5)
    ax_lidar.axis('off')

    # ---- Panel 3: Camera + projected annotations ------------------
    ax_cam.set_facecolor(BG_COLOR)
    ax_cam.imshow(cam_img)

    W, H = calib.cam_left_res
    for pr in proj_results:
        cls = pr['class']
        color = CLASS_COLORS.get(cls, CLASS_COLORS['default'])['rgb']
        hexcol = CLASS_COLORS.get(cls, CLASS_COLORS['default'])['hex']
        uvs = pr['corners_projected']
        if uvs is None:
            continue
        uvs_arr = np.array(uvs)

        # Draw projected 3D box edges
        # Bottom face: corners 0-3, top face: corners 4-7
        bottom = uvs_arr[:4]  # z_bottom corners
        top    = uvs_arr[4:]  # z_top corners

        for face in (bottom, top):
            valid = ~np.isnan(face[:, 0])
            if valid.sum() >= 2:
                pts = face[valid].astype(int)
                for i in range(len(pts)):
                    ax_cam.plot([pts[i, 0], pts[(i+1) % len(pts), 0]],
                                [pts[i, 1], pts[(i+1) % len(pts), 1]],
                                color=color, lw=1.2, alpha=0.85)

        # Draw vertical edges
        for i in range(4):
            b = uvs_arr[i]; t = uvs_arr[i + 4]
            if not (np.isnan(b[0]) or np.isnan(t[0])):
                ax_cam.plot([int(b[0]), int(t[0])], [int(b[1]), int(t[1])],
                            color=color, lw=1.0, alpha=0.7)

        # Label using top-face centroid
        valid_top = top[~np.isnan(top[:, 0])]
        if len(valid_top) > 0:
            cx_uv = valid_top[:, 0].mean()
            cy_uv = valid_top[:, 1].min() - 6
            # Compute range from radar center
            cx_px, cy_px, _, _ = pr['bbox_px']
            rng_m = np.sqrt(
                ((cx_px - 576) * 0.17361) ** 2 +
                ((576 - cy_px) * 0.17361) ** 2)
            ax_cam.text(cx_uv, cy_uv,
                        '%s #%d\n%.0fm' % (cls, pr['id'], rng_m),
                        color=hexcol, fontsize=6.5, ha='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d0d1a',
                                  alpha=0.6, edgecolor='none'))

    status_color = '#00ff88' if not any(p['status'] == 'FAIL' for p in proj_results) else '#ff4444'
    status_str   = 'PASS' if status_color == '#00ff88' else 'FAIL'
    ax_cam.set_title('ZED Left Camera  ·  dt=%.1fms  [%s]' % (c_dt*1000, status_str),
                      color=status_color, fontsize=10, pad=5)
    ax_cam.axis('off')

    # Overall title
    fig.suptitle(
        'RADIATE tiny_foggy — Fusion Smoke Test  |  Radar Frame %d' % rf,
        color='#a8c8ff', fontsize=12, fontweight='bold', y=1.01)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    run_smoke_test()
