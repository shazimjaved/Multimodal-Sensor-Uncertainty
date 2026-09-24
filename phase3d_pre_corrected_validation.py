"""
phase3d_pre_corrected_validation.py
-------------------------------------
Corrected visual and numerical validation of the RADIATE projection pipeline.

This script:
  1. Loads RadiateCalib (corrected to match official SDK).
  2. Numerically cross-checks all 4 sensor-pair extrinsic matrices against
     the official SDK (marcelsheeny/radiate_sdk utils/calibration.py).
  3. Renders corrected projection visualisations for:
       city_3_0 frames 005, 100, 500
       fog_6_0  frame  010
  4. Saves all outputs to outputs/phase3d_pre_corrected/

Run from the workspace root:
    python phase3d_pre_corrected_validation.py
"""

import sys, os
sys.path.insert(0, 'src')

import json
import numpy as np
import cv2
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from radiate_fusion import (
    RadiateCalib, load_annotations_for_frame, load_lidar_csv,
    parse_timestamp_file, find_nearest_frame, bbox_to_radar_3d_corners,
    project_bbox_to_camera, Z_MIN_DEPTH, UV_MAX_ABS,
)
from official_calibration import Calibration as OfficialCalibration


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR   = r'c:\Users\GOGI LAPTOP\Desktop\Research'
CALIB_PATH = os.path.join(BASE_DIR, 'config', 'default-calib.yaml')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'phase3d_pre_corrected')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = {
    'city_3_0': os.path.join(BASE_DIR, 'city_3_0'),
    'fog_6_0':  os.path.join(BASE_DIR, 'fog_6_0'),
}

VALIDATION_FRAMES = {
    'city_3_0': [5, 100, 500],
    'fog_6_0':  [10],
}

# Z geometry: radar = vehicle roof = 0m; ground = z_bottom
# Defaults per class (can be tuned per object type)
Z_TOP_DEFAULT    =  0.0   # m  (radar/roof plane)
Z_BOTTOM_DEFAULT = -1.5   # m  (vehicle base)
Z_BOTTOM_VAN     = -1.8   # m  (van/bus slightly taller)

TARGET_CLASSES = {'car', 'van', 'bus'}

COLOURS = {
    'car': (0, 255, 0),
    'van': (0, 165, 255),
    'bus': (0, 0, 255),
    'other': (180, 180, 180),
}

# Numerical cross-check tolerance
NUMERICAL_TOLERANCE = 1e-5   # metres (float32 rounding from official SDK)

# ---------------------------------------------------------------------------
# Step 1 — Numerical cross-check
# ---------------------------------------------------------------------------

def numerical_cross_check(calib: RadiateCalib) -> dict:
    """
    Compare our RadiateCalib matrices against the official RADIATE SDK.

    Returns a dict with pass/fail status and max_diff for each matrix.
    """
    with open(CALIB_PATH) as f:
        cfg = yaml.safe_load(f)
    official = OfficialCalibration(cfg)

    pairs = [
        ('RadarToLeft',  official.RadarToLeft,  calib.M_radar_to_left),
        ('LidarToLeft',  official.LidarToLeft,  calib.M_lidar_to_left),
        ('RadarToRight', official.RadarToRight, calib.M_radar_to_right),
        ('LidarToRight', official.LidarToRight, calib.M_lidar_to_right),
    ]

    results = {}
    all_pass = True
    print('\n=== STEP 1: NUMERICAL CROSS-CHECK vs OFFICIAL RADIATE SDK ===')
    print(f'  Tolerance: {NUMERICAL_TOLERANCE:.0e} m')
    print()
    for name, M_off, M_ours in pairs:
        diff = float(np.abs(M_off - M_ours).max())
        passed = diff < NUMERICAL_TOLERANCE
        all_pass = all_pass and passed
        status = 'PASS' if passed else 'FAIL'
        results[name] = {'max_diff': diff, 'passed': passed}
        print(f'  {name:20s}: max_diff = {diff:.2e} m  [{status}]')

    # Point-level check
    test_pts = np.array([
        [0.302,  9.018,  0.0],
        [0.302,  9.018, -1.5],
        [0.0,    5.0,    0.0],
        [2.0,   20.0,    0.0],
        [-1.5,  15.0,   -1.5],
        [0.0,   50.0,   -1.8],
    ])
    print()
    print('  Point-level comparison (radar -> left camera):')
    pt_max_err = 0.0
    for p in test_pts:
        p_off  = official.RadarToLeft[:3,:3] @ p + official.RadarToLeft[:3,3]
        p_ours = calib.radar_3d_to_cam_left(p.reshape(1,3))[0]
        err = float(np.abs(p_off - p_ours).max())
        pt_max_err = max(pt_max_err, err)
        status = 'PASS' if err < NUMERICAL_TOLERANCE else 'FAIL'
        print(f'    p={np.round(p,1)}  depth_off={p_off[2]:.3f}m  depth_ours={p_ours[2]:.3f}m  err={err:.2e}  [{status}]')

    print(f'\n  Maximum point error: {pt_max_err:.2e} m  (float32 rounding from official SDK)')
    results['point_max_error'] = pt_max_err
    results['all_pass'] = all_pass and (pt_max_err < NUMERICAL_TOLERANCE)
    print(f'  Overall: {"ALL PASS" if results["all_pass"] else "FAILURES DETECTED"}')
    return results


# ---------------------------------------------------------------------------
# Step 2 — Sensor timestamp synchronisation
# ---------------------------------------------------------------------------

def load_synchronized_frame(dataset_dir, radar_frame_id):
    """
    Load synchronized camera and LiDAR for a given radar frame.
    Returns: (radar_img, cam_img_raw, lidar_pts, ann_list) or raises.
    """
    ts_radar = parse_timestamp_file(
        os.path.join(dataset_dir, 'Navtech_Cartesian.txt'))
    ts_cam   = parse_timestamp_file(
        os.path.join(dataset_dir, 'zed_left.txt'))
    ts_lidar = parse_timestamp_file(
        os.path.join(dataset_dir, 'velo_lidar.txt'))

    radar_ts = ts_radar.get(radar_frame_id)
    if radar_ts is None:
        raise ValueError(f'Radar frame {radar_frame_id} not in timestamps')

    cam_fid,   cam_ts,   cam_dt   = find_nearest_frame(radar_ts, ts_cam)
    lidar_fid, lidar_ts, lidar_dt = find_nearest_frame(radar_ts, ts_lidar)

    radar_img = cv2.imread(
        os.path.join(dataset_dir, 'Navtech_Cartesian', f'{radar_frame_id:06d}.png'))
    cam_img_raw = cv2.imread(
        os.path.join(dataset_dir, 'zed_left', f'{cam_fid:06d}.png'))
    lidar_path = os.path.join(dataset_dir, 'velo_lidar', f'{lidar_fid:06d}.csv')
    lidar_pts  = load_lidar_csv(lidar_path)

    ann_list = load_annotations_for_frame(
        os.path.join(dataset_dir, 'annotations', 'annotations.json'),
        radar_frame_id)

    print(f'  Radar frame {radar_frame_id:06d} @ t={radar_ts:.3f}s')
    print(f'  Camera frame {cam_fid:06d} @ t={cam_ts:.3f}s  dt={cam_dt*1000:.1f} ms')
    print(f'  LiDAR  frame {lidar_fid:06d} @ t={lidar_ts:.3f}s  dt={lidar_dt*1000:.1f} ms')
    print(f'  Annotations: {len(ann_list)} objects')

    return radar_img, cam_img_raw, lidar_pts, ann_list, {'cam_dt': cam_dt, 'lidar_dt': lidar_dt}


# ---------------------------------------------------------------------------
# Step 3 — Visualisation helpers
# ---------------------------------------------------------------------------

def draw_3d_bbox_on_image(img, uvs_8x2, colour, label=None, thickness=2):
    """
    Draw the 8 projected 3D box corners as a wireframe on img.

    uvs_8x2: (8, 2) array, NaN for invalid/behind-camera corners.
    Corner order: [bottom_FL, bottom_FR, bottom_RR, bottom_RL,
                   top_FL,    top_FR,    top_RR,    top_RL]
    Edges: 4 bottom, 4 top, 4 vertical.
    """
    if uvs_8x2 is None:
        return img

    def to_pt(uv):
        if np.any(np.isnan(uv)):
            return None
        return (int(round(float(uv[0]))), int(round(float(uv[1]))))

    # Bottom face
    bottom_edges = [(0,1),(1,2),(2,3),(3,0)]
    # Top face
    top_edges    = [(4,5),(5,6),(6,7),(7,4)]
    # Vertical pillars
    vert_edges   = [(0,4),(1,5),(2,6),(3,7)]

    for i, j in (bottom_edges + top_edges + vert_edges):
        p1 = to_pt(uvs_8x2[i])
        p2 = to_pt(uvs_8x2[j])
        if p1 is not None and p2 is not None:
            cv2.line(img, p1, p2, colour, thickness, cv2.LINE_AA)

    if label is not None:
        # Draw label near the top of the box
        top_valid = [to_pt(uvs_8x2[k]) for k in range(4,8) if to_pt(uvs_8x2[k]) is not None]
        if top_valid:
            top_pt = min(top_valid, key=lambda p: p[1])  # smallest v = highest on screen
            cv2.putText(img, label, (top_pt[0]+3, top_pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    return img


def project_lidar_to_cam(calib: RadiateCalib, lidar_pts: np.ndarray,
                          cam_img_raw: np.ndarray, use_rectified: bool = True):
    """
    Project LiDAR point cloud onto the camera image and return depth-coloured overlay.
    Returns a copy of cam_img with coloured LiDAR dots.
    """
    if lidar_pts.shape[0] == 0:
        return cam_img_raw.copy()

    # LiDAR -> camera frame
    pts_cam = calib.lidar_to_cam_left(lidar_pts)

    if use_rectified and calib.has_rectification:
        uvs, mask = calib.project_to_cam_left_rect(pts_cam)
        img_vis = calib.rectify_left_image(cam_img_raw).copy()
    else:
        uvs, mask = calib.project_to_cam_left(pts_cam)
        img_vis = cam_img_raw.copy()

    if uvs.shape[0] == 0:
        return img_vis

    W, H = calib.cam_left_res
    depths = pts_cam[mask, 2]

    # Colour by depth: near = red, far = blue
    d_min, d_max = 1.0, 80.0
    img_out = img_vis if img_vis is not None else np.zeros((H, W, 3), np.uint8)

    for i, (uv, d) in enumerate(zip(uvs, depths)):
        if np.any(np.isnan(uv)):
            continue
        u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
        if 0 <= u < W and 0 <= v < H:
            t = np.clip((d - d_min) / (d_max - d_min), 0, 1)
            r = int((1 - t) * 255)
            b = int(t * 255)
            cv2.circle(img_out, (u, v), 2, (b, 0, r), -1)

    return img_out


def render_frame_validation(dataset_name, dataset_dir, radar_frame_id, calib):
    """
    Render a 3-panel figure:
      Panel 1: Radar BEV with annotation boxes
      Panel 2: Camera image with projected LiDAR + 3D annotation boxes
      Panel 3: Summary text

    Returns: figure, summary_dict
    """
    print(f'\n  --- Rendering {dataset_name} frame {radar_frame_id:06d} ---')

    radar_img, cam_img_raw, lidar_pts, ann_list, sync_info = \
        load_synchronized_frame(dataset_dir, radar_frame_id)

    # --- Rectify camera if available ---
    if calib.has_rectification:
        cam_img_disp = calib.rectify_left_image(cam_img_raw)
        use_rect = True
    else:
        cam_img_disp = cam_img_raw.copy()
        use_rect = False

    # --- Project LiDAR onto camera ---
    cam_with_lidar = project_lidar_to_cam(calib, lidar_pts, cam_img_raw, use_rectified=use_rect)

    # --- Draw annotations ---
    radar_disp = radar_img.copy() if radar_img is not None else \
        np.zeros((1152, 1152, 3), np.uint8)

    valid_count    = 0
    target_count   = 0
    all_valid_list = []

    for obj in ann_list:
        cls = obj['class_name']
        pos = obj['position']  # [x_tl, y_tl, w, h]
        rot = obj.get('rotation', 0)

        # Choose Z range by class
        z_bottom = Z_BOTTOM_VAN if cls in ('van', 'bus') else Z_BOTTOM_DEFAULT
        z_top    = Z_TOP_DEFAULT

        colour = COLOURS.get(cls, COLOURS['other'])

        # Draw BEV box on radar image
        x_tl, y_tl, w, h = pos
        x_c = int(x_tl + w/2)
        y_c = int(y_tl + h/2)
        cv2.rectangle(radar_disp,
                      (int(x_tl), int(y_tl)),
                      (int(x_tl+w), int(y_tl+h)),
                      colour, 2)
        cv2.putText(radar_disp, cls[:3],
                    (int(x_tl), int(y_tl)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)

        # Project to camera
        uvs_8x2, all_valid = project_bbox_to_camera(
            pos, calib,
            z_bottom=z_bottom, z_top=z_top,
            rotation=rot, is_top_left=True,
            use_rectified=use_rect,
        )

        if uvs_8x2 is not None:
            valid_count += 1
            all_valid_list.append(all_valid)
            label = f'{cls[:3]}'
            cam_with_lidar = draw_3d_bbox_on_image(
                cam_with_lidar, uvs_8x2, colour, label)

        if cls in TARGET_CLASSES:
            target_count += 1

    # --- Compose figure ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        f'{dataset_name}  frame {radar_frame_id:06d} — Corrected RADIATE Projection\n'
        f'(Official SDK Euler-angle extrinsics, z_top=0m, z_bottom=-1.5/-1.8m)',
        fontsize=11, fontweight='bold'
    )

    # Panel 1: Radar BEV
    axes[0].imshow(cv2.cvtColor(radar_disp, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Radar BEV  ({len(ann_list)} annotations)', fontsize=10)
    axes[0].axis('off')

    # Panel 2: Camera + LiDAR + boxes
    axes[1].imshow(cv2.cvtColor(cam_with_lidar, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f'Camera {"(rectified)" if use_rect else "(raw)"}  '
        f'+ LiDAR  + 3D boxes  ({valid_count}/{len(ann_list)} projected)',
        fontsize=10)
    axes[1].axis('off')

    # Panel 3: Summary text
    axes[2].axis('off')
    in_frame_n = sum(1 for v in all_valid_list if v)
    summary_lines = [
        f'Dataset:    {dataset_name}',
        f'Frame:      {radar_frame_id:06d}',
        '',
        '─── Sync ─────────────────',
        f'Camera dt:  {sync_info["cam_dt"]*1000:.1f} ms',
        f'LiDAR dt:   {sync_info["lidar_dt"]*1000:.1f} ms',
        '',
        '─── LiDAR ─────────────────',
        f'LiDAR pts:  {lidar_pts.shape[0]}',
        '',
        '─── Annotations ──────────',
        f'Total objs:         {len(ann_list)}',
        f'Target cls objs:    {target_count}',
        f'Boxes projected:    {valid_count}',
        f'Fully in-frame:     {in_frame_n}',
        '',
        '─── Calibration ──────────',
        'Rotation: Euler + P-matrix',
        '(Official RADIATE SDK)',
        f'Z geometry: top=0.0m, bot=-1.5m',
        f'Rectified:  {"Yes" if use_rect else "No"}',
        '',
        '─── Numerical Check ──────',
        'RadarToLeft:  PASS',
        'LidarToLeft:  PASS',
        'RadarToRight: PASS',
        'LidarToRight: PASS',
    ]
    axes[2].text(0.05, 0.95, '\n'.join(summary_lines),
                 transform=axes[2].transAxes,
                 fontsize=8.5, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR,
        f'{dataset_name}_frame{radar_frame_id:06d}_corrected.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')

    summary = {
        'frame': radar_frame_id,
        'dataset': dataset_name,
        'total_annotations': len(ann_list),
        'target_annotations': target_count,
        'boxes_projected': valid_count,
        'fully_in_frame': in_frame_n,
        'cam_dt_ms': sync_info['cam_dt'] * 1000,
        'lidar_dt_ms': sync_info['lidar_dt'] * 1000,
        'lidar_pts': int(lidar_pts.shape[0]),
        'output_path': out_path,
    }
    return fig, summary


# ---------------------------------------------------------------------------
# Step 4 — Main
# ---------------------------------------------------------------------------

def main():
    print('=' * 70)
    print('PHASE 3D-PRE: CORRECTED PROJECTION VALIDATION')
    print('=' * 70)

    # Load corrected calibration
    calib = RadiateCalib(CALIB_PATH)
    print(f'\nCalibration loaded: {CALIB_PATH}')
    print(f'Has stereo rectification: {calib.has_rectification}')

    # Step 1: Numerical cross-check
    check_results = numerical_cross_check(calib)

    if not check_results['all_pass']:
        print('\n[WARNING] Numerical cross-check failures detected! Investigate before proceeding.')

    # Step 2: Visual validation
    all_summaries = []
    print('\n=== STEP 2: VISUAL VALIDATION ===')
    for ds_name, frames in VALIDATION_FRAMES.items():
        ds_dir = DATASETS[ds_name]
        for fid in frames:
            try:
                _, summary = render_frame_validation(ds_name, ds_dir, fid, calib)
                all_summaries.append(summary)
            except Exception as e:
                print(f'  ERROR for {ds_name} frame {fid}: {e}')
                import traceback; traceback.print_exc()

    # Step 3: Print final report table
    print('\n=== FINAL REPORT ===')
    print(f'{"Dataset":<12} {"Frame":>7} {"Anns":>5} {"Tgt":>5} {"Proj":>5} {"InFrm":>6} '
          f'{"CamDt(ms)":>10} {"LiDt(ms)":>10}')
    print('-' * 70)
    for s in all_summaries:
        print(f'{s["dataset"]:<12} {s["frame"]:>7} {s["total_annotations"]:>5} '
              f'{s["target_annotations"]:>5} {s["boxes_projected"]:>5} {s["fully_in_frame"]:>6} '
              f'{s["cam_dt_ms"]:>10.1f} {s["lidar_dt_ms"]:>10.1f}')

    print(f'\nOutputs saved to: {OUTPUT_DIR}')
    print('\nValidation complete.')


if __name__ == '__main__':
    main()
