"""
radiate_fusion.py
-----------------
Calibration loading, sensor transforms, and projection utilities
for RADIATE multimodal fusion.

Coordinate conventions:
  Radar 3D frame:  x = lateral (right+), y = forward (range+), z = up
  Camera frame:    x = right, y = down, z = forward (optical axis)
  Extrinsic:       P_radar = R @ P_sensor + T  (sensor -> radar)
  Inverse:         P_cam   = R_cam @ (P_radar - T_cam)  (radar -> camera)
"""

import os
import re
import json
import numpy as np
import cv2
import yaml


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class RadiateCalib:
    """Loads and exposes RADIATE calibration parameters."""

    RADAR_M_PER_PX = 0.17361
    RADAR_IMAGE_SIZE = (1152, 1152)
    RADAR_CENTER_PX = (576, 576)

    def __init__(self, calib_path: str):
        with open(calib_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # LiDAR
        lc = cfg['lidar_calib']
        self.T_lidar = np.array(lc['T'], dtype=float)
        self.R_lidar_rod = np.array(lc['R'], dtype=float)
        self.R_lidar, _ = cv2.Rodrigues(self.R_lidar_rod)

        # Left camera
        cc = cfg['left_cam_calib']
        self.T_cam_left = np.array(cc['T'], dtype=float)
        self.R_cam_left_rod = np.array(cc['R'], dtype=float)
        self.R_cam_left, _ = cv2.Rodrigues(self.R_cam_left_rod)
        self.K_left = np.array([
            [cc['fx'], 0,        cc['cx']],
            [0,        cc['fy'], cc['cy']],
            [0,        0,        1       ]
        ], dtype=float)
        self.dist_left = np.array(
            [cc['k1'], cc['k2'], cc.get('p1', 0), cc.get('p2', 0), cc.get('k3', 0)],
            dtype=float
        )
        self.cam_left_res = tuple(cc['res'])  # (W, H)

    # ------------------------------------------------------------------
    # LiDAR transforms
    # ------------------------------------------------------------------

    def lidar_to_radar(self, pts_lidar: np.ndarray) -> np.ndarray:
        """
        Transform LiDAR point cloud to radar reference frame.

        Args:
            pts_lidar: (N, 3+) array — x,y,z in LiDAR frame, extra cols ignored.

        Returns:
            (N, 3) array in radar frame.
        """
        xyz = pts_lidar[:, :3]
        return (self.R_lidar @ xyz.T).T + self.T_lidar

    def radar_pts_to_bev_px(self, pts_radar: np.ndarray) -> np.ndarray:
        """
        Convert 3D radar-frame points to BEV pixel coordinates.

        Radar BEV convention:
          col = cx + x / m_per_px   (x = lateral right+)
          row = cy - y / m_per_px   (y = forward+, so forward = up in image)

        Args:
            pts_radar: (N, 3) array in radar metric frame.

        Returns:
            (N, 2) integer array of (col, row) pixel coordinates.
        """
        cx, cy = self.RADAR_CENTER_PX
        m = self.RADAR_M_PER_PX
        cols = (pts_radar[:, 0] / m + cx).astype(int)
        rows = (cy - pts_radar[:, 1] / m).astype(int)
        return np.stack([cols, rows], axis=1)

    # ------------------------------------------------------------------
    # Camera projection
    # ------------------------------------------------------------------

    def radar_3d_to_cam_left(self, pts_radar: np.ndarray) -> np.ndarray:
        """
        Transform 3D points from radar frame to left camera frame.
        Convention:  P_cam = R_cam @ (P_radar - T_cam)

        Args:
            pts_radar: (N, 3) array in radar metric frame.

        Returns:
            (N, 3) array in camera frame.
        """
        shifted = pts_radar - self.T_cam_left  # (N, 3)
        return (self.R_cam_left @ shifted.T).T

    def project_to_cam_left(self, pts_cam: np.ndarray):
        """
        Project 3D camera-frame points to 2D image coordinates (with distortion).

        Args:
            pts_cam: (N, 3) array in camera frame.

        Returns:
            uvs:    (M, 2) float array of (u, v) pixel coordinates (only in-front points).
            mask:   (N,) bool mask of which input points are in front of camera (Z>0).
        """
        mask = pts_cam[:, 2] > 0
        if not np.any(mask):
            W, H = self.cam_left_res
            return np.empty((0, 2), float), mask

        pts_f = pts_cam[mask]
        # Project with distortion using cv2
        rvec = np.zeros((3, 1), dtype=float)
        tvec = np.zeros((3, 1), dtype=float)
        uvs, _ = cv2.projectPoints(
            pts_f.reshape(-1, 1, 3),
            rvec, tvec,
            self.K_left, self.dist_left
        )
        return uvs.reshape(-1, 2), mask


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def bbox_to_radar_3d_corners(cx_px, cy_px, w_px, h_px,
                              z_bottom=0.0, z_top=1.5,
                              m_per_px=0.17361, img_center=(576, 576)):
    """
    Convert a radar BEV bounding box to 3D corners in radar metric frame.

    The 8 corners are at z_bottom and z_top of the rectangular footprint.

    Args:
        cx_px, cy_px: bbox center in radar BEV pixel coords.
        w_px, h_px:   bbox width/height in pixels.
        z_bottom:     ground-plane z (default 0m).
        z_top:        vehicle roof z (default 1.5m).

    Returns:
        (8, 3) array of 3D corner points in radar frame.
    """
    cx_m = (cx_px - img_center[0]) * m_per_px
    cy_m = (img_center[1] - cy_px) * m_per_px
    hw   = w_px * m_per_px / 2.0
    hh   = h_px * m_per_px / 2.0

    # 4 footprint corners (z=0)
    corners_2d = np.array([
        [cx_m - hw, cy_m - hh],
        [cx_m + hw, cy_m - hh],
        [cx_m + hw, cy_m + hh],
        [cx_m - hw, cy_m + hh],
    ])

    corners_3d = []
    for z in (z_bottom, z_top):
        for c in corners_2d:
            corners_3d.append([c[0], c[1], z])
    return np.array(corners_3d, dtype=float)


def project_bbox_to_camera(bbox_pos, calib: RadiateCalib,
                            z_bottom=0.0, z_top=1.5):
    """
    Project a radar BEV bounding box into the left camera image.

    Args:
        bbox_pos: [cx_px, cy_px, w_px, h_px] in radar BEV pixels.
        calib:    RadiateCalib instance.

    Returns:
        corners_uv: (8, 2) projected image coords of 3D box corners, or None if all behind camera.
        all_valid:  bool — True if all corners project inside the image frame.
    """
    cx_px, cy_px, w_px, h_px = bbox_pos
    corners_3d_radar = bbox_to_radar_3d_corners(
        cx_px, cy_px, w_px, h_px, z_bottom, z_top)

    corners_cam = calib.radar_3d_to_cam_left(corners_3d_radar)
    uvs, mask = calib.project_to_cam_left(corners_cam)

    if uvs.shape[0] == 0:
        return None, False

    W, H = calib.cam_left_res
    in_frame = ((uvs[:, 0] >= 0) & (uvs[:, 0] < W) &
                (uvs[:, 1] >= 0) & (uvs[:, 1] < H))
    all_valid = bool(np.all(in_frame))

    # Build full (8,2) array with invalid points marked as NaN
    uvs_full = np.full((8, 2), np.nan)
    uvs_full[mask] = uvs
    return uvs_full, all_valid


# ---------------------------------------------------------------------------
# Timestamp / frame utilities
# ---------------------------------------------------------------------------

def parse_timestamp_file(path: str) -> dict:
    """Return {frame_id: timestamp} dict from a RADIATE .txt timestamp file."""
    result = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'Frame:\s*(\d+)\s+Time:\s*([\d.]+)', line.strip())
            if m:
                result[int(m.group(1))] = float(m.group(2))
    return result


def find_nearest_frame(ts_target: float, ts_dict: dict):
    """Return (frame_id, timestamp, dt_seconds) of nearest frame in ts_dict."""
    best_fid, best_ts, best_dt = None, None, float('inf')
    for fid, ts in ts_dict.items():
        dt = abs(ts - ts_target)
        if dt < best_dt:
            best_dt = dt; best_fid = fid; best_ts = ts
    return best_fid, best_ts, best_dt


def lidar_abs_frame_to_csv(abs_frame_id: int, first_abs_frame: int) -> str:
    """Convert absolute LiDAR frame ID to local CSV filename."""
    local_idx = abs_frame_id - first_abs_frame + 1
    return '%06d.csv' % local_idx


def load_lidar_csv(path: str) -> np.ndarray:
    """Load a RADIATE LiDAR CSV file. Returns (N, 5) array: x,y,z,intensity,ring."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    rows.append([float(p) for p in parts[:5]])
                except ValueError:
                    pass
    return np.array(rows, dtype=float) if rows else np.zeros((0, 5), float)


def load_annotations_for_frame(ann_json_path: str, radar_frame_id: int):
    """
    Load annotations for a given radar frame (1-indexed).

    Returns list of dicts with keys: id, class_name, position, rotation.
    position = [cx_px, cy_px, w_px, h_px] in radar BEV pixels.
    """
    with open(ann_json_path) as f:
        data = json.load(f)

    result = []
    idx = radar_frame_id - 1   # 0-indexed
    for obj in data:
        b = obj['bboxes'][idx] if idx < len(obj['bboxes']) else []
        if isinstance(b, dict) and b.get('position'):
            result.append({
                'id': obj['id'],
                'class_name': obj['class_name'],
                'position': b['position'],       # [cx, cy, w, h] in px
                'rotation': b.get('rotation', 0)
            })
    return result
