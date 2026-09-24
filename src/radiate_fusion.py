"""
radiate_fusion.py
-----------------
Calibration loading, sensor transforms, and projection utilities
for RADIATE multimodal fusion.

Coordinate conventions (official RADIATE SDK, marcelsheeny/radiate_sdk):
  Radar 3D frame:  x = lateral (right+), y = forward (range+), z = up+
  Camera frame:    x = right, y = down, z = forward (optical axis / depth)
  Shared world frame: Radar IS the origin (RadarT=[0,0,0], RadarR=[0,0,0]).

Calibration R fields are Euler angles in DEGREES [rx, ry, rz].
Calibration T fields are positions in METERS in the shared vehicle frame.

The extrinsic transform from sensor A to sensor B is:
    RelativeR = A_R - B_R      (Euler-angle subtraction)
    RelativeT = A_T - B_T      (position subtraction)
    M_4x4     = _make_transform(RelativeR, RelativeT)

where _make_transform builds:
    R = P @ Rx @ Ry @ Rz          (P = axis-permutation from sensor to camera convention)
    M = [[R, T], [0,0,0,1]]

Z geometry in radar frame:
  Radar is mounted on vehicle roof ≈ the Z=0 plane in the radar frame.
  The road surface is ≈ -1.8 m (depends on vehicle roof height, empirically ~1.5-2.0m).
  For bounding-box 3D lifting:
      z_top    = 0.0 m  (vehicle roof ≈ radar mounting height)
      z_bottom = -1.5 m (vehicle base, empirically derived; -1.8m for tall vehicles)
  These are defaults; callers may override per object class.

Projection validity:
  Points are valid iff Z_cam > Z_MIN_DEPTH (0.1 m) AND project inside image FOV.
  Points with very large projected u,v (blow-up) are clipped.
"""

import os
import re
import json
import numpy as np
import cv2
import yaml


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Minimum camera-frame depth to accept a projected point (avoids div-by-zero / blow-up)
Z_MIN_DEPTH = 0.1  # metres

# Maximum absolute pixel coordinate; points projecting beyond this are discarded
UV_MAX_ABS = 1e5


# ---------------------------------------------------------------------------
# Calibration helpers (official RADIATE SDK convention)
# ---------------------------------------------------------------------------

def _euler_to_rotation_matrix(euler_deg: np.ndarray) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from Euler angles [rx, ry, rz] in DEGREES.

    Convention matches official RADIATE SDK (marcelsheeny/radiate_sdk utils/calibration.py):
        R = P @ Rx @ Ry @ Rz
    where P = [[1,0,0],[0,0,1],[0,-1,0]] is the axis-permutation matrix that
    maps the shared vehicle/sensor frame to camera-convention (y-down, z-forward).
    """
    rx = np.deg2rad(float(euler_deg[0]))
    ry = np.deg2rad(float(euler_deg[1]))
    rz = np.deg2rad(float(euler_deg[2]))

    Rx = np.array([
        [1,           0,            0],
        [0,  np.cos(rx), -np.sin(rx)],
        [0,  np.sin(rx),  np.cos(rx)],
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1,          0],
        [-np.sin(ry), 0, np.cos(ry)],
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1],
    ], dtype=np.float64)

    # Axis permutation: (x, y, z) -> (x, z, -y)
    # Converts from vehicle/sensor convention (z-up) to camera convention (y-down, z-forward)
    P = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0],
    ], dtype=np.float64)

    return P @ Rx @ Ry @ Rz


def _make_transform(euler_deg: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Build a 4x4 homogeneous transform matrix from Euler angles and translation.

    Matches official RADIATE SDK transform() method exactly:
        M = [[R, T], [0,0,0,1]]
    where R = _euler_to_rotation_matrix(euler_deg) and T = translation.

    The transform acts as:  p_cam = M[:3,:3] @ p_src + M[:3, 3]

    Args:
        euler_deg:   [rx, ry, rz] Euler angles in degrees.
        translation: [tx, ty, tz] translation vector (A_T - B_T for A->B).

    Returns:
        (4, 4) float64 transform matrix.
    """
    R = _euler_to_rotation_matrix(euler_deg)
    T = np.asarray(translation, dtype=np.float64).flatten()
    M = np.eye(4, dtype=np.float64)
    # The official SDK builds [[R_rows, 0], [T, 1]].T, which stores R^T in [:3,:3].
    # Store R^T so that application  p_cam = M[:3,:3] @ p_src + M[:3,3]
    # is equivalent to  (P @ Rx @ Ry @ Rz)^T @ p_src + T.
    M[:3, :3] = R.T
    M[:3,  3] = T
    return M


# ---------------------------------------------------------------------------
# Calibration class
# ---------------------------------------------------------------------------

class RadiateCalib:
    """
    Loads RADIATE calibration parameters and provides sensor-to-sensor
    transforms and camera projection.

    All R fields in the YAML are Euler angles (degrees).
    All T fields are positions in metres in the shared vehicle/radar frame.
    The radar IS the origin: RadarR=[0,0,0], RadarT=[0,0,0].

    Pre-computed 4×4 extrinsic transform matrices (official SDK convention):
        self.M_radar_to_left   : radar frame  -> left camera frame
        self.M_radar_to_right  : radar frame  -> right camera frame
        self.M_lidar_to_left   : LiDAR frame  -> left camera frame
        self.M_lidar_to_right  : LiDAR frame  -> right camera frame
        self.M_lidar_to_radar  : LiDAR frame  -> radar frame  (derived)

    Transform usage (apply as):
        p_cam = M[:3,:3] @ p_src + M[:3,3]
    """

    RADAR_M_PER_PX  = 0.17361
    RADAR_IMAGE_SIZE = (1152, 1152)
    RADAR_CENTER_PX  = (576, 576)

    def __init__(self, calib_path: str):
        with open(calib_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # ---- Raw calibration parameters ----
        rc_cfg = cfg['radar_calib']
        lc_cfg = cfg['lidar_calib']
        cc_cfg = cfg['left_cam_calib']

        self.RadarR = np.array(rc_cfg['R'], dtype=np.float64)
        self.RadarT = np.array(rc_cfg['T'], dtype=np.float64)
        self.LidarR = np.array(lc_cfg['R'], dtype=np.float64)
        self.LidarT = np.array(lc_cfg['T'], dtype=np.float64)
        self.LeftR  = np.array(cc_cfg['R'], dtype=np.float64)
        self.LeftT  = np.array(cc_cfg['T'], dtype=np.float64)

        # ---- Left camera intrinsics ----
        self.K_left = np.array([
            [cc_cfg['fx'], 0,           cc_cfg['cx']],
            [0,            cc_cfg['fy'], cc_cfg['cy']],
            [0,            0,            1           ],
        ], dtype=np.float64)
        self.dist_left = np.array([
            cc_cfg['k1'], cc_cfg['k2'],
            cc_cfg.get('p1', 0.0), cc_cfg.get('p2', 0.0),
            cc_cfg.get('k3', 0.0),
        ], dtype=np.float64)
        self.cam_left_res = tuple(cc_cfg['res'])   # (W, H)

        # ---- Radar -> Left camera extrinsic (official SDK formula) ----
        self.M_radar_to_left = _make_transform(
            self.RadarR - self.LeftR,
            self.RadarT - self.LeftT,
        )

        # ---- LiDAR -> Left camera extrinsic ----
        self.M_lidar_to_left = _make_transform(
            self.LidarR - self.LeftR,
            self.LidarT - self.LeftT,
        )

        # ---- LiDAR -> Radar extrinsic ----
        # Derived by composing: p_radar = inv(M_radar_to_left) @ M_lidar_to_left @ p_lidar
        # Numerically: inv(M_radar_to_left) composed with M_lidar_to_left
        self.M_lidar_to_radar = (
            np.linalg.inv(self.M_radar_to_left) @ self.M_lidar_to_left
        )

        # ---- Stereo / right camera (optional) ----
        self.has_rectification = False
        if 'stereo_calib' in cfg and 'right_cam_calib' in cfg:
            rc = cfg['right_cam_calib']
            self.RightR = np.array(rc['R'], dtype=np.float64)
            self.RightT = np.array(rc['T'], dtype=np.float64)

            self.K_right = np.array([
                [rc['fx'], 0,        rc['cx']],
                [0,        rc['fy'], rc['cy']],
                [0,        0,        1       ],
            ], dtype=np.float64)
            self.dist_right = np.array([
                rc['k1'], rc['k2'],
                rc.get('p1', 0.0), rc.get('p2', 0.0),
                rc.get('k3', 0.0),
            ], dtype=np.float64)
            self.cam_right_res = tuple(rc['res'])

            self.M_radar_to_right = _make_transform(
                self.RadarR - self.RightR,
                self.RadarT - self.RightT,
            )
            self.M_lidar_to_right = _make_transform(
                self.LidarR - self.RightR,
                self.LidarT - self.RightT,
            )

            sc = cfg['stereo_calib']
            self.stereo_R = np.array(sc['R'], dtype=np.float64)
            self.stereo_T = np.array([sc['TX'], sc['TY'], sc['TZ']], dtype=np.float64)

            # Stereo rectification
            (self.R_rect_left, self.R_rect_right,
             self.P_rect_left,  self.P_rect_right,
             self.Q, self.roi_left, self.roi_right) = cv2.stereoRectify(
                cameraMatrix1=self.K_left,
                distCoeffs1=self.dist_left,
                cameraMatrix2=self.K_right,
                distCoeffs2=self.dist_right,
                imageSize=self.cam_left_res,
                R=self.stereo_R,
                T=self.stereo_T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0,
            )
            self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
                self.K_left, self.dist_left,
                self.R_rect_left, self.P_rect_left,
                self.cam_left_res, cv2.CV_32FC1,
            )
            self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
                self.K_right, self.dist_right,
                self.R_rect_right, self.P_rect_right,
                self.cam_right_res, cv2.CV_32FC1,
            )
            self.K_rect_left = self.P_rect_left[:3, :3]
            self.has_rectification = True

    # ------------------------------------------------------------------
    # Backward-compatibility aliases (attributes removed in Phase 3D-PRE refactor)
    # ------------------------------------------------------------------

    @property
    def R_lidar(self) -> np.ndarray:
        """Backward-compat alias: effective rotation matrix of LiDAR in radar frame (≈I)."""
        return _euler_to_rotation_matrix(self.LidarR - self.RadarR)

    @property
    def T_lidar(self) -> np.ndarray:
        """Backward-compat alias: LiDAR translation in radar frame (= LidarT - RadarT)."""
        return self.LidarT - self.RadarT

    @property
    def R_cam_left(self) -> np.ndarray:
        """Backward-compat alias: effective rotation of left camera in radar frame."""
        return _euler_to_rotation_matrix(self.LeftR - self.RadarR)

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_transform(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Apply a 4x4 homogeneous transform M to (N, 3) points.

        p_out = M[:3,:3] @ p_in + M[:3,3]

        Args:
            M:   (4, 4) float64 transform matrix.
            pts: (N, 3) float64 points in source frame.

        Returns:
            (N, 3) float64 points in destination frame.
        """
        pts = np.asarray(pts, dtype=np.float64)
        R = M[:3, :3]
        T = M[:3,  3]
        return (R @ pts.T).T + T

    # ------------------------------------------------------------------
    # LiDAR transforms
    # ------------------------------------------------------------------

    def lidar_to_radar(self, pts_lidar: np.ndarray) -> np.ndarray:
        """
        Transform LiDAR point cloud from LiDAR frame to radar frame.

        The transform is derived as inv(M_radar_to_left) @ M_lidar_to_left,
        consistent with the official RADIATE SDK calibration chain.

        Args:
            pts_lidar: (N, 3+) array — x,y,z in LiDAR frame, extra cols ignored.

        Returns:
            (N, 3) array in radar frame (metres).
        """
        return self._apply_transform(self.M_lidar_to_radar, pts_lidar[:, :3])

    def lidar_to_cam_left(self, pts_lidar: np.ndarray) -> np.ndarray:
        """
        Transform LiDAR points directly to left camera frame.

        Args:
            pts_lidar: (N, 3+) array in LiDAR frame.

        Returns:
            (N, 3) array in left camera frame.
        """
        return self._apply_transform(self.M_lidar_to_left, pts_lidar[:, :3])

    def radar_pts_to_bev_px(self, pts_radar: np.ndarray) -> np.ndarray:
        """
        Convert 3D radar-frame points to BEV pixel coordinates.

        Radar BEV convention:
          col = cx + x / m_per_px   (x = lateral right+)
          row = cy - y / m_per_px   (y = forward+, forward = up in image)

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

    def rectify_left_image(self, img_raw: np.ndarray) -> np.ndarray:
        """Rectify raw unrectified left camera image using stereo rectification maps."""
        if not self.has_rectification:
            return img_raw
        return cv2.remap(img_raw, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)

    def radar_3d_to_cam_left(self, pts_radar: np.ndarray) -> np.ndarray:
        """
        Transform 3D points from radar frame to left camera frame.

        Uses the official RADIATE SDK extrinsic:
            p_cam = M_radar_to_left[:3,:3] @ p_radar + M_radar_to_left[:3,3]

        Args:
            pts_radar: (N, 3) array in radar metric frame.

        Returns:
            (N, 3) array in left camera frame (x=right, y=down, z=depth/forward).
        """
        return self._apply_transform(self.M_radar_to_left, pts_radar)

    def project_to_cam_left(self, pts_cam: np.ndarray):
        """
        Project 3D camera-frame points to 2D image coordinates (unrectified).

        Validity filter: Z_cam > Z_MIN_DEPTH AND finite projected coordinates
        within UV_MAX_ABS. Returns coordinates for all points that pass;
        in-frame check is left to the caller.

        Args:
            pts_cam: (N, 3) array in camera frame.

        Returns:
            uvs:  (M, 2) float array of (u, v) pixel coords for valid points.
            mask: (N,)  bool mask selecting valid input points.
        """
        mask = (pts_cam[:, 2] > Z_MIN_DEPTH)
        if not np.any(mask):
            return np.empty((0, 2), np.float64), mask

        pts_f = pts_cam[mask]
        rvec = np.zeros((3, 1), np.float64)
        tvec = np.zeros((3, 1), np.float64)
        uvs, _ = cv2.projectPoints(
            pts_f.reshape(-1, 1, 3),
            rvec, tvec,
            self.K_left, self.dist_left,
        )
        uvs = uvs.reshape(-1, 2)

        # Clip blow-up
        finite = np.isfinite(uvs).all(axis=1) & (np.abs(uvs).max(axis=1) < UV_MAX_ABS)
        if not np.all(finite):
            full = np.full((pts_f.shape[0], 2), np.nan)
            full[finite] = uvs[finite]
            uvs = full

        return uvs, mask

    def project_to_cam_left_rect(self, pts_cam: np.ndarray):
        """
        Project 3D camera-frame points onto the RECTIFIED left camera image.

        Points are rotated into the rectified optical frame and projected with
        P_rect_left[:3,:3].  Zero distortion applies to the rectified image.

        Args:
            pts_cam: (N, 3) array in unrectified left camera frame.

        Returns:
            uvs:  (M, 2) float array of (u, v) pixel coords for valid points.
            mask: (N,)  bool mask selecting valid input points.
        """
        if not self.has_rectification:
            return self.project_to_cam_left(pts_cam)

        mask = (pts_cam[:, 2] > Z_MIN_DEPTH)
        if not np.any(mask):
            return np.empty((0, 2), np.float64), mask

        pts_f = pts_cam[mask]
        pts_rect = (self.R_rect_left @ pts_f.T).T
        proj = (self.K_rect_left @ pts_rect.T).T
        uvs = proj[:, :2] / proj[:, 2:3]

        # Clip blow-up
        finite = np.isfinite(uvs).all(axis=1) & (np.abs(uvs).max(axis=1) < UV_MAX_ABS)
        if not np.all(finite):
            full = np.full((pts_f.shape[0], 2), np.nan)
            full[finite] = uvs[finite]
            uvs = full

        return uvs, mask


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def bbox_to_radar_3d_corners(cx_px, cy_px, w_px, h_px,
                              z_bottom=-1.5, z_top=0.0,
                              rotation=0.0,
                              m_per_px=0.17361, img_center=(576, 576),
                              is_top_left=False):
    """
    Convert a radar BEV bounding box to 3D corners in radar metric frame.

    Z geometry (radar = vehicle roof = Z=0 origin):
        z_top    = 0.0 m  (default) — vehicle roof / radar mounting height
        z_bottom = -1.5 m (default) — vehicle base (≈1.5 m below roof)
    Callers can override z_bottom/z_top per object class (e.g. -1.8m for vans).

    The 8 corners are at z_bottom and z_top of the rectangular footprint.

    Args:
        cx_px, cy_px: bbox center (or top-left if is_top_left=True) in radar BEV pixel coords.
        w_px, h_px:   bbox width/height in pixels.
        z_bottom:     vehicle base z in radar frame (default -1.5 m, negative = below radar).
        z_top:        vehicle roof z in radar frame (default 0.0 m = radar plane).
        rotation:     rotation angle in degrees (counter-clockwise in radar metric frame).
        m_per_px:     metres per pixel (default 0.17361).
        img_center:   radar image center (default (576, 576)).
        is_top_left:  if True, cx_px/cy_px are upper-left corner coords (RADIATE format).

    Returns:
        (8, 3) float64 array of 3D corner points in radar frame.
    """
    if is_top_left:
        cx_px = cx_px + w_px / 2.0
        cy_px = cy_px + h_px / 2.0

    # Convert pixel centre to metric (radar-frame x,y)
    cx_m = (cx_px - img_center[0]) * m_per_px
    cy_m = (img_center[1] - cy_px) * m_per_px
    hw   = w_px * m_per_px / 2.0
    hh   = h_px * m_per_px / 2.0

    # 4 footprint corners relative to centre
    corners_2d = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ], dtype=np.float64)

    if rotation != 0.0:
        theta = np.deg2rad(float(rotation))
        R2 = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        corners_2d = (R2 @ corners_2d.T).T

    corners_2d[:, 0] += cx_m
    corners_2d[:, 1] += cy_m

    corners_3d = []
    for z in (z_bottom, z_top):
        for c in corners_2d:
            corners_3d.append([c[0], c[1], z])
    return np.array(corners_3d, dtype=np.float64)


def project_bbox_to_camera(bbox_pos, calib: RadiateCalib,
                            z_bottom=-1.5, z_top=0.0,
                            rotation=0.0, is_top_left=True,
                            use_rectified=True):
    """
    Project a radar BEV bounding box into the left camera image.

    Args:
        bbox_pos:      [x, y, w_px, h_px] in radar BEV pixels.
                       (x, y) = UPPER-LEFT pixel coords per RADIATE annotation format.
        calib:         RadiateCalib instance.
        z_bottom:      vehicle base z in radar frame (default -1.5 m).
        z_top:         vehicle roof z in radar frame (default 0.0 m).
        rotation:      rotation in degrees.
        is_top_left:   True if bbox_pos[:2] is upper-left corner (default True).
        use_rectified: True to project onto rectified image using P_rect_left (default True).

    Returns:
        corners_uv: (8, 2) projected image coords of 3D box corners (NaN for invalid pts).
                    None if ALL points are behind the camera.
        all_valid:  bool — True if all corners project inside the image frame.
    """
    p0, p1, w_px, h_px = bbox_pos[:4]
    corners_3d_radar = bbox_to_radar_3d_corners(
        p0, p1, w_px, h_px,
        z_bottom=z_bottom, z_top=z_top,
        rotation=rotation, is_top_left=is_top_left,
    )

    corners_cam = calib.radar_3d_to_cam_left(corners_3d_radar)
    if use_rectified and calib.has_rectification:
        uvs, mask = calib.project_to_cam_left_rect(corners_cam)
    else:
        uvs, mask = calib.project_to_cam_left(corners_cam)

    if uvs.shape[0] == 0:
        return None, False

    W, H = calib.cam_left_res
    # Only consider non-NaN uvs for the in-frame check
    finite_uvs = uvs[np.isfinite(uvs).all(axis=1)]
    if finite_uvs.shape[0] == 0:
        return None, False

    in_frame = ((finite_uvs[:, 0] >= 0) & (finite_uvs[:, 0] < W) &
                (finite_uvs[:, 1] >= 0) & (finite_uvs[:, 1] < H))
    all_valid = bool(np.all(in_frame)) and (finite_uvs.shape[0] == mask.sum())

    # Build full (8, 2) array with behind-camera points as NaN
    uvs_full = np.full((8, 2), np.nan)
    # mask is over the original 8 points; uvs are the valid subset
    valid_indices = np.where(mask)[0]
    for i, vi in enumerate(valid_indices):
        uvs_full[vi] = uvs[i]

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
    """Load a RADIATE LiDAR CSV file.  Returns (N, 5) array: x,y,z,intensity,ring."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                try:
                    rows.append([float(p) for p in parts[:5]])
                except ValueError:
                    pass
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 5), np.float64)


def load_annotations_for_frame(ann_json_path: str, radar_frame_id: int):
    """
    Load annotations for a given radar frame (1-indexed).

    Returns list of dicts with keys: id, class_name, position, rotation.
    position = [x_tl, y_tl, w_px, h_px] in radar BEV pixels (upper-left + size).
    """
    with open(ann_json_path) as f:
        data = json.load(f)

    result = []
    idx = radar_frame_id - 1   # convert to 0-indexed
    for obj in data:
        b = obj['bboxes'][idx] if idx < len(obj['bboxes']) else []
        if isinstance(b, dict) and b.get('position'):
            result.append({
                'id':         obj['id'],
                'class_name': obj['class_name'],
                'position':   b['position'],       # [x_tl, y_tl, w, h] in px
                'rotation':   b.get('rotation', 0),
            })
    return result
