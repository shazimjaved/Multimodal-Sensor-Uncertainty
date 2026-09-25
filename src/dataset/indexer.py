"""
src/dataset/indexer.py
----------------------
Deterministic indexing and nearest-neighbor timestamp synchronization
for RADIATE multimodal sequences.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from radiate_fusion import parse_timestamp_file, find_nearest_frame


class RadiateIndexer:
    """Indexes multimodal frames with nearest-neighbor synchronization and strict rejection criteria."""

    def __init__(
        self,
        dataset_dir: str,
        split: Optional[str] = None,
        split_range: Optional[Tuple[int, int]] = None,
        exclude_frames: Optional[List[int]] = None,
        max_dt_sec: float = 0.050,  # 50 ms maximum allowed temporal offset
        require_camera: bool = True,
        require_lidar: bool = True
    ):
        self.dataset_dir = dataset_dir
        if split == "train":
            split_range = (5, 626)
            exclude_frames = [1, 2, 3, 4]
        elif split in ("val", "validation"):
            split_range = (627, 713)
            exclude_frames = []
        self.split_range = split_range
        self.exclude_frames = set(exclude_frames or [])
        self.max_dt_sec = max_dt_sec
        self.require_camera = require_camera
        self.require_lidar = require_lidar

        self.index_records: List[Dict[str, Any]] = []
        self.rejection_log: Dict[int, str] = {}

        self._build_index()

    def _build_index(self):
        """Parse timestamp indexes, perform temporal matching, and filter frames."""
        radar_txt = os.path.join(self.dataset_dir, "Navtech_Cartesian.txt")
        lidar_txt = os.path.join(self.dataset_dir, "velo_lidar.txt")
        cam_l_txt = os.path.join(self.dataset_dir, "zed_left.txt")
        cam_r_txt = os.path.join(self.dataset_dir, "zed_right.txt")
        ann_path  = os.path.join(self.dataset_dir, "annotations", "annotations.json")

        if not os.path.exists(radar_txt):
            raise FileNotFoundError(f"Navtech_Cartesian.txt missing in {self.dataset_dir}")

        radar_ts = parse_timestamp_file(radar_txt)
        lidar_ts = parse_timestamp_file(lidar_txt) if os.path.exists(lidar_txt) else {}
        cam_l_ts = parse_timestamp_file(cam_l_txt) if os.path.exists(cam_l_txt) else {}
        cam_r_ts = parse_timestamp_file(cam_r_txt) if os.path.exists(cam_r_txt) else {}

        for rf, t_r in sorted(radar_ts.items()):
            # 1. Check explicit exclusion
            if rf in self.exclude_frames:
                self.rejection_log[rf] = "explicitly_excluded"
                continue

            # 2. Check split range
            if self.split_range is not None:
                start_f, end_f = self.split_range
                if not (start_f <= rf <= end_f):
                    self.rejection_log[rf] = f"outside_split_range [{start_f}, {end_f}]"
                    continue

            # 3. Match nearest LiDAR
            l_id, t_l, dt_l = find_nearest_frame(t_r, lidar_ts) if lidar_ts else (None, None, float('inf'))
            if self.require_lidar and (l_id is None or dt_l > self.max_dt_sec):
                dt_ms = dt_l * 1000 if dt_l != float('inf') else -1
                self.rejection_log[rf] = f"lidar_sync_exceeded (dt={dt_ms:.1f}ms > {self.max_dt_sec*1000:.0f}ms)"
                continue

            # 4. Match nearest Left Camera
            c_l_id, t_cl, dt_cl = find_nearest_frame(t_r, cam_l_ts) if cam_l_ts else (None, None, float('inf'))
            if self.require_camera and (c_l_id is None or dt_cl > self.max_dt_sec):
                dt_ms = dt_cl * 1000 if dt_cl != float('inf') else -1
                self.rejection_log[rf] = f"camera_left_sync_exceeded (dt={dt_ms:.1f}ms > {self.max_dt_sec*1000:.0f}ms)"
                continue

            # 5. Match nearest Right Camera (optional/informational)
            c_r_id, t_cr, dt_cr = find_nearest_frame(t_r, cam_r_ts) if cam_r_ts else (None, None, float('inf'))

            # Check relative file existence
            radar_path = os.path.join(self.dataset_dir, "Navtech_Cartesian", f"{rf:06d}.png")
            lidar_path = os.path.join(self.dataset_dir, "velo_lidar", f"{l_id:06d}.csv") if l_id else None
            cam_l_path = os.path.join(self.dataset_dir, "zed_left", f"{c_l_id:06d}.png") if c_l_id else None
            cam_r_path = os.path.join(self.dataset_dir, "zed_right", f"{c_r_id:06d}.png") if c_r_id else None

            record = {
                "radar_frame": rf,
                "radar_time": t_r,
                "radar_path": radar_path,
                "lidar_frame": l_id,
                "lidar_time": t_l,
                "lidar_dt_ms": dt_l * 1000 if dt_l != float('inf') else None,
                "lidar_path": lidar_path,
                "cam_left_frame": c_l_id,
                "cam_left_time": t_cl,
                "cam_left_dt_ms": dt_cl * 1000 if dt_cl != float('inf') else None,
                "cam_left_path": cam_l_path,
                "cam_right_frame": c_r_id,
                "cam_right_time": t_cr,
                "cam_right_dt_ms": dt_cr * 1000 if dt_cr != float('inf') else None,
                "cam_right_path": cam_r_path,
                "ann_path": ann_path,
            }
            self.index_records.append(record)

    def __len__(self) -> int:
        return len(self.index_records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.index_records[idx]

    def get_summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics on indexed and rejected frames."""
        return {
            "total_indexed": len(self.index_records),
            "total_rejected": len(self.rejection_log),
            "rejection_reasons": dict(self.rejection_log),
            "mean_lidar_dt_ms": float(np.mean([r["lidar_dt_ms"] for r in self.index_records])) if self.index_records else 0.0,
            "mean_cam_left_dt_ms": float(np.mean([r["cam_left_dt_ms"] for r in self.index_records])) if self.index_records else 0.0,
        }
