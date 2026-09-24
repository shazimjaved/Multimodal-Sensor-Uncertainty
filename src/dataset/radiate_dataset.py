"""
src/dataset/radiate_dataset.py
------------------------------
Deterministic PyTorch Dataset for RADIATE multimodal perception.
Provides lazy-loading, BEV rasterization, target encoding, and ignore-mask generation.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset

from radiate_fusion import RadiateCalib, load_annotations_for_frame
try:
    from src.preprocessing.bev_grid import BEVGridConfig, encode_bev_boxes
    from src.preprocessing.radar import RadarPreprocessor
    from src.preprocessing.lidar import LiDARPreprocessor
    from src.preprocessing.camera import CameraPreprocessor
except ModuleNotFoundError:
    from preprocessing.bev_grid import BEVGridConfig, encode_bev_boxes
    from preprocessing.radar import RadarPreprocessor
    from preprocessing.lidar import LiDARPreprocessor
    from preprocessing.camera import CameraPreprocessor
from .indexer import RadiateIndexer


class RadiateMultimodalDataset(Dataset):
    """
    PyTorch Dataset providing multimodal synchronized sensor inputs in a common BEV coordinate system.
    """

    TARGET_CLASSES = {
        'car': 0,
        'van': 1,
        'bus': 2
    }

    IGNORE_CLASSES = {
        'pedestrian',
        'group_of_pedestrians',
        'truck',
        'motorbike',
        'bicycle'
    }

    def __init__(
        self,
        indexer: RadiateIndexer,
        calib: RadiateCalib,
        bev_cfg: Optional[BEVGridConfig] = None,
        load_camera_image: bool = True
    ):
        self.indexer = indexer
        self.calib = calib
        self.bev_cfg = bev_cfg or BEVGridConfig()
        self.load_camera_image = load_camera_image

        # Preprocessors
        self.radar_proc = RadarPreprocessor(self.bev_cfg)
        self.lidar_proc = LiDARPreprocessor(self.calib, self.bev_cfg)
        self.camera_proc = CameraPreprocessor(self.calib, self.bev_cfg)

    def __len__(self) -> int:
        return len(self.indexer)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Lazily load and preprocess multimodal frame at given index.
        """
        record = self.indexer[idx]
        rf = record["radar_frame"]

        # 1. Load and process Radar
        radar_bev = self.radar_proc.process_file(record["radar_path"])

        # 2. Load and process LiDAR
        lidar_bev, pts_radar_filtered = self.lidar_proc.process_file(record["lidar_path"])

        # 3. Load and process Camera
        cam_img_norm, camera_bev, camera_bev_mask = self.camera_proc.process_file(
            record["cam_left_path"], pts_radar=pts_radar_filtered
        )

        # 4. Load Annotations and generate targets + ignore mask
        raw_anns = load_annotations_for_frame(record["ann_path"], rf)
        encoded_targets = encode_bev_boxes(
            annotations=raw_anns,
            bev_cfg=self.bev_cfg,
            target_classes=self.TARGET_CLASSES,
            ignore_classes=self.IGNORE_CLASSES
        )

        # 5. Convert to PyTorch tensors
        sample = {
            # Sensor BEV Rasters: (C, H_bev, W_bev)
            "radar_bev": torch.from_numpy(radar_bev),                      # (1, 512, 512) float32
            "lidar_bev": torch.from_numpy(lidar_bev),                      # (3, 512, 512) float32
            "camera_bev": torch.from_numpy(camera_bev),                    # (3, 512, 512) float32
            "camera_bev_mask": torch.from_numpy(camera_bev_mask),          # (1, 512, 512) float32

            # Camera Perspective View
            "camera_image": torch.from_numpy(cam_img_norm) if self.load_camera_image else None, # (3, 376, 672)

            # Ground-Truth Targets & Ignore Regions
            "target_boxes_metric": torch.from_numpy(encoded_targets["target_boxes_metric"]), # (N, 5) [x, y, w, l, rot]
            "target_boxes_bev": torch.from_numpy(encoded_targets["target_boxes_bev"]),       # (N, 4) [col, row, w, h]
            "target_labels": torch.from_numpy(encoded_targets["target_labels"]),             # (N,) int64
            "ignore_mask": torch.from_numpy(encoded_targets["ignore_mask"]).unsqueeze(0),    # (1, 512, 512) float32
            "ignore_boxes_metric": torch.from_numpy(encoded_targets["ignore_boxes_metric"]), # (M, 5)

            # Informational / Metadata
            "metadata": {
                "radar_frame": rf,
                "radar_time": record["radar_time"],
                "radar_path": record["radar_path"],
                "lidar_frame": record["lidar_frame"],
                "lidar_dt_ms": record["lidar_dt_ms"],
                "lidar_path": record["lidar_path"],
                "cam_left_frame": record["cam_left_frame"],
                "cam_left_dt_ms": record["cam_left_dt_ms"],
                "cam_left_path": record["cam_left_path"],
                "ann_path": record["ann_path"],
                "pts_radar_filtered": pts_radar_filtered,
                "target_class_names": encoded_targets["target_class_names"],
                "raw_annotations": raw_anns,
            }
        }

        return sample
