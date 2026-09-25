# PHASE 3B — Multimodal Dataset Loader & BEV Preprocessor Specification

**Phase**: 3B  
**Status**: COMPLETE & VERIFIED  
**Date**: September 24, 2026  
**Repository**: `shazimjaved/Multimodal-Sensor-Uncertainty`  
**Protocol Reference**: `docs/PHASE3A_EXPERIMENTAL_PROTOCOL.md`

---

## 1. Overview & Objective

Phase 3B implements the deterministic, CPU-compatible PyTorch data pipeline for the multimodal sensor uncertainty and calibration research project. In adherence to the locked Phase 3A experimental protocol:
- **No model training** is performed.
- **No synthetic degradation** is applied at this stage (Phase 3C will introduce degradation transforms).
- **`fog_6_0`** remains completely untouched and isolated as the held-out real-world test sequence.
- **Raw annotations** are preserved in full fidelity.
- **Memory efficiency**: Lazy per-sample decoding ensures the entire dataset is never loaded into RAM, guaranteeing smooth execution on CPU-only machines with $\le 8$ GB RAM.

---

## 2. Common Bird's-Eye-View (BEV) Coordinate System

All sensor modalities (Radar, LiDAR, Camera-derived BEV features) and ground-truth bounding boxes are mapped into a single, standardized, deterministic spatial grid.

| Parameter | Specification | Note |
|---|---|---|
| **Origin Reference Frame** | Radar physical sensor center $(0, 0, 0)$ | Forward: $+Y$, Lateral Right: $+X$, Up: $+Z$ |
| **Lateral Span ($X$)** | $[-50.0\text{ m}, +50.0\text{ m}]$ | Width = $100.0\text{ m}$ |
| **Longitudinal Span ($Y$)** | $[0.0\text{ m}, +100.0\text{ m}]$ | Forward reach = $100.0\text{ m}$ |
| **Grid Resolution** | $512 \times 512$ pixels | Raster height ($H$) = 512, width ($W$) = 512 |
| **Pixel Size ($\Delta x, \Delta y$)** | $\frac{100.0}{512} \approx 0.1953125\text{ m/pixel}$ | Isotropic spatial discretization |
| **Coordinate Mapping Formula** | $\text{col} = \frac{X - (-50.0)}{\Delta x}$<br>$\text{row} = 512 - 1 - \frac{Y - 0.0}{\Delta y}$ | Row 0 = $+100\text{ m}$ (far forward), Row 511 = $0\text{ m}$ (sensor origin) |

---

## 3. Sensor Preprocessing Pipelines

### 3.1 Navtech Radar Preprocessor (`src/preprocessing/radar.py`)
- **Raw Input**: Navtech Cartesian 8-bit grayscale PNG ($1152 \times 1152$ pixels, radar center at $(576, 576)$).
- **Physical Extent**: $100\text{ m}$ radius ($0.17361\text{ m/pixel}$).
- **Cropping**: Forward semicircle quadrant covering $Y \in [0, 100]\text{ m}$ and $X \in [-50, 50]\text{ m}$:
  $$\text{Row} \in [0, 576], \quad \text{Col} \in [288, 864]$$
- **Rasterization**: Resized via bilinear interpolation to $(512, 512)$ and normalized to $[0.0, 1.0]$.
- **Output Tensor**: `radar_bev` of shape `torch.Size([1, 512, 512])` (float32).

### 3.2 Velodyne LiDAR Preprocessor (`src/preprocessing/lidar.py`)
- **Raw Input**: 32-ring Velodyne point cloud from CSV (`x, y, z, intensity, ring`).
- **Rigid Transformation**: Points transformed from LiDAR frame to Radar coordinate frame:
  $$\mathbf{p}_{\text{radar}} = R_{\text{lidar}\to\text{radar}} \mathbf{p}_{\text{lidar}} + \mathbf{t}_{\text{lidar}\to\text{radar}}$$
- **Spatial Filtering**: Retain points within $X \in [-50, 50]\text{ m}$, $Y \in [0, 100]\text{ m}$, $Z \in [-3.0, 5.0]\text{ m}$.
- **Feature Aggregation**:
  - **Channel 0 (Height)**: Maximum $Z$ per cell, normalized by $\frac{Z - Z_{\min}}{Z_{\max} - Z_{\min}} \in [0, 1]$.
  - **Channel 1 (Point Density)**: Count of points per cell, log-transformed: $\frac{\log(1 + N)}{\log(1 + 64)} \in [0, 1]$.
  - **Channel 2 (Intensity)**: Mean reflectance intensity per cell, normalized by $\frac{I}{255.0} \in [0, 1]$.
- **Output Tensor**: `lidar_bev` of shape `torch.Size([3, 512, 512])` (float32).

### 3.3 ZED Front Camera Preprocessor (`src/preprocessing/camera.py`)
- **Depth Association (No Learned Networks)**:
  - In strict compliance with Phase 3A, depth is obtained deterministically from the synchronized LiDAR point cloud using the validated RADIATE extrinsic and intrinsic calibration parameters:
    $$\mathbf{p}_{\text{cam}} = R_{\text{radar}\to\text{cam}} \mathbf{p}_{\text{radar}} + \mathbf{t}_{\text{radar}\to\text{cam}}$$
    $$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_{\text{cam}}} K_{\text{cam}} \mathbf{p}_{\text{cam}}$$
  - Only points with $Z_{\text{cam}} > 0.5\text{ m}$ falling within the left camera dimensions ($376 \times 672$) are retained.
  - RGB values sampled at $(u, v)$ are painted back to the point's corresponding $(X, Y)$ cell in the common BEV grid.
  - A small $3 \times 3$ morphological dilation kernel densifies sparse beam traces into coherent BEV footprints.
- **Missing Depth Handling**: Cells without projected LiDAR points are marked 0 in `camera_bev_mask`.
- **Output Tensors**:
  - `camera_bev`: `torch.Size([3, 512, 512])` (float32, $[0, 1]$ RGB).
  - `camera_bev_mask`: `torch.Size([1, 512, 512])` (float32, $\{0.0, 1.0\}$ validity mask).
  - `camera_image`: `torch.Size([3, 376, 672])` (float32, $[0, 1]$ perspective RGB).

---

## 4. Ground-Truth Target Encoding & Ignore Masks

### 4.1 Target Classes vs. Ignore Classes
To avoid false-negative training penalties when non-vehicle objects (pedestrians, bicycles, etc.) are detected, objects are partitioned strictly as specified in Phase 3A:

| Category | Classes | Encoded Representation |
|---|---|---|
| **Primary Detection Targets** | `car` (0), `van` (1), `bus` (2) | `target_boxes_metric` $[N, 5]$: $[x, y, w, l, \theta]$<br>`target_boxes_bev` $[N, 4]$: $[\text{col}, \text{row}, w_{\text{px}}, h_{\text{px}}]$<br>`target_labels` $[N]$: $\{0, 1, 2\}$ |
| **Ignored Objects (Non-Background)** | `pedestrian`, `group_of_pedestrians`, `truck`, `motorbike`, `bicycle` | `ignore_mask` $[1, 512, 512]$ (binary float32 $\{0.0, 1.0\}$) covering object BEV bounding footprints |
| **Background** | Free space, static road structures | Unmasked space where `ignore_mask == 0` and no target box exists |

### 4.2 Raw Annotation Preservation
All original annotations from `annotations.json` are retained uncompressed under `raw_annotations` in each sample dictionary, preserving ID tracking, 3D metric coordinates, and raw class strings.

---

## 5. Dataset Indexing & Synchronization (`src/dataset/indexer.py`)

- **Index Anchor**: Each radar frame serves as the temporal reference anchor.
- **Nearest-Neighbor Timestamp Matching**: LiDAR, left-camera, and right-camera frames are matched by finding the minimum absolute temporal difference $|\Delta t| = |t_{\text{sensor}} - t_{\text{radar}}|$.
- **Synchronization Threshold**: Maximum allowed temporal offset is $\Delta t_{\max} = 0.050\text{ s}$ ($50\text{ ms}$). Any frame exceeding this is discarded and logged.
- **Startup Frame Filtering**: `city_3_0` frames `000001`–`000004` are systematically excluded due to camera initialization delay ($> 300\text{ ms}$ temporal offset).
- **Usable Frame Counts**:
  - `city_3_0` **TRAIN** (frames `000005`–`000626`): **622 frames** (Mean $|\Delta t_{\text{lidar}}| \approx 13.5\text{ ms}$, Mean $|\Delta t_{\text{cam}}| \approx 14.1\text{ ms}$).
  - `city_3_0` **VAL** (frames `000627`–`000713`): **87 frames** (Mean $|\Delta t_{\text{lidar}}| \approx 13.9\text{ ms}$, Mean $|\Delta t_{\text{cam}}| \approx 13.8\text{ ms}$).
  - `city_3_0` **Total Usable**: **709 frames** ($622 + 87$).
  - `fog_6_0` **HELD-OUT TEST**: **711 synchronized frames** ($\le 50\text{ ms}$), comprising **657 target-annotated frames** (Primary Detection/Calibration Evaluation Benchmark) and **54 empty roadway frames** (False-Alarm Diagnostic Set). Startup frames `000001`–`000003` excluded due to camera latency ($> 50\text{ ms}$). Full audit in [`docs/PHASE3B_MODALITY_DEPENDENCY_AUDIT.md`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/docs/PHASE3B_MODALITY_DEPENDENCY_AUDIT.md).

---

## 6. Verification and Empirical Checks

### 6.1 Sample Verification Across Splits

Four representative frames were evaluated end-to-end to verify data shapes, synchronization, bounding box consistency, and ignore mask activation:

| Split | Sequence | Radar Frame | $\Delta t_{\text{lidar}}$ | $\Delta t_{\text{cam}}$ | Target Boxes | Ignore Regions | Status |
|---|---|---|---|---|---|---|---|
| **Early Train** | `city_3_0` | `000005` | $0.007\text{ s}$ | $0.012\text{ s}$ | 1 (`bus`) | 0 | Validated |
| **Middle Train**| `city_3_0` | `000100` | $0.016\text{ s}$ | $0.009\text{ s}$ | 1 (`car`) | 1 (`pedestrian`) | Validated |
| **Validation**  | `city_3_0` | `000550` | $0.010\text{ s}$ | $0.013\text{ s}$ | 1 (`car`) | 0 | Validated |
| **Held-out Test**| `fog_6_0`  | `000010` | $0.014\text{ s}$ | $0.013\text{ s}$ | 0 | 0 | Validated |

### 6.2 Test Suite Execution
The automated test suite (`tests/test_dataset_pipeline.py` and `tests/test_calibration.py`) verified:
- Exact train/val frame counts ($622$ and $87$) and zero frame overlap.
- Startup frames `000001`–`000004` exclusion.
- All temporal deltas $\le 50\text{ ms}$ with strictly monotonic timestamps.
- Exact tensor shapes and ranges ($[0, 1]$ float32).
- Class filtering: target labels strictly $\in \{0, 1, 2\}$, ignore classes generating binary masks.
- Bitwise determinism across repeated accesses.
- Complete isolation of `fog_6_0`.

**Result**: **46 out of 46 unit tests passed** in $5.64\text{ s}$.

---

## 7. Artifacts and Visual Debug Panels

Debug visualization panels showing Radar BEV, LiDAR BEV, Camera RGB, PointPainted Camera BEV, Multimodal Fused Overlays, Target Bounding Boxes, and Ignore Masks were generated and saved to:
- `outputs/phase3b/debug_bev_city3_0_train_frame_000005.png`
- `outputs/phase3b/debug_bev_city3_0_train_frame_000100.png`
- `outputs/phase3b/debug_bev_city3_0_val_frame_000550.png`
- `outputs/phase3b/debug_bev_fog6_0_test_frame_000010.png`

---

## 8. Summary of Pipeline Artifacts

```
Research/
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── bev_grid.py          # BEVGridConfig & bounding box/ignore mask rasterizer
│   │   ├── radar.py             # Radar front-quadrant cropping & normalization
│   │   ├── lidar.py             # LiDAR 3-channel BEV rasterization
│   │   └── camera.py            # LiDAR-depth calibrated camera point-painting
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── indexer.py           # Timestamp sync (<= 50ms) & frame exclusion logging
│   │   └── radiate_dataset.py   # Deterministic lazy-loading PyTorch Dataset
│   └── generate_phase3b_debug_visuals.py
├── tests/
│   ├── test_calibration.py      # Calibration math & projection tests (29 tests)
│   └── test_dataset_pipeline.py # Phase 3B pipeline tests (17 tests)
├── docs/
│   ├── PHASE3A_EXPERIMENTAL_PROTOCOL.md
│   └── PHASE3B_DATA_PIPELINE.md
└── outputs/
    └── phase3b/                 # 4 debug visualization figures
```
