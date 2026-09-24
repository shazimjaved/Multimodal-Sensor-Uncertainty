# PHASE 0 — Environment Inspection Report

**Research Direction:** Calibration Stress Testing of Multimodal Sensor Fusion under Progressive Sensor Degradation
**Dataset:** RADIATE `tiny_foggy` sample
**Date Recorded:** 2026-09-24
**Status:** Read-only inspection — no files modified, no models built.

---

## A. Current Environment Summary

| Property | Value |
|---|---|
| **Operating System** | Windows 10 (Build 10.0.26200) |
| **Python Version** | 3.11.0 |
| **Python Executable** | `C:\Program Files\Python311\python.exe` |
| **CPU** | Intel Core i-series (Family 6, Model 154 — 13th Gen Raptor Lake) |
| **CPU Logical Cores** | 12 |
| **Total RAM** | 7.6 GB |
| **Available RAM (at inspection)** | ~1.0 GB (system under moderate load) |
| **GPU** | Not detected via wmic |
| **CUDA Available** | No — `torch` built as `2.10.0+cpu` |
| **CUDA Build Version** | None (CPU-only PyTorch wheel) |
| **OpenCL** | YES via NVD3D11 — NVIDIA GPU may be present but not exposed to PyTorch |

> **Note on GPU:** OpenCV's build info shows `OpenCL: YES (NVD3D11)`, which suggests an NVIDIA GPU *may* exist on this machine, but PyTorch was installed as the CPU-only variant (`+cpu`). Verify with Device Manager or `nvidia-smi` in a terminal. If a GPU is present, reinstalling `torch` with the correct CUDA wheel is strongly recommended before model development.

---

## B. Dataset Path and Structure

**Root path:** `C:\Users\GOGI LAPTOP\Desktop\Research\tiny_foggy\`

```
tiny_foggy/
├── meta.json                     # Sequence metadata (see below)
├── annotations/
│   └── annotations.json          # 17 tracked objects, 714 frames each (~97 KB)
│
├── Navtech_Cartesian/            # Radar — Cartesian BEV images
│   └── 000001.png ... 000018.png   # 18 frames | 1152x1152 px | ~550 KB each
├── Navtech_Cartesian.txt         # Timestamps for each radar frame
│
├── Navtech_Polar/                # Radar — Polar representation images
│   └── 000001.png ... 000018.png   # 18 frames | 400x576 px | ~151 KB each
├── Navtech_Polar.txt             # Timestamps for each polar frame
│
├── velo_lidar/                   # Velodyne LiDAR point clouds (CSV)
│   └── 000001.csv ... 000060.csv   # 60 frames | ~23,660 pts/frame | 5 cols (x,y,z,intensity,ring)
├── velo_lidar.txt                # Timestamps (frames 18-77 of full sequence)
│
├── zed_left/                     # ZED stereo camera — left channel
│   └── 000001.png ... 000050.png   # 50 frames | 672x376 px | ~80-107 KB each
│
├── zed_right/                    # ZED stereo camera — right channel
│   └── 000001.png ... 000050.png   # 50 frames | 672x376 px | ~84-112 KB each
│
└── GPS_IMU_Twist/                # GPS, IMU, and wheel odometry
    └── 000001.txt ... 000250.txt   # 250 files | covariance matrices + quaternion + twist
```

### Frame Count Comparison

| Modality | Frames | Approx. Rate |
|---|---|---|
| Navtech Radar (Cartesian) | 18 | ~4.25 Hz |
| Navtech Radar (Polar) | 18 | ~4.25 Hz |
| Velodyne LiDAR | 60 | ~10 Hz |
| ZED Left Camera | 50 | ~10 Hz |
| ZED Right Camera | 50 | ~10 Hz |
| GPS/IMU/Twist | 250 | ~50 Hz |

> **Timing Note:** All modalities share a common Unix timestamp epoch (starting ~1574859771.7 s, i.e. 27 Nov 2019). The `.txt` timestamp files define the synchronisation mapping between modalities.

### meta.json Contents

```json
{
  "name": "fog_6_0",
  "type": "fog",
  "set": "test",
  "version": "1.0",
  "date_created": "Sat Feb 29 03:09:49 2020"
}
```

### Annotations Structure

- **Format:** JSON list of 17 tracked objects
- **Each object:** `{ "id": int, "class_name": str, "bboxes": [ { "position": [cx, cy, w, h], "rotation": float_deg }, ... ] }`
- **Frames annotated:** 714 per object (covers full sequence, not just this tiny split)
- **Coordinate system:** Pixel-space on the Navtech Cartesian radar image (1152x1152)
- **Classes observed:** includes `bus`, `car` — full class list requires iterating all 17 items

### LiDAR CSV Column Schema

| Col 0 | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| x (m) | y (m) | z (m) | intensity | ring index |

Sample row: `-0.45903, -0.28572, -0.075989, 0, 17`  
~23,660 points per frame.

### GPS/IMU/Twist Format

Each `.txt` file contains a flattened 3x3 covariance matrix (position), quaternion (orientation), linear velocity, and angular velocity (9 rows, 3 values each).

---

## C. Available Dependencies

| Package | Version | Notes |
|---|---|---|
| `torch` | 2.10.0+cpu | CPU-only build |
| `torchvision` | 0.25.0 | Matched to torch |
| `numpy` | 2.4.2 | Latest; breaking API changes vs 1.x |
| `pandas` | 3.0.2 | Latest |
| `scipy` | 1.17.1 | Latest |
| `opencv-python` | 4.13.0 | AND `opencv-contrib-python` both installed |
| `Pillow` | 12.1.1 | Latest |
| `matplotlib` | 3.10.8 | Latest |
| `seaborn` | 0.13.2 | |
| `scikit-learn` | 1.8.0 | Latest |
| `tqdm` | 4.67.3 | |
| `rich` | 14.3.3 | |
| `PyYAML` | 6.0.3 | |
| `jupyter_core` | 5.9.1 | Core only; full notebook server not installed |

---

## D. Missing Dependencies

The following packages are **not installed** but are relevant to this research direction:

| Package | Reason Needed | Priority |
|---|---|---|
| `radiate` SDK | Official RADIATE loader — sensor sync, calibration matrices, coordinate transforms | **High** |
| `scikit-image` | Image quality metrics (SSIM, SNR) for degradation measurement | High |
| `open3d` | LiDAR point cloud visualisation and processing | High |
| `torch` (CUDA build) | GPU acceleration — current CPU-only build unusable for training | High |
| `jupyter` / `ipykernel` | Exploratory analysis notebooks | Medium |
| `h5py` or `zarr` | Efficient storage for processed tensors and experiment results | Medium |
| `pycocotools` | COCO-style detection evaluation metrics | Medium |
| `shapely` | Geometry ops for rotated bounding boxes | Medium |
| `laspy` / `pyntcloud` | Alternative LiDAR loaders | Low |

---

## E. Recommended Python Environment

Create an isolated virtual environment to avoid polluting the system Python:

```powershell
# From C:\Users\GOGI LAPTOP\Desktop\Research
python -m venv .venv
.venv\Scripts\Activate.ps1

# Core scientific stack — pin numpy to <2.0 for RADIATE SDK compatibility
pip install "numpy==1.26.4"
pip install scipy pandas matplotlib seaborn tqdm rich pyyaml

# Vision and point clouds
pip install opencv-python Pillow scikit-image open3d

# PyTorch — CPU fallback (replace with CUDA wheel if GPU confirmed):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# If NVIDIA GPU with CUDA 12.x confirmed:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# RADIATE SDK (no PyPI release — clone from GitHub):
# git clone https://github.com/marcelsheeny/radiate_sdk.git
# pip install -e ./radiate_sdk

# Notebooks and storage
pip install jupyter ipykernel h5py zarr scikit-learn
```

> **Critical:** `numpy==1.26.4` is strongly recommended. The RADIATE SDK was authored against NumPy 1.x. NumPy 2.x removes legacy aliases (`np.float`, `np.int`, `np.bool`) and will likely cause import errors in the SDK without patching or downgrading.

---

## F. Immediate Compatibility Issues

| Issue | Severity | Detail |
|---|---|---|
| **NumPy 2.x vs RADIATE SDK** | HIGH | NumPy 2.4.2 removes legacy aliases used by the RADIATE SDK (~2021 vintage). Import will likely fail without patching the SDK source or downgrading NumPy to 1.26.x. |
| **CPU-only PyTorch** | MEDIUM | `torch 2.10.0+cpu` — no CUDA. Any training loop will be very slow. Verify GPU presence with `nvidia-smi` before Phase 1. |
| **Low available RAM** | MEDIUM | Only ~1.0 GB free at inspection. Loading all LiDAR frames (60 x ~23K pts x 5 floats = ~28 MB) is manageable, but model weights + gradients will pressure the 7.6 GB total. Close other applications during experiments. |
| **No RADIATE SDK installed** | MEDIUM | The SDK provides pre-built calibration matrices (camera-radar-lidar extrinsics) and a synchronisation loader. Without it, cross-modal alignment requires manual implementation. |
| **OpenCV built against Python 3.9** | LOW | The installed wheel references Python 3.9 internals, but loads correctly under Python 3.11. No immediate issue; monitor if ABI-sensitive contrib modules are used. |
| **Frame count mismatch across modalities** | LOW | Radar: 18, LiDAR: 60, Camera: 50, GPS/IMU: 250. This is expected RADIATE behaviour. Timestamp-based synchronisation is required before any cross-modal analysis. |

---

*End of Phase 0 Environment Report. Do not proceed to model development until Phase 1 is approved.*
