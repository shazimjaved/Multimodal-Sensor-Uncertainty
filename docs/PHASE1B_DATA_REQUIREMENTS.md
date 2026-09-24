# PHASE 1B — RADIATE Full-Dataset and Calibration Availability Check

**Date:** 2026-09-24
**Purpose:** Determine exactly what additional data and configuration is required before any model development.
**Status:** Investigation only — no downloads performed, no raw files modified.

---

## Executive Summary

> **Key finding:** The RADIATE calibration parameters are **publicly available** — they are published in full on the official RADIATE documentation page and embedded in `config/default-calib.yaml` in the GitHub SDK repository. **No Dropbox access is needed to obtain calibration values.** Only the full fog sequence data itself requires registration.

---

## 1. Calibration File Location and Contents

### Source 1: Official documentation page (publicly accessible, no login required)

URL: `https://pro.hw.ac.uk/radiate/doc/dataset/#sensor-calibration`

The full calibration parameters are embedded directly on this page in YAML format.

### Source 2: RADIATE SDK GitHub repository (publicly accessible, no login required)

File: `https://github.com/marcelsheeny/radiate_sdk/blob/master/config/default-calib.yaml`

### Calibration parameters (extracted verbatim)

```yaml
# Radar calibration (reference frame — origin)
radar_calib:
    T: [0.0, 0.0, 0.0]
    R: [0.0, 0.0, 0.0]

# LiDAR -> Radar extrinsics (translation in meters, rotation as Rodrigues vector)
lidar_calib:
    T: [0.6003, -0.120102, 0.250012]
    R: [0.0001655, 0.000213, 0.000934]

# Left camera -> Radar extrinsics + intrinsics
left_cam_calib:
    T: [0.34001, -0.06988923, 0.287893]
    R: [1.278946, -0.530201, 0.000132]
    fx: 337.9191448899105
    fy: 338.6957068549526
    cx: 341.7366010946575
    cy: 200.7359735313929
    k1: -0.183879883467351
    k2:  0.0308609205858947
    k3: 0
    p1: 0
    p2: 0
    res: [672, 376]

# Right camera -> Radar extrinsics + intrinsics
right_cam_calib:
    T: [0.4593822, -0.0600343, 0.287433309324]
    R: [0.8493049332, 0.37113944, 0.000076230]
    fx: 337.873451599077
    fy: 338.530902554779
    cx: 329.137695760749
    cy: 186.166590759716
    k1: -0.181771143569008
    k2:  0.0295682692890613
    k3: 0
    p1: 0
    p2: 0
    res: [672, 376]

# Stereo (left-right) calibration
stereo_calib:
    TX: -120.7469
    TY:  0.1726
    TZ:  1.1592
    CV:  0.0257154
    RX: -0.0206928
    RZ: -0.000595637
    R: [[0.999983541478846,  0.000655753417350, -0.005699715684273],
        [-0.000622470939159, 0.999982758359834,  0.005839136322126],
        [0.005703446445424, -0.005835492311203,  0.9999667083098977]]
```

**Coordinate convention:** Radar is the origin. Translations are in meters. Rotations are Rodrigues vectors (use `cv2.Rodrigues()` to convert to 3x3 rotation matrix).

### What each transform enables

| Transform | Use case |
|---|---|
| `lidar_calib` T+R | Project LiDAR point cloud into radar BEV space |
| `left_cam_calib` T+R+intrinsics | Project radar BEV annotations into camera image frame |
| `right_cam_calib` T+R+intrinsics | Stereo depth + right camera annotations |
| `stereo_calib` | Stereo rectification and disparity-to-depth conversion |

### Action: Create local calibration file now

The calibration values can be saved locally as `config/default-calib.yaml` **immediately** — no download needed. This unblocks spatial fusion development while full data is obtained.

---

## 2. SDK Configuration File

File: `https://github.com/marcelsheeny/radiate_sdk/blob/master/config/config.yaml`

```yaml
radar_timestamp_file: 'Navtech_Cartesian.txt'
lidar_timestamp_file:  'velo_lidar.txt'
camera_timestamp_file: 'zed_left.txt'

use_camera_left_raw:   False
use_camera_right_raw:  False
use_camera_left_rect:  False
use_camera_right_rect: True
use_radar_polar:       False
use_radar_cartesian:   True
use_lidar_pc:          True
use_lidar_bev_image:   True
use_proj_lidar_left:   False
use_proj_lidar_right:  True

interpolate_bboxes:    False
```

These files are freely downloadable via `git clone https://github.com/marcelsheeny/radiate_sdk.git`.

---

## 3. Minimum Data Requirements Analysis

### Option A: tiny_foggy only (current state)

| Attribute | Value |
|---|---|
| Usable radar frames | 14 |
| Total duration | 3.3s of overlap |
| Suitable for training? | **No** |
| Suitable for loader prototyping? | Yes |

**Verdict:** Insufficient. Cannot train or evaluate any model.

### Option B: Full fog_6_0 sequence only ← **Recommended minimum**

| Attribute | Value |
|---|---|
| Estimated radar frames | ~2,000 |
| Estimated duration | ~7-8 minutes |
| Suitable for a prototype training run? | Yes (with train/val split) |
| Requires registration? | Yes |
| Estimated download size | ~6.1 GB uncompressed |

**Verdict:** This is the **minimum viable dataset** for this research direction. Fog degradation is the core phenomenon being studied, so one full fog sequence provides:
- Sufficient labeled radar frames for a small-scale detection prototype (~1,600 train / ~400 val)
- Consistent fog conditions matching the Phase 0/1 sample
- All modalities: radar, LiDAR, stereo camera, GPS/IMU

### Option C: Multiple fog sequences (~8 sequences)

| Attribute | Value |
|---|---|
| Estimated radar frames | ~16,000 |
| Estimated download size | ~48 GB |
| Requires registration? | Yes |

**Verdict:** Needed for robust model training and fog-level generalization, but not required for initial prototyping. Pursue after Option B pipeline is validated.

### Option D: Full RADIATE dataset (all weather conditions)

| Attribute | Value |
|---|---|
| Total dataset | ~3 hours of annotated radar (200K+ instances) |
| Scenarios | 7 (Sunny Parked, Sunny/Overcast Urban, Overcast Motorway, Night Motorway, Rain Suburban, Fog Suburban, Snow Suburban) |
| Estimated size | ~50-70 GB |
| Use case | Cross-weather degradation experiments |

**Verdict:** Required for cross-weather generalization study but premature for the current phase.

---

## 4. Storage Requirements (Per Modality)

Based on measured data rates from `tiny_foggy` (57 MB for 4.2s):

### For 500 radar frames (~2 min, minimum useful training subset)

| Modality | Estimated size |
|---|---|
| Navtech_Cartesian (radar BEV) | ~264 MB |
| Navtech_Polar (radar raw) | ~72 MB |
| velo_lidar (point clouds, ASCII CSV) | ~986 MB |
| zed_left | ~119 MB |
| zed_right | ~125 MB |
| GPS_IMU_Twist | ~6 MB |
| annotations | ~0.1 MB (one JSON per sequence) |
| **Total (500 radar frames)** | **~1.6 GB** |

### For full fog_6_0 (~2,000 radar frames, ~7-8 min)

| Modality | Estimated size |
|---|---|
| Navtech_Cartesian | ~1,056 MB |
| Navtech_Polar | ~289 MB |
| velo_lidar | ~3,944 MB |
| zed_left | ~478 MB |
| zed_right | ~500 MB |
| GPS_IMU_Twist | ~22 MB |
| annotations | ~0.1 MB |
| **Total (full fog_6_0)** | **~6.3 GB** |

> **Note on LiDAR dominance:** LiDAR CSVs are stored as plain-text ASCII, making them disproportionately large (~650 KB/frame). A converted binary format (numpy `.npy`) would reduce this to ~50-60 MB for 500 frames. This conversion is recommended as a preprocessing step.

### Disk space recommendation

| Requirement | Space |
|---|---|
| Operating minimum (full fog_6_0) | 10 GB free (download + extracted + processed) |
| Comfortable working space | 20 GB free |
| For 3 fog sequences | 30 GB free |
| Current available (7.6 GB RAM machine) | Small RAM — favor batch/stream loading |

---

## 5. Exact Files and Directories Required

### From RADIATE SDK (GitHub — no registration required)

```
radiate_sdk/
├── config/
│   ├── config.yaml           # Sensor usage flags
│   └── default-calib.yaml    # All calibration matrices
├── radiate/                  # Python SDK source
│   ├── scene.py              # Main sequence loader
│   ├── sensor.py             # Per-sensor readers
│   └── ...
└── requirements.txt          # numpy, opencv-python, pandas, pyyaml, matplotlib
```

**Action:** `git clone https://github.com/marcelsheeny/radiate_sdk.git`

### From RADIATE Dropbox (registration required)

For full fog_6_0 sequence, the expected directory structure is:

```
fog_6_0/
├── meta.json
├── annotations/
│   └── annotations.json
├── Navtech_Cartesian/
│   └── 000001.png ... 00NNNN.png
├── Navtech_Cartesian.txt
├── Navtech_Polar/
│   └── 000001.png ... 00NNNN.png
├── Navtech_Polar.txt
├── velo_lidar/
│   └── 000001.csv ... 00NNNN.csv
├── velo_lidar.txt
├── zed_left/
│   └── 000001.png ... 00NNNN.png
├── zed_left.txt
├── zed_right/
│   └── 000001.png ... 00NNNN.png
├── zed_right.txt
└── GPS_IMU_Twist/
    └── 000001.txt ... 00NNNN.txt
```

**Naming note:** The full sequence may be named differently (e.g., `fog_6_0`, `fog_1_0`, etc.). All fog sequences share identical directory structure.

---

## 6. Can tiny_foggy Be Reused?

| Use | Can reuse tiny_foggy? |
|---|---|
| Validate data loader code | **Yes** |
| Test calibration transform pipeline | **Yes** (once calib YAML is added locally) |
| Visualize synchronized multimodal frames | **Yes** |
| Train any detection model | **No** — 14 frames is insufficient |
| Evaluate model performance | **No** — no test-set diversity |
| Develop degradation simulation logic | **Yes** (can test augmentation code on these 14 frames) |

**Conclusion:** tiny_foggy remains valid as a **development testbed**. All loader code and calibration logic developed against it will transfer directly to full sequences without modification.

---

## 7. Where to Obtain the Full Dataset

### Step 1: Register (required for Dropbox access)

- **Registration form:** `https://pro.hw.ac.uk/radiate/downloads`
- Fill in institutional/organizational email address
- Submit Google Form embedded on the page
- You will receive a Dropbox invitation to your email

### Step 2: Sample data (no registration needed)

- **tiny_foggy.zip:** `https://www.dropbox.com/s/dh361agxhn42ywm/tiny_foggy.zip?dl=0`
- Already obtained and in use

### Step 3: SDK and calibration (no registration needed)

- **GitHub:** `https://github.com/marcelsheeny/radiate_sdk`
- `git clone` and copy `config/` directory to workspace

---

## 8. Licensing and Access Restrictions

| Aspect | Detail |
|---|---|
| **License** | Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) |
| **Permitted uses** | Non-commercial academic research |
| **Commercial use** | Contact Prof. Andrew Wallace and Dr. Sen Wang at Heriot-Watt University |
| **Attribution required** | Yes — cite: Sheeny et al., "RADIATE: A Radar Dataset for Automotive Perception," arXiv:2010.09076, 2020 |
| **Share-alike** | Any derivatives must be shared under same license |
| **Registration** | Required for Dropbox access to full sequences |
| **Calibration data** | Publicly available without registration (published on docs page + GitHub) |

> **Important:** The CC BY-NC-SA 4.0 license permits this research prototype for academic non-commercial purposes. Any publication using RADIATE must cite the paper.

---

## 9. Recommended Next Actions (Priority Order)

### Immediate (no download required)

1. **Save calibration YAML locally** from the published values above:
   ```
   C:\Users\GOGI LAPTOP\Desktop\Research\config\default-calib.yaml
   ```
   This immediately unblocks LiDAR-to-radar projection and camera annotation overlay on the existing 14 tiny_foggy frames.

2. **Clone the RADIATE SDK:**
   ```powershell
   git clone https://github.com/marcelsheeny/radiate_sdk.git
   ```
   Store at: `C:\Users\GOGI LAPTOP\Desktop\Research\radiate_sdk\`

3. **Patch SDK for NumPy 2.x compatibility** (or pin NumPy to 1.26.4 in a virtual environment before installing the SDK).

### Short-term (requires registration)

4. **Submit registration form** at `https://pro.hw.ac.uk/radiate/downloads` using an institutional or organizational email address.

5. **Download minimum: one full fog sequence** (`fog_6_0` or whichever is the principal annotated fog sequence). Estimated 6.3 GB.

6. **Validate the full loader** against the full fog sequence with the SDK, confirming timestamp sync and calibration projection.

### Before Phase 2 model development

7. Confirm at least 500 usable multimodal frames with annotations are available.
8. Convert LiDAR CSVs to binary `.npy` format to reduce I/O overhead.
9. Decide: **radar-only** or **radar + camera** fusion (calibration is now available for either).

---

## 10. Summary Table

| Item | Status | Source | Action Required |
|---|---|---|---|
| tiny_foggy sample data | **Available** | Dropbox (public link) | None |
| Calibration YAML | **Available** (not yet local) | GitHub SDK / official docs | Save locally now |
| SDK source code | **Available** | GitHub | git clone |
| Full fog_6_0 sequence | **Not yet obtained** | Dropbox (registration required) | Submit registration form |
| GPU for training | **Not confirmed** | Local machine | Verify with `nvidia-smi` |
| NumPy compatibility | **Issue identified** | System Python | Create venv with numpy==1.26.4 |

---

*End of Phase 1B. The calibration blocker is resolved. The primary remaining blocker is the full fog sequence download, which requires registration.*
