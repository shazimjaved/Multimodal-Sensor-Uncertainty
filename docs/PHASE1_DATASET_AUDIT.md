# PHASE 1 — RADIATE tiny_foggy Dataset and Synchronization Audit

**Date:** 2026-09-24  
**Dataset root:** `C:\Users\GOGI LAPTOP\Desktop\Research\tiny_foggy\`  
**Status:** Read-only audit — no raw files modified.

---

## 1. RADIATE SDK Status

### Availability

The official RADIATE SDK (`marcelsheeny/radiate_sdk`) is **not installed** in the current Python environment, and is **not available on PyPI**.

It is distributed exclusively via GitHub. The SDK source files (e.g., `scene.py`, `radar.py`) are not present locally.

### What the SDK provides (from README inspection)

Based on the official SDK README, the key components are:

| Component | Purpose |
|---|---|
| `scene.py` | Top-level loader: reads all modality streams for a given sequence, handles frame indexing |
| Timestamp parsing | Reads `Navtech_Cartesian.txt`, `velo_lidar.txt`, `zed_left.txt` etc. and builds a synchronized frame index |
| Calibration loader | Reads a `config/` directory (per-sequence or global) containing camera intrinsics, extrinsic transforms, and radar resolution parameters |
| Annotation loader | Reads `annotations/annotations.json` and maps bboxes to radar frame indices |
| Coordinate transform | Projects LiDAR points into the radar BEV frame using extrinsic calibration |

> **Finding:** The RADIATE SDK can be used without modification for data loading. However, its NumPy 1.x dependency will conflict with the current NumPy 2.4.2 installation. The core timestamp parsing and annotation loading logic can be replicated from the file formats, which are fully documented in the README.

---

## 2. Timestamp File Inspection

All timestamp files follow the format: `Frame: NNNNNN Time: XXXXXXXXXX.XXXXXXXXX`

### Navtech_Cartesian.txt (Radar)

- **Frames:** 1 – 18 (18 total)
- **Time range:** `1574859771.745` → `1574859775.933`
- **Duration:** 4.189 s
- **Approx. rate:** 4.3 Hz (consistent ~0.233s per frame)

### Navtech_Polar.txt (Radar Polar)

- **Identical timestamps** to Navtech_Cartesian.txt — same scan, two representations.

### velo_lidar.txt (LiDAR)

- **Frames:** 18 – 77 (but only 60 CSV files exist: 1–60)
- **Time range:** `1574859771.701` → `1574859775.804`
- **Duration:** 4.104 s
- **Rate:** ~10 Hz
- **Note:** Frame numbering in the timestamp file starts at 18 but the actual files are `000001.csv` – `000060.csv`. The frame numbers in the `.txt` are absolute sequence indices from the full dataset; local files are renumbered 1–60. LiDAR starts **0.044s before radar**.

### zed_left.txt (Left Camera)

- **Frames:** 1 – 100 (100 entries; 50 PNG files present)
- **Time range:** `1574859772.428` → `1574859779.105`
- **Duration:** 6.677 s
- **Rate:** ~15 Hz
- **Note:** Timestamp file has 100 entries but only 50 PNG files exist. The extra 50 timestamps correspond to frames not included in the tiny_foggy subset. Camera starts **0.683s after radar**.

### zed_right.txt (Right Camera)

- **Frames:** 1 – 50 (50 entries, 50 PNG files)
- **Time range:** `1574859772.428` → `1574859775.767`
- **Duration:** 3.339 s
- **Rate:** ~15 Hz
- **Note:** Right camera ends earliest of all modalities.

### GPS_IMU_Twist.txt

- **Frames:** 1 – 250 (250 entries, 250 TXT files)
- **Time range:** `1574859835.750` → `1574859841.950`
- **Offset from radar:** **+64.005 s** — does NOT overlap with radar/lidar/camera
- **Conclusion:** GPS/IMU timestamps are from a **different ROS bag** or use a separate time reference in this tiny_foggy clip. GPS cannot be temporally synchronized with the other modalities in this sample without external offset correction. This modality is effectively **isolated** in the tiny_foggy subset.

---

## 3. Temporal Overlap Analysis

Overlap is computed as the intersection of all modality time windows (excluding GPS due to confirmed offset):

| Modality | Start (s) | End (s) | Duration (s) | Frames |
|---|---|---|---|---|
| Radar | 1574859771.745 | 1574859775.933 | 4.189 | 18 |
| LiDAR | 1574859771.701 | 1574859775.804 | 4.104 | 60 (42 in txt) |
| ZED Left | 1574859772.428 | 1574859779.105 | 6.677 | 50 (of 100 timestamps) |
| ZED Right | 1574859772.428 | 1574859775.767 | 3.339 | 50 |

**Effective overlap window (all 4 modalities):**

```
Start: 1574859772.428  (camera start — constrains lower bound)
End:   1574859775.767  (cam_right end — constrains upper bound)
Duration: 3.339 s
```

**Frames falling within the overlap window:**

| Modality | Frames in overlap |
|---|---|
| Radar | 14 (frames 4–17) |
| LiDAR | 33 (of the 60 available files) |
| ZED Left | 50 (all available) |
| ZED Right | 50 (all available) |

> **Key finding:** Radar frames 1–3 predate the camera (before camera first frame), and radar frame 18 falls after the right camera ends. The **14-frame window (radar frames 4–17) has full 4-modality coverage.**

---

## 4. Synchronization Table

For every radar frame, the nearest LiDAR frame and nearest left camera frame are identified by minimum absolute timestamp difference.

| Radar Frame | Radar ts | Nearest LiDAR | LiDAR dt (ms) | Nearest Cam Left | Cam dt (ms) |
|---|---|---|---|---|---|
| 1 | 771.745 | 18 | 43.7 | 1 | 683.1 |
| 2 | 771.978 | 21 | 23.7 | 1 | 450.2 |
| 3 | 772.214 | 23 | 12.5 | 1 | 213.8 |
| **4** | **772.453** | **26** | **49.1** | **1** | **24.8** |
| **5** | **772.696** | **28** | **5.7** | **4** | **1.3** |
| **6** | **772.936** | **30** | **34.3** | **8** | **25.7** |
| **7** | **773.185** | **33** | **17.1** | **11** | **22.9** |
| **8** | **773.433** | **35** | **30.2** | **15** | **3.3** |
| **9** | **773.685** | **38** | **17.9** | **19** | **11.7** |
| **10** | **773.933** | **40** | **29.7** | **23** | **31.1** |
| **11** | **774.188** | **43** | **15.4** | **26** | **23.9** |
| **12** | **774.440** | **45** | **36.9** | **30** | **9.2** |
| **13** | **774.696** | **48** | **7.4** | **34** | **2.0** |
| **14** | **774.941** | **50** | **37.6** | **38** | **23.9** |
| **15** | **775.183** | **53** | **20.8** | **41** | **17.7** |
| **16** | **775.436** | **55** | **32.3** | **45** | **3.8** |
| **17** | **775.686** | **58** | **18.2** | **49** | **13.5** |
| 18 | 775.933 | 59 | 128.9 | 52 | 33.4 |

> Bold rows = frames within the 4-modality overlap window (frames 4–17).  
> Timestamp suffix shown as last 3 decimal digits of epoch for readability.

### Synchronization Statistics

| Pair | Min (ms) | Max (ms) | Mean (ms) | Median (ms) |
|---|---|---|---|---|
| Radar ↔ LiDAR | 5.7 | 128.9 | 31.2 | 26.7 |
| Radar ↔ ZED Left | 1.3 | 683.1 | 88.6 | 23.4 |

> **For frames 4–17 only (clean overlap):**
> - LiDAR dt: min=5.7ms, max=49.1ms, mean=25.7ms, median=25.2ms
> - Camera dt: min=1.3ms, max=31.1ms, mean=14.2ms, median=14.6ms

---

## 5. Calibration / Extrinsic Information

### What is required

For multimodal fusion:
- **LiDAR → Radar:** 3D rotation + translation from LiDAR frame to radar polar frame
- **Camera → Radar:** Camera intrinsics (K matrix, distortion) + extrinsic rotation/translation
- **LiDAR → Camera:** Required for LiDAR-image projection

### What is present in tiny_foggy

**No calibration files are present** in `C:\Users\GOGI LAPTOP\Desktop\Research\tiny_foggy\`.

Per the RADIATE SDK README: *"Sensor calibration is required for multi-sensor fusion... In terms of extrinsic calibration, the radar sensor is chosen as [the reference frame]"*. The calibration data is stored in a separate `config/` directory distributed with the full dataset download, not included in individual sequence subsets.

### Verified absent files

| Expected file | Present? |
|---|---|
| `config/camera.yaml` or equivalent | No |
| `config/lidar_to_radar.yaml` | No |
| `config/camera_to_radar.yaml` | No |
| Any `.yaml`, `.json` calibration | No |

### Consequence

Without calibration matrices:
- LiDAR points cannot be projected onto the radar BEV image using ground-truth transforms
- Camera pixels cannot be matched to radar detections in metric space
- Multimodal spatial fusion is **not possible** from ground truth
- However, individual modality processing (radar-only, camera-only, lidar-only) remains fully feasible

> The RADIATE SDK README states that Navtech radar pixel size is 0.17361 m/px (Cartesian). LiDAR and camera extrinsics can be sourced from the full dataset download or the RADIATE paper (Tables 1–2 in the original publication). They are **not invented** here.

---

## 6. Annotation Inspection

### Format

`annotations.json` is a **list of 17 tracked objects**, each with:

```json
{
  "id": 1,
  "class_name": "bus",
  "bboxes": [<714 entries>]
}
```

Each bbox entry is either:
- `{ "position": [cx, cy, w, h], "rotation": float_degrees }` — a valid annotation
- `[]` — empty list, meaning the object is not visible in that frame

### Frame indexing

The `bboxes` list is **0-indexed** and corresponds directly to **radar frame IDs** (index 0 = radar frame 1). Confirmed by cross-referencing the 714-entry length against the full sequence length. The annotations were created in radar-centric coordinates (Cartesian BEV image, 1152×1152 px, 0.17361 m/px).

### Coverage of tiny_foggy (radar frames 1–18)

| Object ID | Class | Valid bboxes in frames 1–18 | Frames covered |
|---|---|---|---|
| 1 | bus | 18 / 18 | All frames |
| 2 | car | 14 / 18 | 1–14 |
| 3 | car | 8 / 18 | 11–18 |
| 4 | car | 2 / 18 | 17–18 |
| 5–17 | car/van | 0 / 18 | Not visible |

### Per-frame annotation count (frames 1–18)

| Frames | Annotations | Classes |
|---|---|---|
| 1–10 | 2 | bus, car |
| 11–14 | 3 | bus, car, car |
| 15–16 | 2 | bus, car |
| 17–18 | 3 | bus, car, car |

### Class distribution (full sequence vs. tiny_foggy)

| Class | Objects (full) | Valid bboxes (full) | Valid bboxes (tiny_foggy 18 frames) |
|---|---|---|---|
| bus | 1 | 59 | 18 |
| car | 12 | 269 | 24 |
| van | 4 | 81 | 0 |

> **Key finding:** Van class has **zero annotations** in the 18-frame tiny_foggy subset. Only `bus` and `car` are annotated. Maximum 3 objects visible simultaneously. The dataset is **severely class-sparse** at this scale.

---

## 7. Per-Frame Usability Assessment

Criteria: radar frame has a valid annotation AND a LiDAR frame within 50ms AND a camera frame within 100ms.

| Radar Frame | Annotations | LiDAR dt (ms) | Cam dt (ms) | Usable? |
|---|---|---|---|---|
| 1 | 2 | 43.7 | 683.1 | NO — cam too far |
| 2 | 2 | 23.7 | 450.2 | NO — cam too far |
| 3 | 2 | 12.5 | 213.8 | NO — cam too far |
| **4** | **2** | **49.1** | **24.8** | **YES** |
| **5** | **2** | **5.7** | **1.3** | **YES** |
| **6** | **2** | **34.3** | **25.7** | **YES** |
| **7** | **2** | **17.1** | **22.9** | **YES** |
| **8** | **2** | **30.2** | **3.3** | **YES** |
| **9** | **2** | **17.9** | **11.7** | **YES** |
| **10** | **2** | **29.7** | **31.1** | **YES** |
| **11** | **3** | **15.4** | **23.9** | **YES** |
| **12** | **3** | **36.9** | **9.2** | **YES** |
| **13** | **3** | **7.4** | **2.0** | **YES** |
| **14** | **3** | **37.6** | **23.9** | **YES** |
| **15** | **2** | **20.8** | **17.7** | **YES** |
| **16** | **2** | **32.3** | **3.8** | **YES** |
| **17** | **3** | **18.2** | **13.5** | **YES** |
| 18 | 3 | 128.9 | 33.4 | NO — LiDAR too far |

**14 of 18 radar frames are fully usable** with synchronized LiDAR and camera data and valid annotations.

---

## 8. Synchronization Visualizations

Three examples were generated. Each panel shows:
- Radar Cartesian BEV with ground-truth bounding boxes drawn
- LiDAR Bird's Eye View projection (50m × 50m range)
- Corresponding ZED left camera frame

Files saved to `outputs/audit/`:
- `sync_example_01.png` — radar frame 5 (2 annotations, dt_lidar=5.7ms, dt_cam=1.3ms)
- `sync_example_02.png` — radar frame 10 (2 annotations, dt_lidar=29.7ms, dt_cam=31.1ms)
- `sync_example_03.png` — radar frame 16 (2 annotations, dt_lidar=32.3ms, dt_cam=3.8ms)

---

## 9. Simplest Technically Valid Multimodal Task

Based on actual labels, synchronization quality, calibration availability, and data volume:

### Option A: Radar-centric object detection
- **Annotations:** Directly in radar coordinate space ✓
- **Labels available:** 2–3 objects per frame, classes bus and car ✓
- **Calibration required:** No ✓
- **Sample count:** 14 usable frames

### Option B: Multimodal object detection (radar + LiDAR + camera)
- **Annotations:** In radar space only; LiDAR/camera need calibration to align ✗
- **Calibration:** Missing from tiny_foggy ✗
- **Note:** Would require sourcing extrinsics from the full RADIATE download

### Option C: Object-level classification
- **Labels available:** Only bus and car (van has 0 instances) — too few classes
- **Not suitable as a standalone task**

### Option D: Radar-centric detection + camera as auxiliary (no spatial fusion)
- Radar provides detection bounding boxes; camera is loaded as contextual context without geometric projection
- Calibration NOT required for this use case
- Directly supported by available data

### Recommended task: **Option D — Radar-centric detection with camera auxiliary**

This is the **simplest technically valid multimodal task** that:
1. Uses actual ground-truth annotations (radar BEV space)
2. Does not require calibration matrices
3. Involves two modalities (radar + camera) with acceptable temporal sync (median ~14ms)
4. Has 14 labeled frames — sufficient for a prototype loader/pipeline, insufficient for training

> If calibration is sourced (from full RADIATE download or the paper), **Option B becomes viable** and would be the preferred research task.

---

## 10. Limitations of tiny_foggy

| Limitation | Detail | Impact |
|---|---|---|
| **Very small — 14 usable frames** | Only 14 radar frames with full multimodal sync and labels | Cannot train or evaluate any model; only prototyping and inspection |
| **Only 2 classes visible** | bus (1 object) and car (2–3 objects); van class has 0 instances in 18 frames | Single-class-dominated; class imbalance is extreme |
| **Max 3 objects per frame** | Scene density is very low | Object detection baseline trivially biased |
| **No calibration files** | `config/` directory not included in tiny_foggy | Spatial sensor fusion impossible without external sourcing |
| **GPS not temporally aligned** | GPS timestamps are +64s offset from radar/lidar/camera | GPS/IMU cannot be used for ego-motion compensation in this subset |
| **Camera starts 0.683s after radar** | Radar frames 1–3 have no usable camera sync | Reduces usable frame count from 18 to 14 |
| **LiDAR ends before radar (frame 18)** | Frame 18 LiDAR dt = 128.9ms | Radar frame 18 also excluded |
| **zed_left.txt has 100 timestamps but only 50 PNGs** | Frames 51–100 listed but absent | Must detect and skip missing files in loader |
| **Annotations cover 714 frames but only 18 are local** | Annotation file is the full-sequence annotation; only first 18 entries apply | Must slice correctly (index 0–17) |
| **Fog degradation is fixed** | Sequence is `fog_6_0` — single fog intensity level | No progressive degradation within this clip |

---

## Feasibility Decision

### **READY WITH RESTRICTIONS**

**Evidence:**

1. **14 fully synchronized frames** exist with valid radar, LiDAR, and camera data within acceptable timing tolerances (LiDAR median 25ms, camera median 14ms).

2. **Ground-truth annotations are present and correct** for all 14 usable frames, in radar BEV pixel coordinates. No annotation gaps within the usable window.

3. **Radar-centric detection is directly supported** without calibration. The radar image, annotation bboxes, and nearest camera frame can all be loaded with a custom parser using only standard Python and OpenCV — no SDK required.

4. **The dataset is too small for model training.** 14 labeled frames is sufficient only for: (a) building and validating a data loader, (b) visual inspection of multimodal synchronization quality, (c) establishing a baseline pipeline that can scale to the full RADIATE dataset.

**Restrictions:**

- Spatial fusion (LiDAR-radar projection, camera-radar projection) requires obtaining calibration matrices from the full RADIATE dataset download before Phase 2.
- Any model development requires using the full RADIATE fog sequence (`fog_6_0` full, or other sequences) — not just this tiny subset.
- GPS/IMU is not usable in its current form in this sample.

---

*End of Phase 1 Dataset Audit. Pending: obtain full RADIATE fog sequence and calibration config before proceeding to Phase 2.*
