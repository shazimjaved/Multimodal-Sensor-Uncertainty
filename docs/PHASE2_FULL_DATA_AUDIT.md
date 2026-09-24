# PHASE 2 — Full fog_6_0 Dataset Verification and Preparation

**Date:** 2026-09-24  
**Sequence path:** `C:\Users\GOGI LAPTOP\Desktop\Research\fog_6_0\`  
**Status:** ✅ All checks passed

---

## 1. Dataset Location and Structure

```
fog_6_0/
├── meta.json                    ← sequence metadata
├── annotations/
│   └── annotations.json         ← all annotations (0.3 MB)
├── Navtech_Cartesian/           ← 714 PNG files (372.0 MB)
├── Navtech_Cartesian.txt        ← 714 timestamp entries
├── Navtech_Polar/               ← 714 PNG files (102.2 MB)
├── Navtech_Polar.txt            ← 714 timestamp entries
├── velo_lidar/                  ← 1799 CSV files (1243.9 MB)
├── velo_lidar.txt               ← 1799 timestamp entries
├── zed_left/                    ← 2659 PNG files (234.2 MB)
├── zed_left.txt                 ← 2659 timestamp entries
├── zed_right/                   ← 2659 PNG files (236.3 MB)
├── zed_right.txt                ← 2659 timestamp entries
├── GPS_IMU_Twist/               ← 7200 TXT files (6.0 MB)
└── GPS_IMU_Twist.txt            ← 7200 timestamp entries
```

**meta.json:**
```json
{ "name": "fog_6_0", "type": "fog", "set": "test", "version": "1.0", "date_created": "Sat Feb 29 03:09:49 2020" }
```

> [!IMPORTANT]
> `meta.json` marks this sequence as `set: "test"` in the official RADIATE dataset split. This means in the full RADIATE benchmark, fog_6_0 is reserved for evaluation. For our prototype, we apply a temporal within-sequence split. A production model would require separate training sequences.

---

## 2. Modality File Counts

| Modality | Files | Timestamps | Match | Size |
|---|---|---|---|---|
| Navtech_Cartesian | 714 | 714 | ✅ | 372.0 MB |
| Navtech_Polar | 714 | 714 | ✅ | 102.2 MB |
| velo_lidar | 1799 | 1799 | ✅ | 1,243.9 MB |
| zed_left | 2659 | 2659 | ✅ | 234.2 MB |
| zed_right | 2659 | 2659 | ✅ | 236.3 MB |
| GPS_IMU_Twist | 7200 | 7200 | ✅ | 6.0 MB |
| **Total** | | | | **2,194.6 MB (2.1 GB)** |

File naming is sequential, 1-indexed, continuous — no gaps detected.

---

## 3. Temporal Characteristics

| Sensor | Frames | Duration | FPS | ID Range |
|---|---|---|---|---|
| Radar | 714 | 178.0 s | 4.01 Hz | 1–714 |
| LiDAR | 1799 | 180.0 s | 10.00 Hz | 1–1799 |
| Camera Left | 2659 | 177.6 s | 14.98 Hz | 1–2659 |
| Camera Right | 2659 | 177.6 s | 14.98 Hz | 1–2659 |
| GPS/IMU | 7200 | 180.0 s | 40.01 Hz | 1–7200 |

**Total sequence duration: ~178 seconds (~3 minutes)**

Radar frame continuity: **no gaps** (frames 1–714 are contiguous).

---

## 4. Cross-Sensor Synchronization

| Pair | Min Δt | Max Δt | Mean Δt | Median Δt |
|---|---|---|---|---|
| Radar ↔ LiDAR | 0.0 ms | **49.9 ms** | 24.9 ms | 25.1 ms |
| Radar ↔ Camera Left | 0.0 ms | 683.1 ms | 18.3 ms | **16.7 ms** |
| Radar ↔ Camera Right | 0.0 ms | 683.1 ms | 18.3 ms | 16.7 ms |

> [!WARNING]
> Camera max Δt of **683 ms** is anomalous. This occurs only at the tail of the sequence (final ~3 frames) where the camera stream ends slightly before the radar. The **median is 16.7 ms**, which is excellent. The 3 outlier frames are safely excluded by the dt < 100ms filter.

**Usable frames (all sensors Δt < 100 ms):** **711 / 714** (99.6%)

---

## 5. Annotation Audit

### Object Inventory

| Class | Tracked objects | Valid bboxes | % of all bboxes |
|---|---|---|---|
| car | 31 | 1,382 | 84.6% |
| van | 7 | 193 | 11.8% |
| bus | 1 | 59 | 3.6% |
| **Total** | **39** | **1,634** | 100% |

### Frame Coverage

| Metric | Value |
|---|---|
| Total radar frames | 714 |
| Frames with ≥ 1 annotation | **660 / 714 (92.4%)** |
| Frames without annotations | 54 (7.6%) |
| First annotated frame | 1 |
| Last annotated frame | 691 |
| Frames with ≥ 2 annotations | 483 |
| Frames with ≥ 3 annotations | 144 |
| Max objects in a single frame | **8** |
| Mean objects per frame | 2.29 |
| Median objects per frame | 2.0 |

**Usable multimodal frames (Δt < 100ms AND ≥ 1 annotation):** **657 frames**

> ✅ This exceeds the 500-frame minimum confirmed in Phase 1B.

---

## 6. Calibration and Projection Verification

The Phase 1C calibration (`config/default-calib.yaml`) was used without modification.

Three frames were tested across the full temporal range:

| Frame position | Radar frame | t from start | LiDAR Δt | Cam Δt | Annotations | Projection |
|---|---|---|---|---|---|---|
| **Early** | 60 | +14s | ~25ms | ~17ms | **5** (cars+vans) | ✅ PASS |
| **Middle** | 357 | +88s | ~25ms | ~17ms | **2** (car+van) | ✅ PASS |
| **Late** | 640 | +159s | ~25ms | ~17ms | **2** (bus+car) | ✅ PASS |

LiDAR point counts in BEV (after forward/height filtering):
- Early frame: ~12,949 pts — dense (close objects visible)
- Middle frame: ~12,847 pts — nominal
- Late frame: ~13,622 pts — nominal

**Calibration remains geometrically stable across the entire 178s sequence.**

---

## 7. Visualizations

````carousel
![Early frame (t+14s, 5 annotations)](C:\Users\GOGI LAPTOP\.gemini\antigravity-ide\brain\ed71568d-dec3-4bad-ba88-f6aa4a23eede\early_frame.png)
<!-- slide -->
![Middle frame (t+88s, 2 annotations)](C:\Users\GOGI LAPTOP\.gemini\antigravity-ide\brain\ed71568d-dec3-4bad-ba88-f6aa4a23eede\middle_frame.png)
<!-- slide -->
![Late frame (t+159s, 2 annotations)](C:\Users\GOGI LAPTOP\.gemini\antigravity-ide\brain\ed71568d-dec3-4bad-ba88-f6aa4a23eede\late_frame.png)
````

Each panel shows: Radar BEV with annotation boxes | LiDAR in radar BEV (height-coloured, range rings at 25/50/75/100m) | Camera image with 3D wireframe projections.

---

## 8. Train / Validation / Test Split Proposal

> [!IMPORTANT]
> Temporal contiguous blocks are used — **no random shuffling of individual frames**. This prevents neighboring frames (which are nearly identical due to 4 Hz capture rate) from appearing in different splits, which would cause data leakage.

### Proposed split

| Split | Radar frames | Annotated | Rationale |
|---|---|---|---|
| **Train** | 1–499 (499 frames, 70%) | 483 | Early + mid sequence; diverse scenes |
| **Val** | 500–606 (107 frames, 15%) | 92 | Mid-to-late; gap of 1 frame between train and val |
| **Test** | 607–714 (108 frames, 15%) | 85 | Final segment; includes bus at close range |

**Gap strategy:** No gap between contiguous segments is needed since radar is at 4 Hz — adjacent frames are already 250ms apart. Each split boundary represents a clean temporal cut.

### Important caveat

The official RADIATE split labels fog_6_0 as a **test** sequence. For a rigorous evaluation, train splits should come from other RADIATE sequences (e.g., `fog_1_0`, `fog_2_0`). The prototype split above is valid for internal development and ablation studies.

---

## 9. Output Files

```
outputs/full_data_audit/
├── dataset_summary.json       ← complete metadata and stats
├── modality_counts.csv        ← file counts and sizes per modality
├── synchronization_summary.json  ← per-pair sync statistics
├── class_distribution.csv    ← annotation class breakdown
├── split_proposal.json        ← train/val/test frame ranges
├── early_frame.png            ← radar frame 60 (t+14s)
├── middle_frame.png           ← radar frame 357 (t+88s)
└── late_frame.png             ← radar frame 640 (t+159s)
```

---

## 10. Summary Checklist

| Check | Result |
|---|---|
| All modality directories present | ✅ |
| File count matches timestamp count for all sensors | ✅ |
| No frame ID gaps in radar sequence | ✅ |
| 500+ usable annotated frames | ✅ **657** |
| Calibration valid across full sequence | ✅ |
| Annotations span 92.4% of radar frames | ✅ |
| LiDAR sync Δt < 50ms (100% of frames) | ✅ |
| Camera sync Δt < 100ms (99.6% of frames) | ✅ |
| Train/val/test split defined (temporal) | ✅ |

---

## Final Decision

```
╔══════════════════════════════════════════════════════════╗
║   PHASE 2 VERDICT:  ✅ READY FOR MODEL DEVELOPMENT      ║
╠══════════════════════════════════════════════════════════╣
║  Sequence:   fog_6_0  (178s, ~3 min of foggy driving)   ║
║  Usable frames:  657  (annotated + all sensors synced)  ║
║  Classes:    car (85%), van (12%), bus (4%)              ║
║  Calibration verified at early / middle / late frames   ║
║  Train/val/test:  499 / 107 / 108 radar frames          ║
╚══════════════════════════════════════════════════════════╝
```
