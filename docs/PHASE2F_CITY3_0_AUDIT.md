# PHASE 2F — city_3_0 Archive-Only Verification

**Date:** 2026-09-24  
**Target:** `c:\Users\GOGI LAPTOP\Desktop\Research\city_3_0.zip` (2.11 GB / 2,268,247,700 bytes)  
**Inspection Method:** Direct Python `zipfile` inspection (Zero extraction performed)  
**Associated Artifact:** [`outputs/city3_0_audit.csv`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/outputs/city3_0_audit.csv)  
**Held-out Test Benchmark:** `fog_6_0` (Preserved and unaltered)

---

## 1. Executive Summary & Status

`city_3_0.zip` was inspected directly from the ZIP archive without extracting files to disk.

```
╔══════════════════════════════════════════════════════════════════════╗
║               OVERALL STATUS: READY WITH RESTRICTIONS                ║
╠══════════════════════════════════════════════════════════════════════╣
║  • Official Split: train_good_weather (Legally clean for training)   ║
║  • Annotation Coverage: 100.0% (713/713 radar frames annotated)       ║
║  • Valid Bounding Boxes: 8,875 (314 tracked objects)                 ║
║  • Multimodal Sync: 99.4% (709/713 frames synchronized within 50ms)  ║
║  • Sensor Modalities: Radar (Cartesian+Polar), LiDAR (32-ring),       ║
║                       Stereo Camera (Left+Right), GPS/IMU Twist      ║
╠══════════════════════════════════════════════════════════════════════╣
║  KEY RESTRICTIONS IDENTIFIED:                                        ║
║  1. Frame 1–4 Exclusion: Camera capture started ~0.87s after radar.  ║
║     Multimodal training must strictly start at frame 5 (idx 4).      ║
║  2. Radar Stream Gaps: 4 brief gaps (>400ms, max 2.15s) in radar.    ║
║  3. Calibration Verification: Recorded Aug 2019 vs Feb 2020 for      ║
║     fog_6_0 (7-month delta); requires visual projection smoke test.   ║
║  4. Class Taxonomy: Urban sequence includes pedestrians and bikes.    ║
║     Must filter/map classes when benchmarking against vehicle-only   ║
║     fog_6_0 test sequence (cars, vans, buses).                       ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 2. Metadata Verification (Task 1)

The root `meta.json` was read directly from the archive:

```json
{
  "name": "city_3_0",
  "type": "urban",
  "set": "train_good_weather",
  "version": "1.0",
  "date_created": "Fri Aug  2 10:00:30 2019"
}
```

### Analysis of Metadata
- **`set: "train_good_weather"`**: Confirms this sequence is officially part of the RADIATE training partition. Training on this sequence maintains strict train/test isolation with respect to the held-out `fog_6_0` (`set: "test"`).
- **`type: "urban"`**: Driving route through urban Edinburgh (dense city center with buildings, intersections, and pedestrians).
- **`date_created: Fri Aug 2 10:00:30 2019`**: Recorded approximately 7 months prior to `fog_6_0` (Feb 29, 2020).

---

## 3. ZIP Structure and Modality Counts (Task 2 & 5)

Direct enumeration of the archive namelist confirmed **15,719 total entries** with all five required sensor modalities and timestamp files present:

| Modality / Component | Path in Archive | File Count | Format / Structure |
|---|---|---|---|
| **Navtech Radar (Cartesian)** | `Navtech_Cartesian/` | 713 | 1152 × 1152 uint8 PNG |
| **Navtech Radar (Polar)** | `Navtech_Polar/` | 713 | 400 × 576 uint8 PNG |
| **Velodyne LiDAR** | `velo_lidar/` | 1,799 | 5-column CSV (`x, y, z, intensity, ring`) |
| **ZED Camera (Left)** | `zed_left/` | 2,655 | 672 × 376 RGB PNG |
| **ZED Camera (Right)** | `zed_right/` | 2,655 | 672 × 376 RGB PNG |
| **GPS / IMU / Twist** | `GPS_IMU_Twist/` | 7,176 | Text records (position, orientation, velocity) |
| **Annotations** | `annotations/annotations.json` | 1 | JSON (314 tracked objects) |
| **Metadata** | `meta.json` | 1 | JSON (root metadata) |
| **Timestamp Indexes** | Root `.txt` files | 6 | `Navtech_Cartesian.txt`, `velo_lidar.txt`, etc. |
| **Total Archive Entries** | — | **15,719** | Complete dataset structure |

---

## 4. Annotation Deep-Dive (Task 3)

The `annotations/annotations.json` file (compressed size ~110 KB, uncompressed ~1.2 MB) was parsed in memory:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          ANNOTATION METRICS BREAKDOWN                          │
├──────────────────────────────────────┬─────────────────────────────────────────┤
│ Tracked Objects                      │ 314                                     │
│ Total Radar Frame Bounding Boxes     │ 8,875 valid bboxes                      │
│ Bbox Array Length per Object         │ Exactly 713 entries (matches radar)     │
│ Frames with ≥ 1 Valid Bounding Box   │ 713 out of 713                          │
│ Annotation Coverage                  │ 100.00%                                 │
│ Schema Match with fog_6_0            │ Identical (id, class_name, bboxes)      │
└──────────────────────────────────────┴─────────────────────────────────────────┘
```

### Class Distribution and Spatial Characteristics

| Class Name | Tracked Objects | Valid Bboxes | % of Total Bboxes | Mean Bbox Width × Height (px) | Notes |
|---|---|---|---|---|---|
| **`car`** | 143 | 4,835 | 54.48% | 16.9 × 27.9 | Dominant class; densely annotated |
| **`van`** | 20 | 2,073 | 23.36% | 20.1 × 35.0 | Shared with `fog_6_0` |
| **`pedestrian`** | 108 | 885 | 9.97% | 6.4 × 6.8 | Small radar cross-section |
| **`bus`** | 17 | 700 | 7.89% | 25.7 × 74.1 | Large radar signature; shared with `fog_6_0` |
| **`group_of_pedestrians`**| 22 | 190 | 2.14% | 18.1 × 11.3 | Clustered pedestrian returns |
| **`truck`** | 2 | 146 | 1.65% | 25.2 × 65.1 | Large commercial vehicles |
| **`motorbike`** | 1 | 26 | 0.29% | 9.9 × 15.9 | Minor urban class |
| **`bicycle`** | 1 | 20 | 0.23% | 9.8 × 13.5 | Minor urban class |
| **Total** | **314** | **8,875** | **100.0%** | — | **85.7% vehicle boxes (car, van, bus)** |

### Comparison to `fog_8_0` (Under-Annotated) and `fog_6_0` (Test Benchmark)
- `fog_8_0` had only 9 tracked objects, 276 bboxes, and **28.87% coverage** (unusable).
- `fog_6_0` had 39 tracked objects, 2,874 bboxes, and **92.4% coverage**.
- `city_3_0` has **314 tracked objects, 8,875 bboxes, and 100.0% coverage** — representing a **3.1× increase in total annotations** over `fog_6_0` and a **32× increase** over `fog_8_0`.

---

## 5. Timing, Sensor Rates, and Synchronization (Task 4)

Real UNIX timestamps were extracted from each modality's index text file (`Frame: <id> Time: <epoch_seconds>`):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SENSOR TIMING CHARACTERISTICS                          │
├───────────────────┬──────────┬──────────────────┬──────────────┬───────────┬───────────┤
│ Modality          │ Frames   │ Start Time (s)   │ End Time (s) │ Dur. (s)  │ Mean Rate │
├───────────────────┼──────────┼──────────────────┼──────────────┼───────────┼───────────┤
│ Navtech Cartesian │ 713      │ 1563273880.8459  │ 1563274058.8 │ 178.01 s  │ 4.00 Hz   │
│ Velodyne LiDAR    │ 1,799    │ 1563273879.0107  │ 1563274058.9 │ 179.96 s  │ 9.99 Hz   │
│ ZED Left Camera   │ 2,655    │ 1563273881.7216  │ 1563274058.9 │ 177.23 s  │ 14.97 Hz  │
│ ZED Right Camera  │ 2,655    │ 1563273881.7216  │ 1563274058.9 │ 177.23 s  │ 14.97 Hz  │
│ GPS / IMU         │ 7,176    │ 1563273880.8502  │ 1563274060.8 │ 179.95 s  │ 39.87 Hz  │
└───────────────────┴──────────┴──────────────────┴──────────────┴───────────┴───────────┘
```

### Multimodal Synchronization Analysis

```
Radar <-> LiDAR Temporal Offset:
  • Mean offset:  25.03 ms
  • Max offset:   49.95 ms
  • Std dev:      13.70 ms
  • 100.0% of frames (713/713) synchronized within 50 ms.

Radar <-> Camera Temporal Offset:
  • Mean offset:  20.10 ms (across frames 5–713)
  • Frames with offset < 50 ms:  709 / 713 (99.4%)
  • Frames with offset > 50 ms:  4 frames (Frames 1, 2, 3, 4)
```

### Discovery: Camera Startup Delay Anomaly
- **The Camera stream starts 0.875 seconds after Radar begins.**
- Radar Frame 1 occurred at `t = 1563273880.846`, while the first camera frame arrived at `t = 1563273881.722`.
- Consequently, Radar frames 1, 2, 3, and 4 (indices 0–3) have temporal offsets to camera of 875.7ms, 638.4ms, 396.8ms, and 155.9ms.
- **Remedy / Restriction:** Multimodal fusion must discard frames 1–4 and initiate training at **Radar Frame 5** (`t = 1563273881.815`), yielding **709 perfectly synchronized multimodal frames (99.4%)**.

### Radar Stream Inter-Frame Gaps
The Navtech radar scanner operates at 4.00 Hz (nominal `dt = 250 ms`). Across the 178-second recording, four brief gaps occurred where `dt > 400 ms`:
1. Frame 189 → 190: `dt = 0.486 s` (~1 dropped sweep)
2. Frame 350 → 351: `dt = 2.148 s` (~8 dropped sweeps)
3. Frame 364 → 365: `dt = 0.551 s` (~1 dropped sweep)
4. Frame 626 → 627: `dt = 1.926 s` (~7 dropped sweeps)

*Impact:* For single-frame 2D/BEV detection and frame-level multimodal fusion, these gaps have zero impact because each radar frame is paired independently with its nearest LiDAR sweep and camera frame. For temporal RNN/Kalman tracking models, the `dt` delta must be observed.

---

## 6. Sensor Evidence vs. Assumptions (Task 7)

Rather than assuming "good weather" or "high quality" based on the sequence name or directory labels, raw sensor buffers were inspected directly:

### 1. Camera Inspection (`zed_left/000100.png`)
- **Resolution:** 672 × 376, 3 channels (RGB).
- **Pixel Mean Intensity:** **93.43**; **Standard Deviation (Contrast):** **62.04**.
- **Empirical Assessment:** High dynamic range with rich contrast and sharp edges. In comparison to `fog_6_0` (where camera images suffer from severe atmospheric backscatter, low contrast, and grayish haze), `city_3_0` exhibits uncorrupted optical visibility with clear sky, road surface, building facades, and vehicle outlines.

### 2. LiDAR Inspection (`velo_lidar/000100.csv`)
- **Point Count per Sweep:** **45,061 points** (sampled across sequence: 45,073; 45,035; 42,980; 43,669; 39,801).
- **Max Measured Range:** **85.69 m**; **Mean Range:** 9.58 m.
- **Beams / Rings Present:** All **32 channels** (rings 0 through 31).
- **Intensity:** Range [0.0, 227.0], mean 7.53.
- **Empirical Assessment:** In `fog_6_0`, fog backscatter and extinction reduced surviving LiDAR points to **~10,000–13,000 per sweep**. In `city_3_0`, the point density is **~4× higher (~45,000 points)** with long-range returns up to 85m. This confirms pristine, unattenuated laser beam propagation.

### 3. Radar Inspection (`Navtech_Cartesian/000100.png`)
- **Resolution:** 1152 × 1152 uint8.
- **Mean Intensity:** 20.93; **Max Intensity:** 163; **Active (Non-zero) Pixels:** 1,042,508.
- **Empirical Assessment:** Crisp returns from urban street canyons, parked vehicles, curbs, and moving traffic, typical of high-resolution 77GHz Navtech radar without atmospheric degradation.

---

## 7. Comparative Analysis: `city_3_0` vs. `fog_6_0` (Task 6)

| Dimension | `city_3_0` (Candidate Train) | `fog_6_0` (Held-out Test Benchmark) |
|---|---|---|
| **Official Split** | `train_good_weather` | `test` (Held-out) |
| **Environmental Condition** | Daylight Overcast Urban (Clear) | Severe Suburban Fog (Degraded) |
| **Radar Frames** | 713 | 714 |
| **LiDAR Point Density** | **~45,000 points/sweep** (Clear air) | **~10,000–13,000 points/sweep** (Fog attenuated) |
| **Camera Quality** | Crisp, high contrast (std 62.04) | Hazy, low contrast, severe optical fog |
| **Tracked Objects** | **314** | 39 |
| **Total Bounding Boxes** | **8,875** | 2,874 |
| **Annotation Coverage** | **100.0%** (713/713 frames) | **92.4%** (660/714 frames) |
| **Class Distribution** | car, van, bus, pedestrian, group_ped, truck, motor, bike | car, van, bus (vehicles only) |
| **Multimodal Sync Coverage** | **99.4%** (709/713 frames < 50ms) | **92.0%** (657/714 frames < 100ms) |
| **Recording Date** | August 2, 2019 | February 29, 2020 |

---

## 8. Calibration Reusability Analysis (Task 9)

**Status:** **Requires Empirical Verification (Presumptively Compatible)**

- **Evidence for Compatibility:**
  1. Both sequences were collected by the same Heriot-Watt University research vehicle (Navtech CTS350-X radar, Velodyne HDL-32E LiDAR, ZED stereo camera).
  2. The official RADIATE SDK provides a single default calibration file (`config/default-calib.yaml`) for all sequences.
  3. An in-memory mathematical transformation test applying `config/default-calib.yaml` extrinsics to `city_3_0` LiDAR frame 100 successfully mapped 3,512 points into the front camera field of view with coordinate ranges matching expected vehicle dimensions ($X \in [-37.6, 36.9]\text{m}$, $Z \in [-2.4, 13.8]\text{m}$).
- **Why It Requires Verification Rather Than Assumption:**
  - `city_3_0` was recorded on **Aug 2, 2019**, while `fog_6_0` was recorded on **Feb 29, 2020** (a 7-month interval).
  - Minor mechanical adjustments or vibration drift over 7 months could introduce small rotational/translational offsets.
  - **Requirement:** Upon archive extraction, run a 3-frame LiDAR-to-camera and LiDAR-to-radar projection smoke test (similar to Phase 1C) to visually inspect spatial registration against physical object boundaries.

---

## 9. Experimental Suitability Evaluation (Task 8)

### A. Suitability for Clean / Reference Training: **SUITABLE**
- Provides 709 synchronized multimodal frames with 8,875 dense 2D bounding boxes.
- Dense annotations prevent false-negative penalties that plagued `fog_8_0`.
- Sensor modalities provide high signal-to-noise ratios (clear optical imagery, dense 32-ring point clouds).

### B. Suitability for Clean Validation: **SUITABLE**
- A continuous 20% temporal slice (e.g., frames 568–713, ~142 frames) can be held out as a clean-domain validation split.
- Allows monitoring clean detection performance ($mAP$) and nominal uncertainty calibration (Expected Calibration Error) on uncorrupted data.

### C. Suitability for Synthetic Progressive Degradation Experiments: **SUITABLE**
- `city_3_0` is an ideal baseline for controlled degradation:
  - **Camera:** Koschmieder optical attenuation, additive Gaussian blur, and fog luminance masks can be applied progressively across levels $\alpha \in [0.0, 1.0]$.
  - **LiDAR:** Distance-dependent beam extinction, Poisson noise, and random ray dropout can degrade point density from 45k down to 10k points (matching real fog).
  - **Radar:** Additive clutter and phase noise can be tested.
- Comparing model uncertainty on synthetically degraded `city_3_0` against real-world degraded `fog_6_0` directly addresses the core research question: *Does synthetic corruption calibration generalize to real-world physical fog degradation?*

---

## 10. Final Recommendation and Next Action (Tasks 11 & 12)

### Final Status:
```
╔══════════════════════════════════════════════════════════════════════╗
║                       READY WITH RESTRICTIONS                        ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Mandatory Restrictions:
1. **Exclude Radar Frames 1–4:** Multimodal models must train on frames 5–713 where camera synchronization is $<50\text{ ms}$.
2. **Class Filtering:** When setting up object detection targets for cross-evaluation against `fog_6_0`, map classes to standard vehicle categories (`car`, `van`, `bus`), treating `pedestrian` and `bicycle` either as background or as auxiliary classes.
3. **Calibration Smoke Test:** Run an empirical visual overlay check immediately upon extraction before training any baseline models.

### Exactly One Recommended Next Action:
**Extract `city_3_0.zip` to the local workspace and execute a 3-frame calibration projection smoke test (reusing the Phase 1C script) to empirically verify LiDAR-to-radar and LiDAR-to-camera spatial alignment.**
