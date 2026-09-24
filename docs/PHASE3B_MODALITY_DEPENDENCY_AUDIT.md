# PHASE 3B PATCH — Evaluation Population & Cross-Modal Dependency Audit

**Document ID**: `PHASE3B_MODALITY_DEPENDENCY_AUDIT`  
**Date**: September 24, 2026  
**Status**: APPROVED & LOCKED  
**Reference Protocol**: `docs/PHASE3A_EXPERIMENTAL_PROTOCOL.md`  
**Dataset Reference**: `docs/PHASE3B_DATA_PIPELINE.md`  
**Supporting CSV**: `outputs/phase3b/modality_dependency_audit.csv`

---

## Executive Summary

This audit addresses two methodological questions identified following the implementation of Phase 3B:
1. **The apparent population discrepancy in `fog_6_0`** between Phase 2 (657 frames) and Phase 3B (711 frames), providing an exact empirical reconciliation without altering historical audit values.
2. **The architectural cross-modal dependency** between LiDAR and Camera in the PointPainting BEV representation, analyzing its consequences for sensor degradation experiments (C1–C4) and formally recommending Strategy D (explicit, transparent protocol formulation) for the 4-page IEEE conference paper.

---

## PART 1 — `fog_6_0` Evaluation Population Audit

### 1.1 The Context of the Discrepancy
- **Phase 2 Audit (`docs/PHASE2_FULL_DATA_AUDIT.md`, line 113)** reported:  
  `Usable multimodal frames (Δt < 100ms AND ≥ 1 annotation): 657 frames`
- **Phase 3B Indexer (`src/dataset/indexer.py`)** reported:  
  `711 synchronized frames at Δt ≤ 50ms`

Both numbers are empirically accurate within their respective definitions. The discrepancy is purely a distinction between **all synchronized frames** (including empty road scenes) versus **synchronized frames with active object annotations**.

---

### 1.2 Exact Population Breakdown

An exhaustive frame-by-frame audit of `fog_6_0` was conducted across timestamp files and `annotations.json`:

```
Total Radar Frames in Navtech_Cartesian.txt: 714 (Frames 000001–000714)
 │
 ├── [Desynchronized: 3 frames] (Frames 1, 2, 3: camera startup delay Δt_cam > 50ms)
 │     Frame 000001: Δt_cam = 683.1 ms (excluded)
 │     Frame 000002: Δt_cam = 450.2 ms (excluded)
 │     Frame 000003: Δt_cam = 213.8 ms (excluded)
 │
 └── [Synchronized at Δt ≤ 50ms: 711 frames] (Frames 000004–000714)
       │
       ├── [Empty Road / Zero Annotations: 54 frames]
       │     Frame 000077 (1 frame)
       │     Frames 000114–000128 (15 frames)
       │     Frames 000579–000593 (15 frames)
       │     Frames 000692–000714 (23 frames at sequence end)
       │
       └── [Annotated Target Frames: 657 frames]
             Contains ≥ 1 ground-truth target object (car, van, bus)
             Target class composition:
               - car: 31 tracked instances
               - van: 7 tracked instances
               - bus: 1 tracked instance
             (Pedestrians / trucks / bikes: 0 instances in fog_6_0)
```

### 1.3 Formal Population Definitions

| Population Identifier | Population Name | Frame Count | Criteria / Filter Condition |
|---|---|---|---|
| **Population A** | Total Radar Sequence | **714** | All recorded radar PNG frames in `fog_6_0/Navtech_Cartesian/` (Frames 1–714) |
| **Population B** | Synchronized Multimodal Frames | **711** | Frames with $\|\Delta t_{\text{lidar}}\| \le 50\text{ ms}$ AND $\|\Delta t_{\text{cam}}\| \le 50\text{ ms}$ (Frames 4–714) |
| **Population C** | Frames with Valid Annotations | **660** | Frames with $\ge 1$ 3D bounding box in `annotations.json` (Frames 1–76, 78–113, 129–578, 594–691) |
| **Population D** | Frames with Target Classes | **660** | Frames containing $\ge 1$ `car`, `van`, or `bus`. (Identical to Population C as `fog_6_0` contains only vehicles) |
| **Population E** | **Primary Evaluation Benchmark** | **657** | **Intersection of Population B and Population D**: Synchronized ($\le 50\text{ ms}$) AND $\ge 1$ target annotation |
| **Population F** | **False-Alarm Diagnostic Set** | **54** | Synchronized ($\le 50\text{ ms}$) BUT zero annotations (empty roadway negative scenes) |
| **Population G** | Hardware Startup Exclusion | **3** | Frames 1, 2, 3 where ZED camera initialization latency exceeded $50\text{ ms}$ |

---

### 1.4 Comparison with Training/Validation Sequence (`city_3_0`)

To demonstrate methodological symmetry, the identical audit was executed on `city_3_0`:

| Sequence Split | Total Radar | Sync ($\le 50\text{ms}$) | Annotated | Target Annotated | Eval Population | Excluded Reason |
|---|---|---|---|---|---|---|
| **`city_3_0` Train** | 532 | 532 | 532 (100%) | 532 (100%) | **532** | None (frames 000005–000536) |
| **`city_3_0` Val** | 177 | 177 | 177 (100%) | 177 (100%) | **177** | None (frames 000537–000713) |
| **`city_3_0` Startup** | 4 | 0 | 4 (100%) | 4 (100%) | **0** | Desynchronized startup (frames 1–4, $\Delta t_{\text{cam}} > 50\text{ms}$) |
| **`city_3_0` Combined**| 713 | 709 | 713 (100%) | 713 (100%) | **709** | 4 startup frames excluded |
| **`fog_6_0` Test** | 714 | 711 | 660 (92.4%)| 660 (92.4%)| **657** | 3 startup frames desynced; 54 frames unannotated |

---

### 1.5 Protocol Specification for Final Evaluation

For the final detection and calibration evaluation on held-out `fog_6_0`:

1. **Primary Object Detection & Calibration Metrics (mAP, ECE, Brier, NLL, Recall, Precision)**:
   - Evaluated strictly on **Population E (657 frames)**.
   - **Scientific Justification**: Evaluating True Positives (TP), False Negatives (FN), and prediction-event calibration requires confirmed ground-truth target bounding boxes. Calculating precision-recall curves or matching prediction events against an empty ground truth produces undefined division-by-zero or distorts recall statistics.
2. **False Alarm & Specificity Diagnostic Metric (FP / Frame)**:
   - Evaluated on **Population F (54 frames)** as a dedicated negative-sample stress test.
   - **Scientific Justification**: Autonomous perception models under fog frequently hallucinate ghost obstacles due to backscatter clutter. The 54 empty frames provide a clean baseline to quantify the False Positive per Frame (FPPI) and overconfidence of ghost detections without confounding target overlaps.

---

## PART 2 — Cross-Modal Dependency Audit (Camera BEV via LiDAR)

### 2.1 The Underlying Dependency
In Phase 3B, `CameraPreprocessor` maps RGB color from the front ZED camera into the common $512 \times 512$ BEV grid using calibrated LiDAR point projection:
$$\mathbf{p}_{\text{cam}} = R_{\text{radar}\to\text{cam}}(R_{\text{lidar}\to\text{radar}}\mathbf{p}_{\text{lidar}} + \mathbf{t}_{\text{lidar}\to\text{radar}}) + \mathbf{t}_{\text{radar}\to\text{cam}}$$
$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z_{\text{cam}}} K_{\text{cam}} \mathbf{p}_{\text{cam}}$$
The resulting `camera_bev` raster is populated by sampling $(R, G, B)$ at pixel coordinates $(u, v)$ and placing them at the BEV grid coordinates $(X_{\text{lidar}}, Y_{\text{lidar}})$.

This approach mirrors **PointPainting** (Vora et al., CVPR 2020), a well-established and widely cited fusion paradigm. However, it creates a structural coupling: **`camera_bev` inherits its geometric coordinates $(X, Y)$ entirely from LiDAR points.**

---

### 2.2 Answering the Four Critical Questions

#### Q1: What happens to `camera_bev` if raw LiDAR is degraded before preprocessing?
- If LiDAR points are degraded prior to projection (e.g., beam dropouts, atmospheric attenuation, range reduction, backscatter clutter):
  - Dropped LiDAR points produce no image query $\to$ corresponding pixels in `camera_bev` remain empty.
  - Clutter LiDAR points (e.g., near-field fog backscatter) project rays through incorrect camera pixels, mapping irrelevant background/road textures into near-field BEV cells.
  - **Result**: Degraded LiDAR directly corrupts both `lidar_bev` AND `camera_bev`, even if the front camera optics are completely pristine.

#### Q2: What happens if LiDAR is degraded after `camera_bev` generation?
- If `camera_bev` is constructed using clean LiDAR, and subsequent synthetic degradation is applied only to `lidar_bev`:
  - `camera_bev` retains sharp, pristine LiDAR spatial coordinates painted with RGB color.
  - In a multimodal fusion network receiving `[radar_bev, lidar_bev, camera_bev]`, the network can bypass the degraded `lidar_bev` by extracting spatial bounding geometry from `camera_bev`.
  - **Result**: Clean LiDAR geometry leaks into degraded LiDAR experiments, creating an artificial information bypass that conceals the true vulnerability of the fusion model.

#### Q3: Do C1 (Camera-only), C2 (LiDAR-only), and C3 (Radar-only) experiments remain logically isolated?
- **C3 (Radar-only degradation / ablation)**: **Completely isolated**. Radar operates on its own 77 GHz FMCW transceiver grid; its degradation or zeroing affects zero LiDAR or camera coordinates.
- **C1 (Camera-only degradation)**: **Isolated photometrically**. Applying optical noise, contrast reduction, lens glare, or atmospheric haze to the front camera image degrades the RGB feature values painted into `camera_bev`, while preserving the underlying ray geometry.
- **C2 (LiDAR-only degradation)**: **NOT isolated**. Under naive pipelines, degrading LiDAR either breaks camera projection (Q1) or leaks clean LiDAR coordinates through camera BEV (Q2).
- **Single-Modality Baselines**: A "Camera-only" detector using `camera_bev` is not purely a camera detector—it is a LiDAR-guided multimodal detector.

#### Q4: Does the current representation introduce cross-modal dependency?
- **YES**. `camera_bev` is fundamentally a **hybrid sensor-assisted representation**, not a standalone single-sensor modality.

---

### 2.3 Evaluation of Potential Strategies (A–D)

| Dimension | Strategy A: Clean LiDAR Fixed for Camera | Strategy B: Degraded LiDAR Feeds Camera | Strategy C: Standalone Camera BEV (IPM / Depth Net) | Strategy D: Explicit Protocol Formulation (RECOMMENDED) |
|---|---|---|---|---|
| **Mechanism** | Generate `camera_bev` once using clean LiDAR; degrade only `lidar_bev`. | Recompute `camera_bev` using degraded LiDAR point cloud. | Eliminate LiDAR: use Inverse Perspective Mapping (IPM) or train a monocular depth net. | Retain PointPainting; formally define sensor roles & decouple branch ablations from physical tests. |
| **Scientific Validity** | **Low**: Leaks pristine spatial coordinates into LiDAR failure tests; invalid on real fog (`fog_6_0`). | **Moderate**: Models physical cascaded failure, but conflates LiDAR degradation with camera degradation. | **Poor (IPM)**: Severe vertical distortion ($1.5\text{m}$ vehicle smears $30\text{m}$); **Invalid (Depth Net)**: Violates protocol constraint against learned depth. | **Highest**: Transparently characterizes the architecture as PointPainting; isolates optical vs geometric vs branch failure modes cleanly. |
| **Implementation Complexity** | Very Low | Low | Extremely High (Learned depth) or Moderate (IPM homography) | **Very Low**: Zero new code required; uses validated deterministic pipeline. |
| **CPU Feasibility (~8 GB RAM)** | High (Fast) | High (Fast) | Infeasible for neural depth (requires GPU, heavy memory); Fast for IPM. | **Optimal**: Full suite runs in $<6$ seconds on CPU with $<2$ GB RAM. |
| **Suitability for 4-Page IEEE Paper** | **Poor**: Reviewers will criticize coordinate leakage during sensor failure tests. | **Fair**: Defensible as system-level cascading failure, but forfeits clean orthogonal ablations. | **Poor**: Space budget cannot accommodate explaining a depth net, or reviewers reject IPM distortion. | **Ideal**: PointPainting is universally recognized; upfront methodological honesty strengthens the paper's scientific rigor. |

---

### 2.4 Formal Recommendation: STRATEGY D

We formally recommend **Strategy D: Keep the validated representation and explicitly define the degradation & ablation protocol to account for cross-modal dependency**.

#### Concrete Protocol Definition Under Strategy D:

1. **Modality Characterization**:
   - **Radar**: *Independent Primary BEV Modality* ($1 \times 512 \times 512$). All-weather, long-range metric reference.
   - **LiDAR**: *Independent Primary BEV Modality* ($3 \times 512 \times 512$). High-precision surface geometry, density, reflectance.
   - **Camera Perspective**: *Independent Optical Modality* ($3 \times 376 \times 672$). High-resolution visual semantics.
   - **Camera BEV**: *LiDAR-Assisted Geometric Projection (PointPainting)* ($3 \times 512 \times 512$).
2. **Experimental Disentanglement**:
   - **Condition C1 (Camera Optical Degradation)**: Apply synthetic haze, contrast reduction, and optical noise to `camera_image` *before* projection. LiDAR geometry remains clean. This isolates the effect of **photometric/semantic degradation**.
   - **Condition C2 (LiDAR Geometric Degradation)**:
     - **Branch Ablation Test**: Zero out `lidar_bev = 0` at the fusion network input while keeping `camera_bev` intact. This tests network reliance on direct LiDAR point features vs. painted visual context.
     - **Physical Cascading Failure Test**: Degrade raw LiDAR points *before* projection, causing simultaneous geometric sparsity in both `lidar_bev` and `camera_bev`. This evaluates real-world cascaded degradation in sensor-assisted pipelines.
   - **Condition C3 (Radar Degradation)**: Apply backscatter clutter and attenuation to `radar_bev`. LiDAR and Camera remain completely untouched. This isolates all-weather radar vulnerability.
   - **Condition C4 (Natural Fog / fog_6_0)**: Real-world physical test where both LiDAR and camera naturally suffer simultaneous atmospheric degradation.
3. **Single-Modality Baselines**:
   - Radar-only: Trained/evaluated using `radar_bev` ($1 \times 512 \times 512$).
   - LiDAR-only: Trained/evaluated using `lidar_bev` ($3 \times 512 \times 512$).
   - Multimodal Fusion: Trained/evaluated using the full multi-channel BEV tensor.

---

## Deliverables Summary

1. `docs/PHASE3B_MODALITY_DEPENDENCY_AUDIT.md` (This document)
2. `outputs/phase3b/modality_dependency_audit.csv` (Exhaustive numerical table)
3. Minor clarification added to `docs/PHASE3B_DATA_PIPELINE.md` reflecting the 657-frame target evaluation population vs. the 711 synchronized population.
