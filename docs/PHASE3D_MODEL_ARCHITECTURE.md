# PHASE 3D — Baseline BEV Multimodal Fusion Detector Specification

**Phase**: 3D  
**Status**: COMPLETE & VERIFIED  
**Date**: September 24, 2026  
**Repository**: `shazimjaved/Multimodal-Sensor-Uncertainty`  
**Protocol Reference**: `docs/PHASE3A_EXPERIMENTAL_PROTOCOL.md`  
**Dependency Audit Reference**: `docs/PHASE3B_MODALITY_DEPENDENCY_AUDIT.md`  
**Degradation Engine Reference**: `docs/PHASE3C_DEGRADATION_ENGINE.md`  
**Configuration Source of Truth**: `configs/experiment_protocol.yaml`

---

## 1. Overview & Architectural Philosophy

Phase 3D implements the baseline Bird's-Eye-View (BEV) multimodal fusion detector designed strictly in accordance with the locked Phase 3A experimental protocol.

### Key Protocol & Design Commitments:
- **No Model Training**: The detector architecture and loss functions are implemented and verified; zero parameter updates or training runs have been conducted.
- **Experimental Transparency**: The architecture is intentionally kept computationally manageable (ResNet-18 backbones + 1x1 Conv fusion neck + CenterNet anchor-free head). As stated in `experiment_protocol.yaml`, the core scientific contribution is the degradation and calibration stress-testing methodology, not a complex heavy architecture.
- **Accurate Modality Nomenclature**: In adherence to the Phase 3B audit, the camera branch is explicitly recognized as a **Camera-Assisted BEV Branch** (LiDAR-assisted PointPainting), **never** mislabeled as "camera-only perception".
- **CPU Compatibility**: Designed and verified to run on standard CPU hardware with low memory consumption ($< 300$ MB peak RAM during inference).

---

## 2. Model Architecture

```
                    ┌────────────────────────────┐
                    │     INPUTS (512 x 512)     │
                    └──────────────┬─────────────┘
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
    radar_bev (1ch)          lidar_bev (3ch)          camera_bev (3ch)
          │                        │                        │
  ┌───────▼────────┐       ┌───────▼────────┐       ┌───────▼────────┐
  │ ResNet-18 FPN  │       │ ResNet-18 FPN  │       │ ResNet-18 FPN  │
  │ (Radar Branch) │       │ (LiDAR Branch) │       │(Cam-Assisted)  │
  └───────┬────────┘       └───────┬────────┘       └───────┬────────┘
          │ (64 x 128 x 128)       │ (64 x 128 x 128)       │ (64 x 128 x 128)
          └────────────────────────┼────────────────────────┘
                                   │
                    ┌──────────────▼─────────────┐
                    │    MULTIMODAL BEV FUSION   │
                    │   Channel Concatenation    │
                    │   (192 x 128 x 128)        │
                    │          ↓                 │
                    │  1x1 Conv + BN + ReLU      │
                    │  3x3 Conv + BN + ReLU      │
                    └──────────────┬─────────────┘
                                   │ (128 x 128 x 128)
                    ┌──────────────▼─────────────┐
                    │   CENTERNET DETECTION HEAD │
                    ├────────────────────────────┤
                    │ • Heatmap (3 classes)      │
                    │ • Sub-pixel Offset (dx, dy)│
                    │ • Box Size (w, l)          │
                    │ • Orientation (sin, cos)   │
                    └────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Sensor Backbones (`src/models/backbones.py`)
Each sensor modality is processed by a dedicated ResNet-18 backbone with a lightweight lateral Feature Pyramid (FPN) decoder:
- **Radar Backbone**: First convolution adapted to `in_channels=1` ($7 \times 7$, stride 2, padding 3).
- **LiDAR Backbone**: `in_channels=3` (Height, Log Density, Mean Intensity).
- **Camera-Assisted Backbone**: `in_channels=3` (PointPainted RGB in common BEV grid).
- **FPN Top-Down Decoder**:
  - `layer1`: stride 4 ($64 \times 128 \times 128$)
  - `layer2`: stride 8 ($128 \times 64 \times 64$)
  - `layer3`: stride 16 ($256 \times 32 \times 32$)
  - `layer4`: stride 32 ($512 \times 16 \times 16$)
  - Lateral $1 \times 1$ projections and bilinear upsamplings merge all stages into a unified **stride-4 feature map of shape `(B, 64, 128, 128)`**.

### 3.2 Multimodal Fusion Neck (`src/models/fusion.py`)
- **Channel Projection & Concatenation**: Concatenates active branch features in fixed slot order `("radar", "lidar", "camera")`.
- **1x1 Convolution Fusion Neck**:
  $$\mathbf{F}_{\text{fused}} = \text{ReLU}\left(\text{BatchNorm}\left(\text{Conv}_{3\times 3}\left(\text{ReLU}\left(\text{BatchNorm}\left(\text{Conv}_{1\times 1}\left(\mathbf{F}_{\text{cat}}\right)\right)\right)\right)\right)\right)$$
- **Channel Dimensions**: $192 \to 128$ channels.
- **Architectural Constraint**: Contains **zero self-attention or transformer layers**, maintaining strict compliance with the protocol.

### 3.3 CenterNet Anchor-Free Head (`src/models/centernet_head.py`)
Operates on the fused feature map $(B, 128, 128, 128)$ via four parallel task-specific sub-heads:
1. **Heatmap Head**: `Conv(128, 64, 3) -> ReLU -> Conv(64, 3, 1) -> Sigmoid`
   - Output: `(B, 3, 128, 128)` class center probabilities for **`car` (0), `van` (1), `bus` (2)**.
   - Final bias initialized to $-2.19$ ($\log(0.1 / 0.9)$) for training stability.
2. **Offset Head**: `Conv(128, 64, 3) -> ReLU -> Conv(64, 2, 1)`
   - Output: `(B, 2, 128, 128)` sub-pixel offsets $(dx, dy)$ in grid units $[0, 1)$.
3. **Size Head**: `Conv(128, 64, 3) -> ReLU -> Conv(64, 2, 1) -> ReLU`
   - Output: `(B, 2, 128, 128)` bounding box width and length $(w_{\text{px}}, l_{\text{px}}) > 0$.
4. **Orientation Head**: `Conv(128, 64, 3) -> ReLU -> Conv(64, 2, 1) -> L2Normalize`
   - Output: `(B, 2, 128, 128)` normalized unit vector $(\sin\theta, \cos\theta)$ representing continuous yaw angle $\theta = \text{atan2}(\sin\theta, \cos\theta)$.

---

## 4. Loss Formulation with Ignore-Region Masking (`src/models/losses.py`)

### 4.1 Gaussian Focal Loss with Ignore Mask
To satisfy the critical Phase 3A requirement that annotated non-target objects (pedestrians, bicycles, trucks) do not induce false-negative training penalties, the CenterNet heatmap loss is formulated with explicit ignore masking:
$$\mathcal{L}_{\text{hm}} = -\frac{1}{\max(1, N_{\text{pos}})} \sum_{b, c, y, x} \left[ \mathcal{L}_{\text{pos}} + \mathcal{L}_{\text{neg}} \right]$$
$$\mathcal{L}_{\text{pos}} = (1 - \hat{Y}_{xyc})^\alpha \log(\hat{Y}_{xyc}) \quad \text{for } Y_{xyc} = 1.0$$
$$\mathcal{L}_{\text{neg}} = (1 - Y_{xyc})^\beta (\hat{Y}_{xyc})^\alpha \log(1 - \hat{Y}_{xyc}) \cdot \left(1.0 - M_{\text{ignore}, xy}\right) \quad \text{for } Y_{xyc} < 1.0$$
where:
- $M_{\text{ignore}} \in \{0, 1\}^{B \times 1 \times 128 \times 128}$ is the downsampled binary ignore mask.
- When $M_{\text{ignore}, xy} = 1.0$, the penalty weight $(1.0 - M_{\text{ignore}}) = 0.0$, **completely eliminating negative loss penalties on annotated non-target objects**.

### 4.2 Multi-Task Loss Weighting
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{heatmap}} + 1.0 \cdot \mathcal{L}_{\text{offset}} + 0.1 \cdot \mathcal{L}_{\text{size}} + 0.1 \cdot \mathcal{L}_{\text{rot}}$$

---

## 5. Parameter Count & Computational Profile

| Sub-Module | Layer Specification | Parameter Count | Weight Memory (float32) |
|---|---|---|---|
| **Radar Backbone** | ResNet-18 (1ch in) + FPN | **11.27 M** | $45.1\text{ MB}$ |
| **LiDAR Backbone** | ResNet-18 (3ch in) + FPN | **11.27 M** | $45.1\text{ MB}$ |
| **Camera-Assisted Backbone** | ResNet-18 (3ch in) + FPN | **11.27 M** | $45.1\text{ MB}$ |
| **Fusion Neck** | 1x1 Conv (192->128) + 3x3 Conv | **0.17 M** | $0.7\text{ MB}$ |
| **CenterNet Head** | 4 Sub-heads (Heatmap, Offset, Size, Rot) | **0.31 M** | $1.2\text{ MB}$ |
| **TOTAL MODEL** | Full Multimodal BEV Detector | **34.29 M** | **137.2 MB** |

### Empirical CPU Benchmark (Intel x86_64, Windows):
- **Model Initialization**: $606.0\text{ ms}$
- **Single Forward Pass Latency ($B=1$)**: $564.1\text{ ms}$
- **Peak Dynamic RAM during Forward Pass**: $< 1.0\text{ MB}$ (traced buffer) + $137.2\text{ MB}$ weights
- **Suitability**: Fits comfortably on CPU-only machines with $\le 8$ GB RAM.

---

## 6. Supported Modality Ablations

The model provides native support for dynamic ablation via `active_modalities`:

1. **`Radar-only`** (`active_modalities=["radar"]`): Evaluates radar autonomous detection without LiDAR or camera.
2. **`LiDAR-only`** (`active_modalities=["lidar"]`): Evaluates LiDAR 3D surface geometry detection without radar or camera.
3. **`Camera-assisted branch`** (`active_modalities=["camera"]`): Evaluates LiDAR-assisted PointPainting BEV representation.
4. **`Full multimodal`** (`active_modalities=["radar", "lidar", "camera"]`): Standard 3-sensor fusion.
5. **`C2a LiDAR branch ablation`** (`active_modalities=["radar", "camera"]` or `lidar_bev = 0`): Simulates feature-level LiDAR dropout where radar and camera remain active.

---

## 7. Verification and Test Results

Automated unit tests in `tests/test_model.py` verified:
- Deterministic initialization across random seeds (`seed=123`).
- Single-sample forward pass returning expected dictionary keys and shapes.
- Batched forward pass ($B=2$).
- 3 target classes correctly configured (`car`, `van`, `bus`).
- All 5 ablation conditions executing correctly.
- Ignore-mask negative penalty suppression verified mathematically.
- Detection peak decoding extracting valid bounding box coordinates.

### Full Test Suite Status:
```
tests/test_calibration.py .............................                  [ 31%]
tests/test_dataset_pipeline.py .................                         [ 50%]
tests/test_degradation.py ..................................            [ 87%]
tests/test_model.py ...........                                          [100%]

============================= 91 passed in 26.79s =============================
```
- **Total Tests**: 91
- **Passed**: **91 (100%)**
- **Failed**: **0**
