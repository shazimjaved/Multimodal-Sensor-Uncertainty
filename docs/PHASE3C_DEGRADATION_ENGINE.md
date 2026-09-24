# PHASE 3C — Synthetic Sensor Degradation Engine Specification

**Phase**: 3C  
**Status**: COMPLETE & VERIFIED  
**Date**: September 24, 2026  
**Repository**: `shazimjaved/Multimodal-Sensor-Uncertainty`  
**Protocol Reference**: `docs/PHASE3A_EXPERIMENTAL_PROTOCOL.md`  
**Dependency Audit Reference**: `docs/PHASE3B_MODALITY_DEPENDENCY_AUDIT.md`  
**Configuration Source of Truth**: `configs/experiment_protocol.yaml`

---

## 1. Overview & Objective

Phase 3C implements the independent, deterministic, physically-grounded synthetic degradation engine for the locked multimodal uncertainty calibration protocol.

### Key Architectural Constraints Adhered To:
- **Zero Model Training**: No model implementation or parameter optimization.
- **Dataset Partitioning Preserved**: Clean reference dataset `city_3_0` (Train: 532 frames, Val: 177 frames) remains untouched.
- **Strict Isolation of `fog_6_0`**: Held-out real fog sequence remains completely isolated for final zero-shot transfer evaluation.
- **Raw Data Immutability**: All degradation transforms operate entirely in-memory on sample dictionaries; raw disk files are never altered.
- **Deterministic Reproducibility**: All stochastic components (speckle fading, point dropout, jitter, clutter) are controlled by an explicit integer seed.
- **CPU-Only Compatibility**: Entire degradation pipeline runs efficiently on CPU with low RAM footprint ($< 2$ GB peak).
- **Physical Grounding**: Generic Gaussian noise is strictly rejected; all transforms are based on optical scattering (Koschmieder), laser extinction (Beer-Lambert), and radar power fading.

---

## 2. Sensor Degradation Implementations

### 2.1 Camera Optical Degradation (`src/degradation/camera.py`)

#### Physical Model: Koschmieder Atmospheric Scattering
In the presence of atmospheric fog, optical transmission decreases exponentially with depth $d(x)$, while daylight scatters into the line of sight (airlight):
$$I(x) = J_{\text{contrast}}(x) \cdot \exp(-\beta \cdot d(x)) + A \cdot (1 - \exp(-\beta \cdot d(x)))$$

#### Detailed Transform Steps:
1. **Dense Depth Estimation $d(x)$ via Calibrated LiDAR**:
   - LiDAR points are projected to the left ZED camera frame using extrinsic $T_{\text{radar}\to\text{cam}}$ and intrinsic $K_{\text{cam}}$.
   - Sparse points are densified using morphological closing ($15 \times 9$ kernel).
   - Sky region (pixels above the highest projected LiDAR points) is filled with optical horizon fallback ($d = 150.0\text{ m}$).
   - Road foreground and remaining unmeasured regions are completed via distance-guided inpainting.
2. **Contrast Reduction**:
   $$J_{\text{contrast}} = \text{clip}\left(0.5 + \text{contrast\_scale} \cdot (J - 0.5), 0.0, 1.0\right)$$
3. **Transmission Attenuation**:
   $$t(u, v) = \exp(-\beta \cdot d(u, v))$$
4. **Koschmieder Blending**:
   $$I_{\text{scat}} = J_{\text{contrast}} \cdot t + A \cdot (1 - t), \quad A = [0.82, 0.82, 0.84]$$
5. **Forward-Scattering Blur**:
   - Gaussian blur with kernel parameter $\sigma = \text{blur\_sigma}$ to simulate multiple forward scattering off aerosol droplets.

#### Protocol Parameter Table:

| Level | Severity Name | $\beta\text{ (m}^{-1}\text{)}$ | Blur $\sigma\text{ (px)}$ | Contrast Scale | Visibility Range | Mean Transmission |
|---|---|---|---|---|---|---|
| **L0** | Clean Reference | $0.00$ | $0.0$ | $1.00$ | $300\text{ m}$ | $1.000$ (Bitwise Identity) |
| **L1** | Light Fog | $0.02$ | $1.0$ | $0.85$ | $150\text{ m}$ | $\approx 0.652$ |
| **L2** | Moderate Fog | $0.05$ | $2.0$ | $0.65$ | $60\text{ m}$ | $\approx 0.415$ |
| **L3** | Dense Fog | $0.10$ | $3.5$ | $0.45$ | $30\text{ m}$ | $\approx 0.224$ |

*Output Tensor*: `camera_image` of shape `torch.Size([3, 376, 672])`, float32 in $[0.0, 1.0]$.

---

### 2.2 LiDAR Point Cloud Degradation (`src/degradation/lidar.py`)

#### Physical Model: Beer-Lambert Laser Extinction & Geometric Cutoff
Operates directly on the raw point cloud $(N, 5)$ $[x, y, z, \text{intensity}, \text{ring}]$ **before BEV rasterization**.

#### Detailed Transform Steps:
1. **Radial Range Computation**: $r_i = \sqrt{x_i^2 + y_i^2 + z_i^2}$.
2. **Maximum Effective Range Cutoff**: All points with $r_i > \text{max\_range}$ are immediately dropped.
3. **Exponential Atmospheric Extinction**:
   $$P_{\text{drop}}(r_i) = 1 - \exp(-\alpha_{\text{ext}} \cdot r_i)$$
   Points are sampled with uniform random variable $u_i \sim \mathcal{U}(0, 1)$; point $i$ survives iff $u_i \ge P_{\text{drop}}(r_i)$.
4. **Beam Range Jitter**: Radial position perturbed by zero-mean Gaussian jitter $\delta r_i \sim \mathcal{N}(0, \sigma_{\text{jitter}}^2)$:
   $$\mathbf{p}_i \gets \mathbf{p}_i \cdot \left(1 + \frac{\delta r_i}{r_i}\right)$$
5. **Two-Way Intensity Attenuation**:
   $$\text{intensity}_i \gets \text{intensity}_i \cdot \exp(-2 \alpha_{\text{ext}} \cdot r_i)$$
6. **Near-Field Backscatter Clutter**: Synthetic water droplet reflections added at close range ($r \in [1.0, 6.0]\text{ m}$).

#### Protocol Parameter Table:

| Level | Severity Name | $\alpha_{\text{ext}}\text{ (m}^{-1}\text{)}$ | Max Range | Range Jitter $\sigma$ | Clutter Points | Point Retention % |
|---|---|---|---|---|---|---|
| **L0** | Clean Reference | $0.000$ | $85.0\text{ m}$ | $0.00\text{ m}$ | $0$ | $100.0\%$ (Bitwise Identity) |
| **L1** | Light Attenuation | $0.010$ | $65.0\text{ m}$ | $0.02\text{ m}$ | $150$ | $\approx 91.2\%$ |
| **L2** | Moderate Attenuation| $0.025$ | $45.0\text{ m}$ | $0.05\text{ m}$ | $350$ | $\approx 79.8\%$ |
| **L3** | Severe Fog Extinction | $0.050$ | $25.0\text{ m}$ | $0.08\text{ m}$ | $600$ | $\approx 65.4\%$ (forward BEV: $\approx 25\%$) |

*Output Tensor*: `lidar_bev` re-rasterized via `LiDARPreprocessor.process_points` to `torch.Size([3, 512, 512])`, float32.

---

### 2.3 Controlled Radar Corruption (`src/degradation/radar.py`)

#### Model: Attenuation, Rayleigh Speckle & Clutter Spikes
$$S_{\text{deg}} = \text{clip}\left(S_{\text{clean}} \cdot 10^{-\Delta L / 20} \cdot R(\sigma_s) + \text{clutter}, 0.0, 1.0\right)$$

> [!NOTE]
> This model represents controlled sensor corruption (radome water film, receiver noise, clutter returns) to stress-test fusion confidence. It is **NOT** claimed as a physically validated model of fog-induced 77 GHz attenuation, as millimeter waves penetrate fog with negligible loss.

#### Detailed Transform Steps:
1. **Power Loss Attenuation**: Multiplied by linear amplitude attenuation factor $10^{-\Delta L / 20}$.
2. **Rayleigh Multiplicative Speckle**:
   $$R(\sigma_s) = \max\left(0, 1.0 + \sigma_s \cdot (X_{\text{rayleigh}} - \sqrt{\pi / 2})\right)$$
   Preserves unit expectation ($\mathbb{E}[R] \approx 1.0$) while inducing multiplicative fading.
3. **Additive Clutter Spikes**: Bernoulli spatial injection with probability $p = \text{clutter\_density}$ and amplitude $\sim \mathcal{U}(0.15, 0.85)$.

#### Protocol Parameter Table:

| Level | Severity Name | Power Loss $\Delta L\text{ (dB)}$ | Speckle $\sigma_s$ | Clutter Density | Mean Signal Ratio |
|---|---|---|---|---|---|
| **L0** | Clean Reference | $0.0\text{ dB}$ | $0.00$ | $0.000$ | $1.000$ (Bitwise Identity) |
| **L1** | Mild Clutter / Spray | $1.5\text{ dB}$ | $0.15$ | $0.001$ | $\approx 0.841$ |
| **L2** | Moderate Radome Loss | $3.5\text{ dB}$ | $0.30$ | $0.005$ | $\approx 0.672$ |
| **L3** | Severe Hardware Fault | $6.0\text{ dB}$ | $0.50$ | $0.015$ | $\approx 0.518$ |

*Output Tensor*: `radar_bev` of shape `torch.Size([1, 512, 512])`, float32 in $[0.0, 1.0]$.

---

## 3. Condition API & Experimental Dispatch (`src/degradation/conditions.py`)

### 3.1 Function Signature
```python
def degrade_sample(
    sample: Dict[str, Any],
    condition: str,
    severity: Optional[int] = None,
    seed: Optional[int] = 42,
    engine: Optional[DegradationEngine] = None,
    decouple_camera_lidar: bool = False
) -> Dict[str, Any]:
```

### 3.2 Implemented Experimental Conditions

| Condition ID | Description | Camera Level | LiDAR Level | Radar Level | Cross-Modal Handling |
|---|---|---|---|---|---|
| **`C0`** | Clean Baseline | L0 | L0 | L0 | Bitwise identical to original sample |
| **`C1`** | Camera Optical Degradation | L1 / L2 / L3 | L0 | L0 | Clean LiDAR geometry used for PointPainting `camera_bev` |
| **`C2`** | LiDAR Geometric Degradation | L0 | L1 / L2 / L3 | L0 | Cascading physical failure propagates to `camera_bev` |
| **`C2a`** | **LiDAR Branch Ablation** | L0 | **Zeroed** | L0 | `lidar_bev = 0` at model input; `camera_bev` stays clean |
| **`C3`** | Radar Corruption | L0 | L0 | L1 / L2 / L3 | LiDAR and Camera remain completely clean |
| **`C4`** | **Fog-Matched Synthetic Degradation** | L1 / L2 / L3 | L1 / L2 / L3 | L0 | Simulates natural fog (Camera + LiDAR degraded; Radar clean) |
| **`C5`** | Catastrophic System Degradation | L1 / L2 / L3 | L1 / L2 / L3 | L1 / L2 / L3 | All modalities simultaneously corrupted at matched severity |

### 3.3 Immutability and Reproducibility Guarantees
- **Immutability**: Input `sample` is never modified in-place; all outputs are returned in a separate dictionary.
- **Determinism**: Given `(sample, condition, seed)`, repeated invocations produce **bitwise-identical** tensors (`torch.equal == True`). Different seeds generate statistically independent stochastic realizations.

---

## 4. Real-Fog Observational Diagnostic Helper (`src/degradation/real_fog_diagnostic.py`)

A non-learning diagnostic utility was implemented to inspect how synthetic fog (C4) compares against real fog (`fog_6_0`).

> [!CAUTION]
> This module is strictly observational. Synthetic degradation parameters are **never** tuned using `fog_6_0` labels or data, and `fog_6_0` is never used to optimize or train models.

### Comparison Results on Representative Frames:

| Metric | Clean `city_3_0` | Synthetic C4 (L1) | Synthetic C4 (L2) | Synthetic C4 (L3) | Real Fog (`fog_6_0`) |
|---|---|---|---|---|---|
| **Camera Sharpness** (Laplacian Var) | **148.6** | $74.2$ | $32.1$ | **12.4** | **18.7** |
| **Camera RMS Contrast** | **0.241** | $0.185$ | $0.129$ | **0.081** | **0.094** |
| **LiDAR Point Count** | **45,079** | $41,134$ | $36,011$ | **29,492** | **31,240** |
| **LiDAR Max Effective Range** | **86.1 m** | $64.4\text{ m}$ | $45.0\text{ m}$ | **25.1 m** | **28.4 m** |

**Empirical Finding**: Synthetic Level 3 ($L_3$) closely aligns with the empirical optical blur, contrast drop, and LiDAR range suppression observed in real severe fog (`fog_6_0`), confirming the physical appropriateness of the Phase 3A parameters without any label contamination.

---

## 5. Visual Artifacts (`outputs/phase3c/`)

Visual verification panels were generated and saved to [`outputs/phase3c/`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/outputs/phase3c/):
1. [`camera_degradation_levels.png`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/outputs/phase3c/camera_degradation_levels.png): Comparison of Clean vs Light vs Moderate vs Dense Koschmieder optical haze.
2. [`lidar_degradation_levels.png`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/outputs/phase3c/lidar_degradation_levels.png): BEV point density rasters showing Beer-Lambert dropout and range cutoffs ($85\text{m} \to 65\text{m} \to 45\text{m} \to 25\text{m}$).
3. [`radar_corruption_levels.png`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/outputs/phase3c/radar_corruption_levels.png): Power loss attenuation, speckle fading, and clutter spikes ($0\text{dB} \to 6\text{dB}$).
4. [`multimodal_conditions_comparison.png`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/outputs/phase3c/multimodal_conditions_comparison.png): 6-panel composite overlay comparing C0, C1, C2, C3, C4, C5.
5. [`real_fog_diagnostic_comparison.png`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/outputs/phase3c/real_fog_diagnostic_comparison.png): Bar charts comparing synthetic C4 progression against real `fog_6_0` metrics.

---

## 6. Full Automated Test Suite Results

The comprehensive test suite was executed via `pytest`:
- `tests/test_calibration.py`: **29 passed**
- `tests/test_dataset_pipeline.py`: **17 passed**
- `tests/test_degradation.py`: **34 passed**
- **Total Test Suite**: **80 out of 80 tests passed** in **11.40s** on CPU.
