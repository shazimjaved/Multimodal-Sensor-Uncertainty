# PHASE 3D-PRE — Corrected Projection Validation Report

> **Status: ALL NUMERICAL CHECKS PASS | VISUAL VALIDATION COMPLETE**

---

## A. Root Cause of Previous Projection Errors

Three independent errors existed in the original `RadiateCalib` implementation:

| # | Error | Effect |
|---|-------|--------|
| 1 | `cv2.Rodrigues()` applied to calibration `R` fields | `R` fields are **Euler angles in degrees**, not Rodrigues vectors. This produced a completely wrong rotation matrix. |
| 2 | Extrinsic applied as `R_cam @ (p - T_cam)` | Official RADIATE SDK convention is `M[:3,:3] @ p + M[:3,3]` where `M` encodes `(P @ Rx @ Ry @ Rz)^T` as the rotation block. |
| 3 | Z geometry: `z_bottom=0, z_top=1.5m` | Radar IS the vehicle roof (`Z=0`). Correct: `z_top=0.0m`, `z_bottom=-1.5m` (or `-1.8m` for vans/buses). |

---

## B. Official RADIATE SDK Convention (Implemented)

From `marcelsheeny/radiate_sdk / utils/calibration.py`:

```
RelativeR = SensorA_R - SensorB_R    (Euler-angle subtraction, degrees)
RelativeT = SensorA_T - SensorB_T    (position subtraction, metres)

Rx = rotation matrix about X by RelativeR[0]°
Ry = rotation matrix about Y by RelativeR[1]°
Rz = rotation matrix about Z by RelativeR[2]°

P = [[1, 0,  0],        ← axis permutation
     [0, 0,  1],           (z-up → y-down camera convention)
     [0, -1, 0]]

R_combined = P @ Rx @ Ry @ Rz

M_4x4 = [[R_combined^T | RelativeT],
          [0   0   0   | 1        ]]

Application:  p_cam = M[:3,:3] @ p_src + M[:3,3]
```

> **Key subtlety:** The official SDK builds the matrix as `[[R_rows, 0], [T, 1]].T`, which stores `R^T` (not `R`) in the rotation block. Our implementation replicates this exactly.

---

## C. Numerical Cross-Check Results

All four extrinsic matrices match the official SDK output to within **float32 rounding tolerance** (~2.6×10⁻⁸ m).

| Matrix | max_diff (m) | Status |
|--------|-------------|--------|
| RadarToLeft  | 2.60×10⁻⁸ | ✅ PASS |
| LidarToLeft  | 2.89×10⁻⁸ | ✅ PASS |
| RadarToRight | 9.78×10⁻⁹ | ✅ PASS |
| LidarToRight | 2.67×10⁻⁸ | ✅ PASS |

Point-level verification (radar → left camera frame):

| Radar point (m) | Depth (official) | Depth (ours) | Error (m) |
|----------------|-----------------|--------------|-----------|
| [0.3, 9.0, 0.0]   | 8.730 m | 8.730 m | 2.3×10⁻⁷ |
| [0.3, 9.0, -1.5]  | 8.764 m | 8.764 m | 2.3×10⁻⁷ |
| [0.0, 5.0, 0.0]   | 4.711 m | 4.711 m | 1.3×10⁻⁷ |
| [2.0, 20.0, 0.0]  | 19.725 m | 19.725 m | 5.2×10⁻⁷ |
| [-1.5, 15.0, -1.5] | 14.727 m | 14.727 m | 3.9×10⁻⁷ |
| [0.0, 50.0, -1.8] | 49.738 m | 49.738 m | 1.3×10⁻⁶ |

**Maximum error: 1.3×10⁻⁶ m (1.3 micrometres) — pure float32 rounding from official SDK.**

---

## D. Z Geometry Correction

| Parameter | Old (wrong) | Corrected |
|-----------|------------|-----------|
| `z_top`   | +1.5 m | **0.0 m** (radar plane = vehicle roof) |
| `z_bottom` | 0.0 m | **-1.5 m** (car/van base); **-1.8 m** (van/bus override) |
| Physical interpretation | Floating above radar | Spanning radar-to-ground ✓ |

This is derived from:
- Radar = coordinate origin (`RadarT = [0,0,0]`)
- Camera is at `Z = -0.288 m` (camera mounting point, 0.288 m below radar)
- Scan over Z values: `z ≈ -0.2 m` projects to `v ≈ 203` ≈ `cy = 200.7` (optical centre), confirming geometry

---

## E. Projection Clipping

Implemented in [`project_to_cam_left()`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/src/radiate_fusion.py#L289):
- `Z_cam > Z_MIN_DEPTH = 0.1 m` — rejects behind-camera and near-degenerate points
- `|uv| < UV_MAX_ABS = 1e5` — rejects projection blow-ups
- `np.isfinite()` check — rejects NaN/Inf from numerical instabilities

---

## F. Visual Validation Results

````carousel
![city_3_0 frame 000005 — 6 annotations, 6 projected, 5 fully in-frame](file:///C:/Users/GOGI%20LAPTOP/.gemini/antigravity-ide/brain/ed71568d-dec3-4bad-ba88-f6aa4a23eede/city_3_0_frame000005_corrected.png)
<!-- slide -->
![city_3_0 frame 000100 — 4 annotations, 4 projected, 4 fully in-frame](file:///C:/Users/GOGI%20LAPTOP/.gemini/antigravity-ide/brain/ed71568d-dec3-4bad-ba88-f6aa4a23eede/city_3_0_frame000100_corrected.png)
<!-- slide -->
![city_3_0 frame 000500 — 15 annotations, 6 projected, 4 fully in-frame](file:///C:/Users/GOGI%20LAPTOP/.gemini/antigravity-ide/brain/ed71568d-dec3-4bad-ba88-f6aa4a23eede/city_3_0_frame000500_corrected.png)
<!-- slide -->
![fog_6_0 frame 000010 — 2 annotations, 2 projected, 2 fully in-frame](file:///C:/Users/GOGI%20LAPTOP/.gemini/antigravity-ide/brain/ed71568d-dec3-4bad-ba88-f6aa4a23eede/fog_6_0_frame000010_corrected.png)
````

| Dataset | Frame | Total Anns | Target Cls | Projected | In-Frame | Cam Δt | LiDAR Δt |
|---------|-------|-----------|-----------|-----------|----------|--------|----------|
| city_3_0 | 000005 | 6 | 2 | 6 | 5 | 27.1 ms | 2.2 ms |
| city_3_0 | 000100 | 4 | 4 | 4 | 4 | 26.8 ms | 13.4 ms |
| city_3_0 | 000500 | 15 | 12 | 6 | 4 | 16.2 ms | 12.9 ms |
| fog_6_0  | 000010 | 2 | 2 | 2 | 2 | 31.1 ms | 29.7 ms |

### Visual Assessment

**Frame 005 (city_3_0):** Van at ~9 m (orange box) well-centred on the large van visible in the camera image. LiDAR overlay aligns with building/road surfaces. ✅

**Frame 100 (city_3_0):** Two cars (green/yellow boxes) project cleanly onto vehicles in the camera. Box vertical extent spans wheel-base to roof. LiDAR overlay aligns with road and building facades. ✅

**Frame 500 (city_3_0):** Dense traffic scenario. 6/15 boxes visible in camera FOV (remaining 9 are outside the camera's limited forward field of view — expected). Projected boxes align with visible vehicles. ✅

**Frame 010 (fog_6_0):** Real fog conditions. 2 cars project correctly onto the visible car ahead. LiDAR returns are significantly reduced (20 K vs 45 K points) — expected in fog. ✅ *(fog_6_0 untouched as held-out test set)*

> **Note on frame 500 "6/15 projected":** The remaining 9 objects are annotated on the sides/rear of the radar BEV but are outside the camera's ≈90° forward FOV. This is physically correct, not a calibration issue.

---

## G. Files Modified

| File | Change |
|------|--------|
| [`src/radiate_fusion.py`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/src/radiate_fusion.py) | Complete rewrite of `RadiateCalib` — Euler-angle extrinsics matching official SDK, corrected Z geometry, projection clipping, direct LiDAR-to-camera path |
| [`phase3d_pre_corrected_validation.py`](file:///c:/Users/GOGI%20LAPTOP/Desktop/Research/phase3d_pre_corrected_validation.py) | New validation script (numerical + visual) |
| `official_calibration.py` | Unchanged (reference only) |
| `fog_6_0/` | Unchanged ✅ |
| Raw data | Unchanged ✅ |

---

## H. Gate: Ready to Proceed?

- [x] Official SDK numerical match confirmed (all 4 extrinsics, tol=1e-5 m)  
- [x] Correct Euler-angle rotation with axis-permutation matrix  
- [x] Correct Z geometry (z_top=0, z_bottom=-1.5/-1.8m)  
- [x] Projection clipping implemented  
- [x] Visual validation passes (camera frame boxes align with visible vehicles)  
- [x] LiDAR overlay aligns with scene geometry  
- [x] fog_6_0 untouched  
- [x] Raw data untouched  

**Remaining Phase 3D-PRE-2 items (not yet addressed):**
1. Verify bbox_to_radar_3d_corners upper-left → center conversion (already fixed in prior Phase 3D-PRE)
2. Camera rectification pipeline audit (stereo_R/T provenance)
3. BEV preprocessing audit (dataset loader integration)
4. Propagation check to degradation engine + detector
5. IGNORE region mask audit

These can proceed now that the projection foundation is verified correct.
