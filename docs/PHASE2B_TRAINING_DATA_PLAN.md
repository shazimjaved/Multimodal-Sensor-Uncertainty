# PHASE 2B — RADIATE Training Sequences Identification and Data Plan

**Date:** 2026-09-24
**Constraint:** `fog_6_0` is the held-out test sequence. Must not be used for training or validation.

---

## 1. RADIATE Dataset Split Mechanism

The official RADIATE SDK uses a `meta.json` file in each sequence directory. The `"set"` field controls which partition a sequence belongs to:

```json
{ "set": "train_good_weather" }   // Sunny/overcast scenes
{ "set": "train_bad_weather"  }   // Night, rain, fog, snow
{ "set": "test"               }   // Held-out evaluation
```

The [`vehicle_detection/train.py`](https://github.com/marcelsheeny/radiate_sdk/blob/master/vehicle_detection/train.py) in the official SDK iterates over all sequence directories and reads `meta["set"]` to assign sequences to training or test. This is the authoritative split mechanism.

---

## 2. Complete RADIATE Sequence Inventory (22 Sequences)

### 2a. Good-Weather Training Sequences (5 sequences, `set: train_good_weather`)

| Sequence | Scenario | Radar frames | Size | LiDAR | Camera | Training use |
|---|---|---|---|---|---|---|
| `city_1_0` | Sunny Parked | ~540 | 1.7 GB | Good | Good | Pretraining backbone |
| `city_2_0` | Sunny Parked | ~540 | 1.7 GB | Good | Good | Pretraining backbone |
| `city_3_0` | Overcast Urban | ~720 | 2.3 GB | Good | Good | Rich annotation diversity |
| `city_4_0` | Overcast Urban | ~720 | 2.3 GB | Good | Good | Rich annotation diversity |
| `motorway_1_0` | Overcast Motorway | ~540 | 1.7 GB | Good | Good | High-speed scenario |

### 2b. Bad-Weather Training Sequences (9 sequences, `set: train_bad_weather`)

| Sequence | Scenario | Radar frames | Size | LiDAR | Camera | Priority |
|---|---|---|---|---|---|---|
| **`fog_1_0`** | **Fog Suburban** | **~700** | **2.1 GB** | **Fog-attenuated** | **Hazy** | **⭐ HIGHEST — same conditions as test** |
| **`fog_2_0`** | **Fog Suburban** | **~700** | **2.1 GB** | **Fog-attenuated** | **Hazy** | **⭐ HIGH — same conditions as test** |
| `fog_3_0` | Fog Suburban | ~700 | 2.1 GB | Fog-attenuated | Hazy | Medium — additional fog diversity |
| `fog_4_0` | Fog Suburban | ~700 | 2.1 GB | Fog-attenuated | Hazy | Medium |
| `fog_5_0` | Fog Suburban | ~700 | 2.1 GB | Fog-attenuated | Hazy | Low (if fog_1+2 sufficient) |
| `night_1_0` | Night Motorway | ~540 | 1.7 GB | Good | Dark/degraded | Optional cross-condition |
| `rain_1_0` | Rain Suburban | ~600 | 1.9 GB | Slightly degraded | Wet/blurred | Optional cross-condition |
| `snow_1_0` | Snow Suburban | ~600 | 1.9 GB | Severely attenuated | Snow/blizzard | Low priority |
| `snow_2_0` | Snow Suburban | ~600 | 1.9 GB | Severely attenuated | Snow/blizzard | Low priority |

### 2c. Test Sequences (8 sequences, `set: test` — DO NOT USE FOR TRAINING)

| Sequence | Scenario |
|---|---|
| `fog_6_0` | **Fog Suburban ← currently downloaded, audited, held out** |
| `city_5_0` | Overcast Urban |
| `city_6_0` | Overcast Urban |
| `motorway_2_0` | Overcast Motorway |
| `night_2_0` | Night Motorway |
| `rain_2_0` | Rain Suburban |
| `snow_3_0` | Snow Suburban |
| `snow_4_0` | Snow Suburban |

---

## 3. Candidate Analysis for Training/Validation

### Why fog sequences must be prioritized

The research question — *calibration stress testing under progressive sensor degradation in fog* — requires that the model has experience with foggy radar images during training. Using only good-weather training sequences would introduce a severe domain gap.

**Key property of fog in RADIATE:**
- Radar: **unaffected** by fog — returns are normal intensity
- LiDAR: **attenuated** — point density and range reduced
- Camera: **hazy** — visibility severely reduced
- This contrast (radar fine, camera/LiDAR degraded) is the core scientific signal

The training model must see this sensor discrepancy during training, or it cannot learn meaningful representations that generalize to fog_6_0.

### Calibration compatibility

All RADIATE sequences share the **same physical rig** and therefore the **same calibration parameters** (`config/default-calib.yaml`). This has been verified for fog_6_0 in Phase 1C and Phase 2. The identical calibration applies to all 22 sequences without modification.

---

## 4. Recommended Data Plans

### Plan A — Minimum Viable (fog-only training) ← **Recommended for prototype**

| Role | Sequences | Approx radar frames | Download size |
|---|---|---|---|
| **Train** | `fog_1_0` | ~700 | 2.1 GB |
| **Validation** | `fog_2_0` | ~700 | 2.1 GB |
| **Test** | `fog_6_0` (already downloaded) | 714 | — |
| **Total new download** | — | ~1,400 | **~4.2 GB** |

**Rationale:**
- fog_1_0 and fog_2_0 are explicitly `set: train_bad_weather` — same scenario type as fog_6_0
- All three sequences were collected in the same suburban location under fog
- Using fog_2_0 for validation provides a domain-matched val set
- The model sees the exact sensor degradation pattern it will be evaluated on
- 700 annotated radar frames is sufficient for a prototype object detection model
- Total additional download: **~4.2 GB** — practical on the available storage

> [!IMPORTANT]
> This is the **recommended minimum** for our calibration stress testing research. Using fog sequences for both train and val ensures that any uncertainty metrics measured on fog_6_0 are meaningful — the model has genuinely learned fog-domain representations.

### Plan B — Extended Fog (increased diversity)

| Role | Sequences | Approx radar frames | Download size |
|---|---|---|---|
| **Train** | `fog_1_0`, `fog_2_0`, `fog_3_0` | ~2,100 | 6.3 GB |
| **Validation** | `fog_4_0` | ~700 | 2.1 GB |
| **Test** | `fog_6_0` | 714 | — |
| **Total new download** | — | ~2,800 | **~8.4 GB** |

**Rationale:** 3x more training data; better handling of different fog density levels across sequences. Recommended if Plan A proves insufficient (under-fitting).

### Plan C — Full Bad-Weather (maximum generalization)

| Role | Sequences | Approx frames | Download size |
|---|---|---|---|
| **Train** | `fog_1_0`, `fog_2_0`, `fog_3_0`, `rain_1_0`, `night_1_0` | ~3,240 | ~9.8 GB |
| **Validation** | `fog_4_0`, `fog_5_0` | ~1,400 | ~4.2 GB |
| **Test** | `fog_6_0` | 714 | — |
| **Total new download** | — | ~4,640 | **~14.0 GB** |

**Rationale:** Includes cross-weather diversity; rain and night share suburban scenario with fog. Recommended for cross-weather generalization experiments, not the immediate prototype.

---

## 5. Important Notes on LiDAR Quality in Fog

From the official RADIATE documentation:

> *"Since the LiDAR signal could be severely attenuated and reflected by intervening fog or snow, the point cloud data may be missing, noisy and incorrect for some sequences in extreme weathers."*

This has direct implications for our research:
- Fog_6_0 audit confirmed ~10,000–13,000 LiDAR points per frame survive (fog-attenuated but not absent)
- Training sequences fog_1_0 and fog_2_0 will have similarly attenuated LiDAR
- This is **intentional and desirable** — the model should learn to function with degraded LiDAR
- Do NOT expect training LiDAR to match good-weather quality

---

## 6. Required Storage and Disk Planning

| Item | Size |
|---|---|
| fog_6_0 (already on disk) | 2.1 GB |
| fog_1_0 download (Plan A train) | ~2.1 GB |
| fog_2_0 download (Plan A val) | ~2.1 GB |
| **Plan A total on disk** | **~6.3 GB** |
| Available disk space needed | **~10 GB free** (extracted + working copies) |

> [!NOTE]
> These size estimates extrapolate from fog_6_0's measured 2.1 GB. Actual sizes may vary by ±20% depending on fog density and number of LiDAR points surviving attenuation.

---

## 7. Download and Registration Requirements

- **Registration required:** Same process as fog_6_0 — submit form at `https://pro.hw.ac.uk/radiate/downloads`
- **If already registered:** Use the same Dropbox link provided; fog_1_0 and fog_2_0 should be in the same Dropbox folder as fog_6_0
- **Calibration:** No additional calibration files needed — same `config/default-calib.yaml` used
- **Code changes:** None — the existing `src/radiate_fusion.py` and data loader will work with any RADIATE sequence unchanged

---

## 8. Post-Download Verification Checklist

For each new sequence downloaded, run:

1. `docs/PHASE2_FULL_DATA_AUDIT.md` checklist (count files, check timestamps, read meta.json)
2. Verify `meta.json["set"]` == `"train_bad_weather"` (not `"test"`)
3. Run synchronization check (all sensors Δt < 100ms)
4. Check annotation coverage (≥ 80% of radar frames annotated)
5. Run a quick 3-frame multimodal projection smoke test (reuse `smoke_test.py` with updated path)

---

## 9. Final Recommendation

```
╔══════════════════════════════════════════════════════════════════════╗
║              PLAN A — MINIMUM VIABLE (RECOMMENDED)                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  Download:   fog_1_0  (train)   ~2.1 GB                             ║
║              fog_2_0  (val)     ~2.1 GB                             ║
║  Total new:  ~4.2 GB                                                 ║
║  Total on disk after: ~6.3 GB (including fog_6_0)                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  Split:      TRAIN    → fog_1_0  (~700 annotated radar frames)       ║
║              VAL      → fog_2_0  (~700 annotated radar frames)       ║
║              TEST     → fog_6_0  (657 usable frames, HELD OUT)       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Calibration: existing config/default-calib.yaml — no changes        ║
║  Code:        existing src/radiate_fusion.py — no changes            ║
║  Conditions:  fog_1_0, fog_2_0, fog_6_0 are all fog_suburban        ║
║               → no domain gap between train/val and test             ║
╚══════════════════════════════════════════════════════════════════════╝
```

> **Next step:** Download fog_1_0 and fog_2_0 from the same Dropbox shared with you for fog_6_0. Once downloaded, run the Phase 2 audit script against each. Do not proceed to model development until both training sequences pass audit.
