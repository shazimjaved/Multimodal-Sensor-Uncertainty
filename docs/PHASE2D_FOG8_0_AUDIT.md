# PHASE 2D — fog_8_0 Archive-Only Verification

**Date:** 2026-09-24
**Target:** `C:\Users\GOGI LAPTOP\Desktop\Research\fog_8_0.zip`

This phase involved direct Python `zipfile` inspection of the `fog_8_0.zip` archive without performing a full extraction. The goal was to determine if the sequence is suitable as training data for the prototype, while keeping `fog_6_0` as the unseen test set.

---

## 1. Metadata Verification

The `meta.json` file was read directly from the root of the ZIP archive:

```json
{
  "name": "fog_8_0",
  "type": "fog",
  "set": "train_good_and_bad_weather",
  "version": "1.0",
  "date_created": "Fri Feb 28 23:01:18 2020"
}
```

**Training Validity:** The `set` value is `"train_good_and_bad_weather"`. This confirms that **fog_8_0 is officially designated as a training set** in the RADIATE conventions, not a test set. Using it for training is legally and methodologically clean with respect to the `fog_6_0` test set.

*(Note on creation dates: `fog_8_0` was created on Friday Feb 28, 23:01, while `fog_6_0` was created on Saturday Feb 29, 03:09. They are from different recording sessions spanning the same night.)*

---

## 2. Zip Structure and Modality Availability

A scan of the archive's namelist confirms all expected modality directories are present. 

| Modality | File Count |
|---|---|
| `Navtech_Cartesian/` (PNGs) | 710 |
| `Navtech_Polar/` (PNGs) | 710 |
| `velo_lidar/` (CSVs) | 1,799 |
| `zed_left/` (PNGs) | 2,647 |
| `zed_right/` (PNGs) | 2,645 |
| `GPS_IMU_Twist/` (TXTs) | 7,201 |
| **Total ZIP Entries** | **15,719** |

No structural anomalies, missing directories, or unusual file types were found. `zed_left` and `zed_right` have a minor frame mismatch (2 frames difference), which is common in RADIATE and handled during synchronization matching.

---

## 3. Annotation Quality (The Critical Issue)

The `annotations/annotations.json` file was parsed from the ZIP.

| Metric | fog_8_0 Value |
|---|---|
| Number of tracked objects | 9 |
| Class distribution | van: 1, car: 8 |
| Bbox list length per object | 710 (Matches radar frames) |
| Total valid bounding boxes | **276** |
| Frames with ≥1 valid annotation | **205** |
| **Annotation Coverage** | **28.87%** (205 / 710 radar frames) |

---

## 4. Comparison: fog_8_0 vs. fog_6_0

| Metric | `fog_8_0` (Candidate Train) | `fog_6_0` (Held-out Test) |
|---|---|---|
| `meta.json["set"]` | `train_good_and_bad_weather` | `test` |
| Radar frames | 710 | 714 |
| Tracked objects | 9 | 39 |
| Valid frames | 205 | 657 |
| **Coverage %** | **28.87%** | **92.4%** |
| Classes | car, van | car, van, bus |

### Verdict on Annotation Quality

While `fog_8_0` has the correct `meta.json` split flag and complete multimodal sensor data, **the annotation density is extremely poor.**
- 71.1% of the radar frames contain zero annotated objects.
- There are only 276 valid bounding boxes across the entire sequence.
- Training an object detection network on this sequence would result in severe false negative penalization during training, as the model would frequently predict objects in the 505 unannotated frames that are actually present but simply unlabelled.

---

## 5. Final Status and Recommendation

**STATUS:** **NOT SUITABLE FOR TRAINING**

Despite having the correct metadata (`"set": "train_good_and_bad_weather"`), the annotation coverage (28.87%) is too sparse for viable supervised learning. Training a model on `fog_8_0` would severely harm prediction quality and make uncertainty calibration stress-testing meaningless.

**Next Recommended Action:**
Do not extract `fog_8_0.zip`. The existing `fog_6_0` test sequence is fully annotated and usable. The prototype must rely on proper, fully annotated training sequences like `fog_1_0` and `fog_2_0`, as originally recommended in the Phase 2B plan. Contact the dataset provider for access to the `fog_1` and `fog_2` sequences, or fall back to good-weather training data with simulated degradation if acquiring additional fog sequences is impossible.
