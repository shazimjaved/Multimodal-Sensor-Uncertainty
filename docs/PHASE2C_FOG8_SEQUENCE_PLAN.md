# PHASE 2C — fog_8 Sequence Verification and Training Data Plan

**Date:** 2026-09-24
**Constraint:** Do not download. Do not alter fog_6_0.

---

## 1. Inspection Method and Hard Limits

### What was inspected

| Archive | Location | Inspectable without extraction? |
|---|---|---|
| `fog_6_0.zip` | `C:\Users\GOGI LAPTOP\Desktop\Research\fog_6_0.zip` (1.39 GB) | ✅ Yes — read directly from zip |
| `fog_8_0.zip` | **NOT ON DISK** — Dropbox only | ❌ No |
| `fog_8_1.zip` | **NOT ON DISK** — Dropbox only | ❌ No |
| `fog_8_2.zip` | **NOT ON DISK** — Dropbox only | ❌ No |

> [!IMPORTANT]
> The fog_8_0, fog_8_1, and fog_8_2 zip files are not on disk. They exist in the Dropbox folder but have not been downloaded. **No metadata, annotations, or file counts can be read from them without downloading and extracting.** This report documents what is confirmed, what is inferred from RADIATE conventions, and what remains unknown.

### Inspection of fog_6_0.zip (direct Python zipfile read, no extraction needed)

`meta.json` was read directly from the zip without extracting any files:

```json
{
  "name": "fog_6_0",
  "type": "fog",
  "set": "test",
  "version": "1.0",
  "date_created": "Sat Feb 29 03:09:49 2020"
}
```

Internal structure confirmed:

| Entry | Files in zip |
|---|---|
| `GPS_IMU_Twist/` | 7,200 `.txt` files |
| `zed_left/` | 2,659 `.png` files |
| `zed_right/` | 2,659 `.png` files |
| `velo_lidar/` | 1,799 `.csv` files |
| `Navtech_Cartesian/` | 714 `.png` files |
| `Navtech_Polar/` | 714 `.png` files |
| `annotations/annotations.json` | 1 file (290.8 KB uncompressed) |
| `meta.json` | 1 file (root level) |
| **Total** | **15,753 entries** |

`annotations.json` read directly from zip:
- 39 tracked objects
- Classes: `{bus: 1, car: 31, van: 7}`
- bbox list length per object: **714** (= number of radar frames)

---

## 2. Confirmed vs. Inferred vs. Unknown per Sequence

### fog_6_0 — Fully Confirmed

| Property | Value | Source |
|---|---|---|
| `meta.json["set"]` | `"test"` | Direct zip read |
| `meta.json["type"]` | `"fog"` | Direct zip read |
| Radar frames | 714 | Phase 2 audit |
| Annotated frames | 660/714 (92.4%) | Phase 2 audit |
| Usable multimodal frames | 657 | Phase 2 audit |
| Classes | car 85%, van 12%, bus 4% | Phase 2 audit |
| LiDAR | 1,799 frames, fog-attenuated | Phase 2 audit |
| Camera | 2,659 frames, hazy | Phase 2 audit |
| Duration | 178 s (~3 min) | Phase 2 audit |
| Calibration | Verified (Phase 1C + Phase 2) | Smoke test |

### fog_8_0, fog_8_1, fog_8_2 — Properties Table

| Property | fog_8_0 | fog_8_1 | fog_8_2 |
|---|---|---|---|
| `meta.json["set"]` | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** |
| `meta.json["type"]` | fog (inferred) | fog (inferred) | fog (inferred) |
| Radar frames | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** |
| Annotated frames | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** |
| LiDAR availability | probable (inferred) | probable (inferred) | probable (inferred) |
| Camera availability | probable (inferred) | probable (inferred) | probable (inferred) |
| Calibration compatibility | likely same rig | likely same rig | likely same rig |
| Duration | **UNKNOWN** | **UNKNOWN** | **UNKNOWN** |
| In original RADIATE paper | **NO** | **NO** | **NO** |

---

## 3. What Is Known from RADIATE Conventions

### RADIATE naming convention

RADIATE sequences follow the pattern `<weather>_<session>_<take>`:

| Part | Meaning |
|---|---|
| `fog` | Weather/scenario type |
| `6` or `8` | **Recording session number** (different days/drives) |
| `0`, `1`, `2` | **Take number within a session** |

- `fog_6_0`: session 6, single take — one continuous ~3-minute drive
- `fog_8_0`: session 8, take 0 — first drive of session 8
- `fog_8_1`: session 8, take 1 — second drive of session 8 (same day as fog_8_0)
- `fog_8_2`: session 8, take 2 — third drive of session 8 (same day as fog_8_0 and fog_8_1)

### What the session number implies

> [!WARNING]
> fog_8_0, fog_8_1, and fog_8_2 are three drives from **the same recording session (session 8)**. They were captured on the same day, in the same location, under the same fog conditions. This means:
>
> 1. **They are correlated** — same ego vehicle, same route, same fog density, same parked/moving actors in the scene
> 2. **They are not independent** — using fog_8_0 for train and fog_8_1 for val creates a validation set that may share parked vehicles, road segments, and similar annotations
> 3. **They are not in the original paper** — fog_1_0 through fog_6_0 are the 6 fog sequences listed in the RADIATE paper (arXiv:2010.09076). fog_8_* sequences were added to the Dropbox later

### What the `meta.json["set"]` value might be

Based on the RADIATE SDK pattern:
- The original 22 sequences all have `"set": "train_good_weather"`, `"train_bad_weather"`, or `"test"`
- fog_8_* are additional sequences — they may use the same conventions or may have a different/null `"set"` value
- **This cannot be determined without downloading and reading their `meta.json`**
- If their `"set"` is `"test"`, using them for training would violate the official split

---

## 4. Split Option Analysis

### Option A: fog_8_0 → TRAIN | fog_8_1 → VAL | fog_6_0 → TEST

| Criterion | Assessment |
|---|---|
| meta.json set compatibility | **UNKNOWN** — must be verified before use |
| Train/val independence | **WEAK** — fog_8_0 and fog_8_1 are same session, same day, same location |
| Train/test independence | ✅ GOOD — fog_6_0 is session 6, fog_8_0 is session 8 (different recording day) |
| Annotation availability | **UNKNOWN** for fog_8_0/fog_8_1 |
| Fog condition match | ✅ LIKELY — all three are fog_suburban |
| Calibration match | ✅ LIKELY — same rig |
| Minimum download needed | ~4.2 GB (fog_8_0 + fog_8_1) |
| **Verdict** | **CONDITIONALLY ACCEPTABLE** — if meta.json set permits and annotation count is sufficient |

**Key weakness:** Val set (fog_8_1) is same-session as train (fog_8_0). Validation loss metrics may be optimistic due to scene overlap. For a prototype, this is acceptable; for publication, it is a significant caveat.

---

### Option B: fog_8_0 + fog_8_1 → TRAIN | fog_8_2 → VAL | fog_6_0 → TEST

| Criterion | Assessment |
|---|---|
| meta.json set compatibility | **UNKNOWN** — all three must be verified |
| Train/val independence | **WEAK** — fog_8_2 is same session as fog_8_0/fog_8_1 |
| Train/test independence | ✅ GOOD — fog_6_0 session 6 vs. session 8 |
| Total training frames | ~1,400 annotated radar frames (if both ~700 each) |
| Annotation availability | **UNKNOWN** |
| Minimum download needed | ~6.3 GB (all three fog_8) |
| **Verdict** | **CONDITIONALLY ACCEPTABLE** — larger training set but same-session correlation persists across all splits |

**Key strength over Option A:** More training data. **Key weakness:** All three sequences are same-session, so train/val/test only achieves true independence at the test level (fog_6_0 vs. fog_8).

---

### Option C = Option B (identical)

Options B and C in the task description are the same split. The analysis above applies.

---

## 5. Pre-Extraction Verification Requirements

Before extracting any fog_8 zip and using it in the pipeline, the following must be confirmed:

### Mandatory checks (from zip, before full extraction)

```python
import zipfile, json

def quick_inspect(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        # 1. Read meta.json (instant, no extraction)
        with z.open('meta.json') as f:
            meta = json.load(f)
        print('set:', meta.get('set'))       # MUST NOT be 'test'
        print('type:', meta.get('type'))     # Should be 'fog'
        print('version:', meta.get('version'))
        
        # 2. Count radar frames
        radar = [n for n in z.namelist() if n.startswith('Navtech_Cartesian/') and n.endswith('.png')]
        print('radar frames:', len(radar))   # Should be ~700
        
        # 3. Read annotations (instant, ~30KB compressed)
        with z.open('annotations/annotations.json') as f:
            anns = json.load(f)
        classes = {}
        for obj in anns:
            c = obj['class_name']
            classes[c] = classes.get(c, 0) + 1
        valid = sum(1 for obj in anns 
                    for b in obj['bboxes'] if isinstance(b, dict) and b.get('position'))
        print('tracked objects:', len(anns))
        print('class distribution:', classes)
        print('total valid bboxes:', valid)
```

### Pass criteria before any training use

| Check | Required value |
|---|---|
| `meta.json["set"]` | `"train_bad_weather"` (NOT `"test"`) |
| Radar frame count | ≥ 400 frames |
| Annotation coverage | ≥ 70% of radar frames annotated |
| Total valid bboxes | ≥ 500 bboxes total |
| All modality dirs present | `Navtech_Cartesian/`, `velo_lidar/`, `zed_left/`, `zed_right/` |

---

## 6. Reasons fog_8_* SHOULD be Used (If Verified)

1. **Same weather type as test set** — fog_8 is fog_suburban, matching fog_6_0 exactly
2. **Same physical rig** — calibration parameters should be identical; no code changes required
3. **Same file format** — identical directory structure, same timestamp format, same annotation schema
4. **Practical availability** — these are the only fog training sequences in the Dropbox; no alternative available without re-contacting the RADIATE team
5. **Fog_6_0 remains genuinely unseen** — session 6 vs. session 8 are distinct recording events

## 7. Reasons fog_8_* MIGHT NOT be Used

| Risk | Severity | Mitigation |
|---|---|---|
| `meta.json["set"]` = `"test"` | 🔴 CRITICAL — would invalidate split | Read meta.json from zip before extracting |
| All three are same-session | 🟡 MODERATE — train/val correlation | Document as prototype limitation; acceptable for internal experiments |
| Not in original RADIATE paper | 🟡 MODERATE — unknown peer-review status | Cite as "additional fog sequences" not from the original benchmark |
| Annotation quality unknown | 🟡 MODERATE — may have sparse labels | Run annotation coverage check before use |
| Shorter sequences possible | 🟢 LOW — sub-takes may be shorter than fog_6_0 | Check radar frame count from zip before extraction |

---

## 8. Final Recommendation

```
╔════════════════════════════════════════════════════════════════╗
║   STATUS: CONDITIONALLY VIABLE — pending verification         ║
╠════════════════════════════════════════════════════════════════╣
║  RECOMMENDED SPLIT (if verification passes):                  ║
║    TRAIN      → fog_8_0                                       ║
║    VALIDATION → fog_8_1                                       ║
║    TEST       → fog_6_0  (held out, fully audited)           ║
╠════════════════════════════════════════════════════════════════╣
║  MINIMUM DOWNLOAD REQUIRED:                                   ║
║    fog_8_0.zip  (~2.1 GB estimated)                          ║
║    fog_8_1.zip  (~2.1 GB estimated)                          ║
║    Total: ~4.2 GB additional                                  ║
╠════════════════════════════════════════════════════════════════╣
║  BEFORE EXTRACTION: run quick_inspect() on each zip to:      ║
║    1. Confirm meta.json["set"] != "test"                      ║
║    2. Confirm radar frame count >= 400                        ║
║    3. Confirm annotation coverage >= 70%                      ║
╠════════════════════════════════════════════════════════════════╣
║  KNOWN LIMITATION:                                            ║
║    fog_8_0, fog_8_1, fog_8_2 are from the SAME session (8)  ║
║    → train/val split is correlated, not independent           ║
║    → acceptable for prototype; disclose in any publication    ║
╠════════════════════════════════════════════════════════════════╣
║  fog_6_0 TEST SET STATUS: VALID — session 6 vs. session 8   ║
║    are distinct recording events; no scene overlap expected   ║
╚════════════════════════════════════════════════════════════════╝
```

### If meta.json["set"] turns out to be "test" for any fog_8 sequence

Those sequences must be moved to the test pool and cannot be used for training. In that case, contact the RADIATE team directly (`pro.hw.ac.uk/radiate`) to request access to `fog_1_0` and `fog_2_0`, which are confirmed `train_bad_weather` sequences.

### Additional data required?

**Not necessarily.** fog_8_0 alone (~700 frames if similar to fog_6_0) is sufficient for a prototype model. fog_8_1 provides the validation set. fog_8_2 is optional — it adds training diversity if combined with fog_8_0 under Option B, but the same-session correlation makes this a modest gain.

---

## 9. Output Files

```
outputs/
└── fog8_audit.csv    ← sequence metadata (confirmed for fog_6_0, UNKNOWN for fog_8_*)

docs/
└── PHASE2C_FOG8_SEQUENCE_PLAN.md   ← this document
```
