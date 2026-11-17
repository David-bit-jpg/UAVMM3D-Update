# A3: block-bootstrap intervals on AntiUAV3D v2 test, and height-ratio alternatives without GT or LiDAR on MAV6D (2026-11-13)

CPU only (`CUDA_VISIBLE_DEVICES=""`). Uses existing predictions only. No training. No file outside `review1113/A3/` was written. Test labels are read only by the scorers. Nothing was tuned on these results.

## A. Moving-block bootstrap, AntiUAV3D v2 test (antifs_test, 2,624 frames)

**Scorer.** Per-frame quantities come from the pack scorer's own functions (`D:/UAV3D_baselines_pack/scripts/score.py`: `box2d`, `iou`, `sym`, rules `outdoor_test`). These are TP at 2D IoU ≥ 0.1, relative centre error, symmetric AOE and ASE. AP (the nuScenes-style 101-point AP, averaged over the 2.5/5/10/20 % thresholds) and UDS are **recomputed from scratch on every resample**, because AP ranks the whole resampled set. The vectorised AP/UDS reproduces `score.metrics` to 1e-9:
- on the full set, for all three arms. Point values: PST 41.735, Sim-only 30.552, RGB-only 38.922, equal to the stored `.score_antifs_test.json`.
- on a resample that contains duplicated frames.

**Resampling.** This is a circular moving-block bootstrap within each sequence. Strata are kept: `train_seq_0010` (M4E, 2,576 frames) and `train_seq_0022` (M400, 48 frames). Each stratum keeps its own length. Frames are put in time order by numeric frame id. B = 2,000, seed 0. In each pair, both arms use the same resampled frames.

**Block length.** We chose L = 176 test frames, from the autocorrelation of per-frame errors in the long sequence:

| Arm | ACF of distance error (lags 1 / 10 / 50) | First lag < 0.1 | ACF of AOE (lags 1 / 10 / 50) | First lag < 0.1 |
|---|---|---|---|---|
| PST | 0.44 / 0.32 / 0.03 | 29 | 0.85 / 0.55 / 0.24 | **176** |
| Sim-only | 0.60 / 0.42 / 0.25 | 130 | 0.93 / 0.36 / 0.07 | 46 |
| RGB-only teachers | 0.45 / 0.33 / 0.06 | 38 | 0.74 / 0.49 / 0.20 | 60 |

- L is the largest first-crossing of ACF < 0.1 over the 3 arms × 2 error series. That conservative choice is set by PST's slowly decorrelating orientation error.
- The cube-root rule (n^(1/3) ≈ 14) would be far too short here, since the ACF at lag 14 is still 0.3–0.5.
- Test frames are mostly every 3rd raw frame, with about 10 large time gaps. So L = 176 test frames is about 530 raw frames.
- The M400 stratum (48 frames) is shorter than L. It is therefore always resampled as one circularly shifted block, contributes no resampling variance, and only keeps its weight.

### A1. 95 % percentile intervals (L = 176, B = 2,000)
| Arm | AP | 95 % CI | UDS | 95 % CI |
|---|---|---|---|---|
| **PST** | **27.89** | [24.08, 32.11] | **41.73** | [38.76, 44.93] |
| Sim-only detector | 16.04 | [10.19, 22.27] | 30.55 | [25.33, 35.52] |
| RGB-only teachers | 23.39 | [19.46, 28.09] | 38.92 | [35.69, 42.42] |
| **PST − Sim-only** (paired) | **+11.85** | [+6.16, +17.42] | **+11.18** | [+7.08, +15.44] |
| **PST − RGB-only teachers** (paired) | **+4.50** | [+2.53, +6.23] | **+2.81** | [+1.53, +3.96] |

### A2. Sensitivity to block length (paired differences)
| L (test frames) | PST − Sim-only AP | PST − Sim-only UDS | PST − RGB-only AP | PST − RGB-only UDS |
|---|---|---|---|---|
| 50 | [+6.91, +16.43] | [+7.68, +14.38] | [+2.25, +6.35] | [+1.35, +3.99] |
| 100 | [+6.55, +16.91] | [+7.31, +14.85] | [+2.14, +6.40] | [+1.33, +4.07] |
| **176 (main)** | [+6.16, +17.42] | [+7.08, +15.44] | [+2.53, +6.23] | [+1.53, +3.96] |

- Fraction of resamples with a difference ≤ 0: 0.000 at every L for both pairs, except PST − RGB-only at L = 100 (0.001).
- Per-arm intervals widen slightly with L. For example, PST UDS is [38.95, 44.48] at L = 50 and [38.81, 44.66] at L = 100. The paired intervals barely move.

**Caveat (to state in the paper).** These intervals capture only within-sequence temporal variation of one M4E flight (plus a fixed 48-frame M400 stratum). They say nothing about variation across scenes, flights or airframes. With 2 test sequences, that variation cannot be estimated from this test set.

## B. Height-ratio alternatives without GT or LiDAR, MAV6D

### B1. Sim-derived aspects h / ((l+w)/2)
Source: the indoor sim train cache `E:/mmcache/full_v2/train` (19,490 frames, 42,020 airframe boxes, 7 models; `boxes9d` l, w, h are constant per model).

| Sim model | l × w × h (m) | Aspect h/((l+w)/2) | Crop brightness, median grey (airframe mask) | Share of crops nearer the bright centre |
|---|---|---|---|---|
| DJI-avata2 | 0.178 × 0.180 × 0.080 | 0.447 | 109.0 | 0.56 |
| DJI-mavic-mini | 0.246 × 0.289 × 0.055 | 0.206 | 134.1 | 0.79 |
| DJI-phantom4 | 0.348 × 0.348 × 0.199 | 0.572 | 139.2 | 0.84 |
| drone-unk3 (K3 Mini) | 0.307 × 0.307 × 0.102 | 0.332 | 143.5 | 0.77 |
| m210-rtk | 0.883 × 0.821 × 0.413 | 0.485 | 103.3 | 0.49 |
| matrix-300-RTK | 1.103 × 1.027 × 0.413 | 0.388 | 86.4 | 0.23 |
| Matrice-600-Pro | 1.839 × 1.711 × 0.619 | 0.349 | 87.5 | 0.38 |

- **(a) Mean aspect over all sim boxes.** Per object: **0.3999**. Per model: **0.3968**.
- **(b) Per brightness class, from sim crops.** For each of up to 400 random objects per model (2,756 crops), lum is the mean grey (cv2 BGR→GREY) over the airframe pixels. This is the statistic `color_feats.color_stats` uses on real crops.
  - The class is the nearest of the chain's real colour centres col_c = (62.6, 144.9). These come from k-means-2 on unlabeled real train lum and equal the deployed `consts.col_c`.
  - Variant: k-means-2 on the sim lum itself, giving centres (84.0, 140.8).
  - **Airframe mask in sim.** The real chain uses the second-stage network's silhouette, which would need GPU inference. Instead we use the sim render itself: sim depth pixels inside the projected 3D box, within the box radius of the box depth. A variant uses the projected 3D-box hull. Both use only sim data and give the same model-to-class assignment.

| (b) variant | r dark | r bright | Model-to-class assignment (majority) |
|---|---|---|---|
| per object, real centres | 0.3987 | 0.3967 | (crop-level) |
| per model, real centres | 0.4071 | 0.3892 | dark: m210, M300, M600; bright: avata2, mavic-mini, phantom4, K3 Mini |
| per model, sim k-means centres | 0.4170 | 0.3699 | as above + avata2 → dark |

- **Is knowledge of the real airframe models needed?** No. Classes come only from image brightness of sim crops and the label-free real centres.
- **Result of (b).** Brightness in sim does **not** separate thin from tall airframes. Both classes average about 0.40, and the per-model split even reverses the order (dark ≥ bright). The LiDAR r_k (dark 0.382, bright 0.710) learned the opposite on real data.
- **Note.** The s̄ constant aspect 0.5717 is exactly the sim Phantom 4 box aspect: s̄ = (0.350, 0.349, 0.200) ≈ sim Phantom 4 dimensions.

### B2. Swap test on the reproduced deployed MAV6D run
- **Method.** Same as `height/height_eval.py` variant (a). Only the height source changes: h := r_class · (l+w)/2. Each frame keeps the brightness class the deployed chain gave it.
- **Scorer.** `cmpeval/paper_table.metrics` (= pack `real_test`).
- **Reproduction.** The deployed run reproduces as test UDS 70.16 and val 62.49.
- **r_k values.** The r_k actually in the deployed predictions is 0.3824 / 0.7099 (`consts.ht_col`). HEIGHT_REPORT.md rounds these to 0.380 / 0.711.
- |Δh| is the mean over the scorer's TP set (top box, 2D IoU ≥ 0.1).
- AP is identical in all rows (56.44 test, 48.33 val), because position and confidence are unchanged.

**Test (10 sequences)**
| Height source | r dark / bright | UDS | ASE (cm) | mean \|Δh\| cm (% GT h) | Mavic 2 \|Δh\| cm (%) | Mavic 2 ASE | Phantom 4 \|Δh\| cm (%) | Phantom 4 ASE |
|---|---|---|---|---|---|---|---|---|
| **PST: LiDAR height ratio r_k** | 0.382 / 0.710 | **70.16** | **1.81** | **1.77 (9.3)** | **1.49 (12.4)** | **3.39** | **1.87 (8.1)** | **1.23** |
| s̄ constant aspect (IR-/RGB-only rule) | 0.572 / 0.572 | 69.13 | 2.68 | 4.41 (27.8) | 8.10 (67.5) | 5.59 | 3.06 (13.3) | 1.62 |
| (a) sim mean aspect, per object | 0.400 | 67.91 | 3.61 | 7.17 (33.4) | 2.07 (17.2) | 3.58 | 9.05 (39.4) | 3.62 |
| (a) sim mean aspect, per model | 0.397 | 67.89 | 3.63 | 7.22 (33.5) | 1.96 (16.3) | 3.55 | 9.16 (39.8) | 3.66 |
| (b) sim brightness class, per object | 0.399 / 0.397 | 67.88 | 3.63 | 7.24 (33.7) | 2.02 (16.8) | 3.57 | 9.16 (39.8) | 3.66 |
| (b) sim brightness class, per model | 0.407 / 0.389 | 67.77 | 3.72 | 7.51 (35.1) | 2.32 (19.3) | 3.67 | 9.42 (40.9) | 3.74 |
| (b) per model, sim k-means centres | 0.417 / 0.370 | 67.55 | 3.91 | 8.08 (38.0) | 2.66 (22.2) | 3.78 | 10.07 (43.8) | 3.96 |

**Val (16 sequences)**
| Height source | r dark / bright | UDS | ASE (cm) | mean \|Δh\| cm (% GT h) | Mavic 2 \|Δh\| cm (%) | Mavic 2 ASE | Phantom 4 \|Δh\| cm (%) | Phantom 4 ASE |
|---|---|---|---|---|---|---|---|---|
| **PST: LiDAR r_k** | 0.382 / 0.710 | **62.49** | **2.63** | **1.99 (12.2)** | **1.66 (13.8)** | **3.50** | **2.38 (10.4)** | **1.61** |
| s̄ constant aspect | 0.572 | 60.93 | 3.87 | 5.73 (42.4) | 8.15 (67.9) | 5.67 | 2.89 (12.6) | 1.78 |
| (a) sim mean, per object | 0.400 | 61.13 | 3.72 | 5.25 (27.3) | 2.10 (17.5) | 3.65 | 8.93 (38.8) | 3.80 |
| (a) sim mean, per model | 0.397 | 61.13 | 3.71 | 5.24 (27.1) | 1.99 (16.6) | 3.61 | 9.04 (39.3) | 3.83 |
| (b) sim class, per object | 0.399 / 0.397 | 61.12 | 3.72 | 5.27 (27.3) | 2.05 (17.1) | 3.63 | 9.04 (39.3) | 3.83 |
| (b) sim class, per model | 0.407 / 0.389 | 61.01 | 3.81 | 5.54 (29.1) | 2.34 (19.5) | 3.73 | 9.28 (40.3) | 3.91 |
| (b) per model, sim k-means | 0.417 / 0.370 | 60.81 | 3.97 | 6.00 (31.9) | 2.68 (22.3) | 3.84 | 9.90 (43.0) | 4.12 |

### B3. Differences vs LiDAR r_k (alternative − r_k), sequence bootstrap 95 % CI
Sequences are resampled with replacement within each group, B = 2,000, seed 0. Both arms share the resampled sequences. For the per-aircraft rows, ΔUDS is UDS computed on that aircraft's frames only.

| Split | Height source | Group (seqs) | ΔUDS | ΔASE (cm) | Δ mean \|Δh\| (cm) | Δ mean \|Δh\| (% GT h) |
|---|---|---|---|---|---|---|
| test | s̄ constant | all (10) | −1.03 [−1.70, −0.53] | +0.87 [+0.43, +1.46] | +2.64 [+1.29, +4.40] | +18.5 [+5.8, +34.8] |
| test | s̄ constant | Mavic 2 (3) | −2.52 [−2.78, −1.89] | +2.20 [+2.16, +2.22] | +6.61 [+6.58, +6.67] | +55.1 [+54.8, +55.6] |
| test | s̄ constant | Phantom 4 (7) | −0.50 [−0.59, −0.39] | +0.40 [+0.31, +0.47] | +1.19 [+0.94, +1.40] | +5.2 [+4.1, +6.1] |
| test | (a) sim mean, per object | all (10) | −2.25 [−2.94, −1.37] | +1.80 [+1.10, +2.35] | +5.40 [+3.30, +7.05] | +24.1 [+15.7, +30.7] |
| test | (a) sim mean, per object | Mavic 2 (3) | −0.24 [−0.26, −0.23] | +0.19 [+0.18, +0.21] | +0.58 [+0.55, +0.62] | +4.8 [+4.6, +5.1] |
| test | (a) sim mean, per object | Phantom 4 (7) | −2.99 [−3.08, −2.89] | +2.39 [+2.31, +2.46] | +7.18 [+6.94, +7.38] | +31.2 [+30.2, +32.1] |
| test | (a) sim mean, per model | all (10) | −2.27 [−2.98, −1.37] | +1.82 [+1.09, +2.39] | +5.45 [+3.28, +7.16] | +24.2 [+15.4, +31.1] |
| test | (b) sim class, per object | all (10) | −2.28 [−2.98, −1.38] | +1.82 [+1.11, +2.39] | +5.47 [+3.32, +7.16] | +24.4 [+15.7, +31.1] |
| test | (b) sim class, per model | all (10) | −2.39 [−3.09, −1.49] | +1.91 [+1.20, +2.47] | +5.74 [+3.60, +7.41] | +25.8 [+17.6, +32.2] |
| test | (b) sim class, per model | Mavic 2 (3) | −0.34 [−0.36, −0.32] | +0.28 [+0.26, +0.29] | +0.83 [+0.80, +0.87] | +6.9 [+6.6, +7.2] |
| test | (b) sim class, per model | Phantom 4 (7) | −3.15 [−3.23, −3.04] | +2.52 [+2.43, +2.59] | +7.55 [+7.29, +7.76] | +32.8 [+31.7, +33.7] |
| val | s̄ constant | all (16) | −1.56 [−2.20, −0.71] | +1.24 [+0.57, +1.76] | +3.73 [+1.71, +5.27] | +30.2 [+12.8, +43.7] |
| val | s̄ constant | Mavic 2 (7) | −2.71 [−2.78, −2.55] | +2.16 [+2.04, +2.22] | +6.49 [+6.13, +6.67] | +54.1 [+51.0, +55.6] |
| val | s̄ constant | Phantom 4 (9) | −0.21 [−0.46, +0.27] | +0.17 [−0.21, +0.37] | +0.50 [−0.64, +1.11] | +2.2 [−2.8, +4.8] |
| val | (a) sim mean, per object | all (16) | −1.36 [−2.23, −0.66] | +1.09 [+0.53, +1.78] | +3.26 [+1.59, +5.35] | +15.1 [+8.1, +23.9] |
| val | (a) sim mean, per object | Mavic 2 (7) | −0.18 [−0.26, −0.01] | +0.15 [+0.01, +0.21] | +0.44 [+0.02, +0.62] | +3.7 [+0.1, +5.1] |
| val | (a) sim mean, per object | Phantom 4 (9) | −2.73 [−2.97, −2.26] | +2.18 [+1.81, +2.38] | +6.55 [+5.43, +7.13] | +28.5 [+23.6, +31.0] |
| val | (a) sim mean, per model | all (16) | −1.35 [−2.26, −0.64] | +1.08 [+0.51, +1.81] | +3.25 [+1.53, +5.42] | +14.8 [+7.5, +24.0] |
| val | (b) sim class, per object | all (16) | −1.37 [−2.26, −0.66] | +1.09 [+0.53, +1.81] | +3.28 [+1.58, +5.43] | +15.1 [+7.9, +24.2] |
| val | (b) sim class, per model | all (16) | −1.48 [−2.37, −0.77] | +1.18 [+0.62, +1.90] | +3.54 [+1.86, +5.70] | +16.9 [+9.8, +25.6] |
| val | (b) sim class, per model | Mavic 2 (7) | −0.28 [−0.36, −0.10] | +0.23 [+0.08, +0.29] | +0.68 [+0.24, +0.87] | +5.7 [+2.0, +7.2] |
| val | (b) sim class, per model | Phantom 4 (9) | −2.87 [−3.13, −2.37] | +2.30 [+1.90, +2.50] | +6.89 [+5.70, +7.50] | +30.0 [+24.8, +32.6] |

The rows not shown are in `b_swap.json`: per-aircraft rows for (a)-per-model, (b)-per-object and the sim k-means variant. They track the shown rows within 0.1–0.6 cm. The Mavic 2 test intervals rest on only 3 sequence clusters, so they are optimistic.

### B4. Reading
- **Every alternative is worse than LiDAR r_k overall, on both splits.** All overall intervals exclude 0.
  - Test ΔUDS ranges from −1.0 (s̄) to −2.6. Test Δ mean |Δh| ranges from +2.6 to +6.3 cm, against r_k's 1.77 cm.
- **One constant cannot fit two airframes.**
  - The sim-mean constant (≈ 0.40) is close to the real Mavic 2 aspect (0.389). So it nearly matches r_k on Mavic 2 (+0.5 cm), but it under-estimates Phantom 4 height by about 40 %, which is +7 cm.
  - The s̄ constant (0.572) does the reverse: it is close on Phantom 4 (+1.2 cm) and off by +6.6 cm on Mavic 2.
- **Sim brightness classes give two near-identical constants (≈ 0.40 / 0.40).** Sim brightness does not predict airframe aspect. Thin bright minis and tall bright Phantom 4 sit in the same class, and the large dark M300/M600 are thin. So "two brightness-class constants" built without LiDAR collapse to a single constant and inherit its failure.
- **What makes r_k work.** Its per-class values (0.382 / 0.710) match the real per-class aspects. Those values come from the LiDAR-taught height network on unlabeled real frames, not from the class split itself.

## Files (`gproto/review1113/A3/`)
- `a_block_boot.py` → `a_block_boot.json`, `a_block_boot.log`: part A (point checks, ACF, bootstrap at L = 50/100/176).
- `b_sim_aspect.py` → `b_sim_aspect.json`, `b_sim_aspect.log`: sim aspects and sim crop brightness.
- `b_swap.py` → `b_swap.json`, `b_swap.log`: swap test plus sequence bootstrap. It imports `height/height_eval.py` read-only.
- `make_tables.py`: prints the tables above from the JSONs.
- Run from bash: `PYTHONUTF8=1 CUDA_VISIBLE_DEVICES= D:/Miniconda3/envs/city/python.exe <script>`. Each script takes 1–5 min on CPU.
