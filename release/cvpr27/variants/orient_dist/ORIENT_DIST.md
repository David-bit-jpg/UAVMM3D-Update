# Orientation three ways (C1) and AntiUAV3D by distance band (C2), plus the box-hull refiner script (11-13; CPU only; existing predictions only)

Files in `review1113/orient_dist/`:
- `orient_dist.py` produces `orient_dist.json` and `orient_dist.log` (C1 and C2).
- `heading_modes.py` produces `heading_modes.json`.
- `lidar_share.py` produces `lidar_share.json`, `lidar_share_rows.pkl` and `lidar_share.log` (C2b).
- `make_tables.py` prints the tables below.

No training was run, and no GPU was used (`CUDA_VISIBLE_DEVICES=""`).

**Runs scored.** These are the reported clean runs:
- **PST:** `srcx/in2mav`, `srcx/out2antifs` (`pcmd_fused_test.pkl`)
- **RGB-only teachers:** `srcx/in2mav_none`, `srcx/out2antifs_none`
- **Sim-only:** `srconly_test.pkl` of the PST run

**Checks.** For every arm, AP, UDS, ATE and AOE-90 reproduce the pack scorer and the stored score JSON exactly (asserted). No TP frame has non-finite Euler angles.

## Metric definitions (code terms)
Every metric uses the pack scorer's loading and TP rule. Per frame, the top-scoring box is taken. A frame is a TP if the IoU between the projected 2-D boxes of the prediction and the GT is at least 0.1. Each arm's orientation errors are averaged over **its own TP frames**, exactly as the pack computes AOE.

The definitions use:
- `Rp = Rot.from_euler('xyz', pred_eul).as_matrix()` and `Rg = Rot.from_euler('xyz', gt_eul).as_matrix()`
- `RZ[q] = Rz(90°·q)`
- `geo(A, B) = degrees(arccos(clip((trace(Aᵀ B) − 1) / 2, −1, 1)))`

| Name | Definition | Range | Reading |
|---|---|---|---|
| **AOE-90°** (the reported AOE) | `min_{q∈{0,1,2,3}} geo(Rp·RZ[q], Rg)` (= pack `score.sym`) | 0–~93 (chance: Haar ≈ 93.7, random yaw = 22.5) | 4-fold symmetric: any 90° turn about body z is free |
| **AOE-180°** | `min_{q∈{0,2}} geo(Rp·RZ[q], Rg)` | 0–180 (random yaw: 45) | 2-fold: only a 180° turn about body z is free |
| **Heading error (unfolded)** | `abs(degrees(arctan2(E[1,0], E[0,0])))`, `E = Rgᵀ·Rp` | 0–180 (random yaw: mean = median = 90) | Yaw of the relative rotation in the GT body frame (Z-Y-X). No folding, so front/back counts |

**Confidence intervals.** Intervals are for PST − RGB-only, paired (the same resample for both arms), with B = 2000 and `default_rng(0)`. The 95 % percentile interval is reported.
- **MAV6D:** whole-sequence bootstrap. "all" resamples the 10 test sequences. Per aircraft, it resamples that aircraft's sequences only (Mavic 2: 3, Phantom 4: 7, so the intervals are coarse).
- **AntiUAV3D:** circular moving-block bootstrap within each sequence, with sequence strata kept and frames in numeric order. L = 176 (= A3 `L_main`).
- **Matrice 400:** 48 frames are fewer than one block, so every resample is a rotation of the same frames. Its interval is degenerate and **not reported** (the cell shows the point value twice).

## C1. Orientation reported three ways
### C1-a. MAV6D test
| Aircraft | n | Arm | TP | AOE-90° mean | AOE-180° mean | Heading err. median | Heading err. mean | heading <45° / 45–135° / ≥135° (%) |
|---|---|---|---|---|---|---|---|---|
| all | 4800 | PST | 4616 | **14.48** | **16.91** | **12.41** | **60.31** | 65 / 6 / 29 |
|  |  | RGB-only | 4607 | 16.32 | 19.82 | 18.05 | 64.05 | 62 / 9 / 29 |
|  |  | Sim-only | 3349 | 20.32 | 54.92 | 94.52 | 87.03 | 22 / 59 / 18 |
| | | **PST − RGB-only** [95% CI] | | −1.84 [−3.14, −0.66] | −2.91 [−4.84, −1.25] | −5.64 [−26.57, +0.86] | −3.74 [−5.89, −1.80] | |
| Mavic 2 | 1290 | PST | 1116 | 21.50 | 25.61 | 13.99 | 30.56 | 81 / 13 / 6 |
|  |  | RGB-only | 1107 | 22.32 | 28.62 | 16.13 | 33.24 | 78 / 16 / 6 |
|  |  | Sim-only | 84 | 21.94 | 25.99 | 166.42 | 154.31 | 2 / 10 / 88 |
| | | **PST − RGB-only** [95% CI] | | −0.83 [−5.42, +0.68] | −3.01 [−7.94, −0.72] | −2.14 [−12.88, +0.73] | −2.68 [−8.63, −0.98] | |
| Phantom 4 | 3510 | PST | 3500 | 12.24 | 14.13 | 10.83 | 69.80 | 59 / 5 / 36 |
|  |  | RGB-only | 3500 | 14.43 | 17.04 | 19.57 | 73.80 | 56 / 7 / 37 |
|  |  | Sim-only | 3265 | 20.28 | 55.67 | 94.17 | 85.30 | 23 / 61 / 16 |
| | | **PST − RGB-only** [95% CI] | | −2.18 [−3.56, −1.03] | −2.91 [−5.36, −0.91] | −8.74 [−38.15, +3.41] | −4.00 [−6.76, −1.80] | |

### C1-b. AntiUAV3D v2 test
| Aircraft | n | Arm | TP | AOE-90° mean | AOE-180° mean | Heading err. median | Heading err. mean | heading <45° / 45–135° / ≥135° (%) |
|---|---|---|---|---|---|---|---|---|
| all | 2624 | PST | 2412 | 17.11 | 31.44 | 165.06 | 145.34 | 3 / 26 / 71 |
|  |  | RGB-only | 2386 | 16.45 | 31.07 | 166.72 | 147.05 | 3 / 26 / 72 |
|  |  | Sim-only | 2061 | 26.06 | 66.26 | 101.39 | 110.53 | 1 / 74 / 24 |
| | | **PST − RGB-only** [95% CI] | | +0.66 [+0.14, +1.15] | +0.37 [−0.26, +0.95] | −1.66 [−3.05, +0.65] | −1.71 [−3.12, −0.46] | |
| Matrice 4E | 2576 | PST | 2365 | 16.96 | 30.47 | 165.46 | 146.51 | 3 / 24 / 72 |
|  |  | RGB-only | 2339 | 16.24 | 30.09 | 167.09 | 148.27 | 3 / 24 / 73 |
|  |  | Sim-only | 2014 | 26.12 | 67.26 | 101.13 | 109.21 | 1 / 76 / 22 |
| | | **PST − RGB-only** [95% CI] | | +0.72 [+0.20, +1.22] | +0.37 [−0.29, +0.99] | −1.63 [−3.01, +0.51] | −1.76 [−3.14, −0.52] | |
| Matrice 400 | 48 | PST | 47 | 24.75 | 80.52 | 89.98 | 86.41 | 9 / 91 / 0 |
|  |  | RGB-only | 47 | 26.87 | 79.83 | 87.71 | 86.34 | 9 / 91 / 0 |
|  |  | Sim-only | 47 | 23.18 | 23.32 | 171.77 | 167.01 | 0 / 6 / 94 |
| | | PST − RGB-only (point; no CI, < 1 block) | | −2.13 | +0.69 | +2.27 | +0.07 | |

On the frames where both PST and RGB-only are TPs, the PST − RGB-only differences agree with the own-TP values to within 0.1° for AOE-90° and AOE-180°, and to within 0.35° for the heading mean (MAV6D 4,595 frames, AntiUAV3D 2,383; `common_tp` in the JSON).

**How to read C1**
1. **The unfolded heading error measures which of the four 90°-equivalent frames a network outputs.** It does not measure front/back skill. Every orientation network in the chain is trained and gated with the quadrotor-symmetric (90°-folded) loss and rule, so the output frame is an arbitrary choice. The heading-mode column shows this directly:
   - **MAV6D Phantom 4:** about 36 % of PST TPs sit near 180°.
   - **AntiUAV3D Matrice 4E:** 72 % sit at ≥ 135° for both PST and RGB-only, so the median is 165°. The networks pick the frame opposite to the AntiUAV3D GT's body-x convention.
   - **Sim-only:** most of its errors are 90° off (59 % on MAV6D, 76 % on AntiUAV3D).

   So the unfolded numbers (medians of 165° and means near 60–147°) reflect convention and folding, not accuracy. They should not be read as a quality measure. A paper sentence could say: "the airframes are 4-fold symmetric and our losses are folded accordingly, so heading is identifiable only up to 90°".
2. **The AOE-180° ranking matches AOE-90° on MAV6D.** PST − RGB-only is −2.9° [−4.8, −1.2], against −1.8° [−3.1, −0.7] at 90°. The interval excludes 0 for each aircraft at 180°, including Mavic 2: −3.0 [−7.9, −0.7], while its 90° interval spans 0. On MAV6D, PST is better under either symmetry convention.
3. **On AntiUAV3D, PST is slightly worse than RGB-only at AOE-90°:** +0.66° [+0.14, +1.15]. At AOE-180° there is no difference: +0.37 [−0.26, +0.95]. The PST gain on AntiUAV3D is therefore in position (C2), not in orientation. Both adapted arms beat sim-only by about 9° (AOE-90°).
4. **Matrice 400** is 48 frames from one sequence. PST and RGB-only are about 80° off at AOE-180° and about 90° in heading, so this airframe is consistently decoded in a 90°-rotated frame. Sim-only is the one arm close to a 180° flip. The slice is too small to rank arms (as in breakdown B2).

## C2. AntiUAV3D v2 test by GT range band
Bands use the GT range ‖t_gt‖, the same quantity the pack uses for its outdoor (rel = True) bins. Metrics are the pack's `metrics()` restricted to the band's frames, which is the scorer's per-band logic with these band edges:
- AP is computed with relative-distance thresholds.
- ATE is relative to the GT range (a = 0.25).
- UDS follows the pack formula.

The pack's own bins (0–25 / 25–50 / ≥50 m) are in the stored score JSONs.

CIs come from the block bootstrap with L = 176 over the whole sequences; each resample is then restricted to the band's frames.

| Band (m) | n (M4E/M400) | Arm | AP | ATE (rel.) | ASE (cm) | AOE-90° | mDR | UDS |
|---|---|---|---|---|---|---|---|---|
| <20 | 855 (855/0) | PST | 26.64 | 0.094 | 17.47 | 15.50 | 81.6 | 40.45 |
|  |  | RGB-only | 24.91 | 0.094 | 16.89 | 14.79 | 78.8 | 39.36 |
|  |  | Sim-only | 6.62 | 0.134 | 15.00 | 20.67 | 54.5 | 23.90 |
| | | **PST − RGB-only** [95% CI] | +1.73 [−6.32, +6.35] | −0.001 [−0.014, +0.020] | | +0.70 [−0.29, +1.74] | | +1.09 [−4.07, +3.91] |
| 20–40 | 485 (438/47) | PST | 27.26 | 0.126 | 17.95 | 15.87 | 95.7 | 40.83 |
|  |  | RGB-only | 23.20 | 0.136 | 17.55 | 15.06 | 95.5 | 38.46 |
|  |  | Sim-only | 11.03 | 0.190 | 16.95 | 25.12 | 79.0 | 25.41 |
| | | **PST − RGB-only** [95% CI] | +4.06 [+1.23, +6.81] | −0.010 [−0.018, −0.003] | | +0.81 [−0.48, +1.90] | | +2.37 [+0.52, +4.21] |
| 40–60 | 949 (949/0) | PST | 29.56 | 0.093 | 15.24 | 17.86 | 96.5 | 43.28 |
|  |  | RGB-only | 23.22 | 0.112 | 14.98 | 17.30 | 96.4 | 39.30 |
|  |  | Sim-only | 22.51 | 0.117 | 15.00 | 29.15 | 92.9 | 35.64 |
| | | **PST − RGB-only** [95% CI] | +6.33 [+3.32, +9.20] | −0.018 [−0.028, −0.008] | | +0.56 [−0.15, +1.17] | | +3.98 [+2.00, +5.75] |
| ≥60 | 335 (334/1) | PST | 30.87 | 0.091 | 14.46 | 20.13 | 99.7 | 43.98 |
|  |  | RGB-only | 25.52 | 0.108 | 14.64 | 19.39 | 99.7 | 40.59 |
|  |  | Sim-only | 29.31 | 0.100 | 15.00 | 26.47 | 98.5 | 41.17 |
| | | **PST − RGB-only** [95% CI] | +5.35 [−1.49, +9.62] | −0.018 [−0.031, −0.002] | | +0.74 [−0.79, +1.82] | | +3.39 [−0.70, +5.93] |

Notes on the bands:
- Test ranges span 13.1–79.2 m. The ≥60 m band is effectively 60–79 m.
- The 20–40 m band contains the 47 Matrice 400 frames, which have AP 0 in every arm (breakdown B2).
- Bands are contiguous stretches of a few sequences. The <20 m and ≥60 m intervals are wide because they hold only a few blocks.

**How to read C2**
1. **The PST − RGB-only gain grows with distance.** UDS: +1.1 (<20 m, CI spans 0), +2.4 (20–40), +4.0 (40–60), +3.4 (≥60 m, CI just spans 0). The relative ATE gain is 0 below 20 m and −1.0 to −1.8 points of range beyond it, with intervals excluding 0 from 20 m on. Most of the AP gain is position accuracy at range.
2. **Below 20 m the adapted arms differ mainly in detection rate** (mDR 81.6 vs 78.8). Their position errors are equal, and sim-only is weakest here: AP 6.6, mDR 54.5. The near band is the domain-gap band for the detector.
3. **AOE-90° is about 0.6–0.8° worse for PST in every band** (no interval excludes 0), consistent with C1-b.

### C2b. Simulated outdoor training objects with a non-empty LiDAR body map
Source: `D:/S2MUAV3D/sim_outdoor_carla/cache/train`, every labelled box in the 5,367 cached frames.

**Body-map rule** (`lidar/kzlid/center_det_mm.py` `targets()`): the map is non-empty if there are at least 5 drone-tagged LiDAR hits with |z_hit − z_box| < 0.75·max(lwh) + 0.3 m.

**How returns are counted:**
- Here they are counted as **raw returns**, not the 5×5-dilated sidecar pixels the network sees on its crop.
- The projection matches `tools/build_mm_sidecar_obj.py` exactly: LiDAR frame index + 6, LiDAR → world → RGB camera, `K_raw`, in-image points only.
- A return belongs to a box if it projects inside the box's projected 2-D extent, expanded by 30 % + 4 px.
- Small airframes are avata2, mavic-mini, phantom4 and drone-unk3 (= K3 Mini). Large airframes are m210, Matrice-600 and matrix-300.
- "Qualified" means the cache's own selection: inside the frame and ≤ 40 m.

| Band (m) | all objects: n / share ≥5 returns / median returns | small airframes | large airframes | qualified objects |
|---|---|---|---|---|
| 0-20 | 1858 / 96.5% / 43 | 1565 / 95.8% / 36 | 293 / 100.0% / 113 | 1825 / 97.0% / 43 |
| 20-40 | 9407 / 93.7% / 14 | 5708 / 89.6% / 9 | 3699 / 99.9% / 28 | 9326 / 93.7% / 14 |
| 40-60 | 4740 / 70.5% / 8 | 2330 / 42.7% / 4 | 2410 / 97.4% / 14 | 0 / – / – |
| 60+ | 1302 / 31.9% / 3 | 823 / 2.2% / 2 | 479 / 82.9% / 7 | 0 / – / – |

**Reading.** For small airframes the median raw-return counts are 9 / 4 / 2 at 20–40 / 40–60 / ≥60 m, which reproduces the earlier 9 / 4 / 2. The share with a non-empty body map falls with range:
- **Small airframes:** 96 % (<20 m), 90 % (20–40), 43 % (40–60), 2 % (≥60 m).
- **Large airframes:** 83 % or more in every band.
- **All objects:** 97 / 94 / 71 / 32 %.

So LiDAR body supervision is almost absent for small airframes beyond 60 m, and weak from 40 m. Yet the PST − RGB-only gain on the real test is largest at 40–60 m and holds at ≥60 m (C2). That gain at range comes mainly through position (ATE). The real test aircraft is almost entirely Matrice 4E; its size relative to the sim groups was not checked here.

Caveats:
- The "qualified" boxes, the cache's ≤ 40 m selection, have no 40–60 or ≥60 m rows by construction.
- The network applies the rule to dilated pixels on its crop, where one return becomes up to 25 pixels. These raw-return shares are therefore a lower bound on what the training loop treats as non-empty.

## Script: refiner with a box-hull mask instead of the IR silhouette (prepared, NOT run)
**Files**
- `review1113/boxhull/boxhull_commands.sh`: the run script (`bash -n` OK).
- `review1113/boxhull/make_boxhull_lib.py`: writes `review1113/boxhull/lib/`, the patched copies.
- `review1113/boxhull/align_hull.py`: checks the regenerated samples and swaps in the hull.

**Usage**
- `bash boxhull_commands.sh` runs MAV6D (`in2mav`), then AntiUAV3D (`out2antifs`).
- `TARGETS=mav bash boxhull_commands.sh` runs one target.
- `NO_WAIT=1` skips the start gate.

**Start gate.** The script polls every 5 min until `review1113/seeds/logs/run_all_seeds.log` contains `run_all_seeds end`. That is the last line `run_all_seeds.sh` writes after all 7 runs, success or not (`say "run_all_seeds end; failed: ..."`).

**What changes**
- **Sim samples.** The patched `refiner_x.py sim` re-runs the sim-only detector s0 on source-sim train, to recover each sample's deployment crop window (cx, cy, w). Those are not stored in `sim_meta.pkl`.
- **The hull.** For each sample it fills the convex hull of the 8 projected GT-box corners (255 inside) at full resolution and crops it with the same `crop_at(cx, cy, w)`.
- **Alignment.** `align_hull.py` requires the regenerated list to reproduce the clean one: same frames in the same order, |Δw| < 0.5 px and |Δδ\*| < 0.01. It also reports the IoU between the regenerated and clean IR silhouettes as a check on the crop geometry. If the lists match, only `sim_sil.npy` is swapped, and the clean `sim_rgb` / `sim_meta` are kept.
- **Fallback.** If the lists do not match, the regenerated rgb / meta / hull set is used consistently, and the run logs `FALLBACK`. Set `BH_STRICT=1` to make that an error instead.
- **CPU smoke test** on 58 outdoor-sim objects, with GT boxes standing in for the detections: a median 99.2 % of IR-silhouette pixels lie inside the hull crop, and the hull is about 3.6× the silhouette area.

**What follows from the swap.** The refiner code itself is unchanged, so the swap has these effects:
- The sim silhouette BCE + Dice target becomes the hull.
- The cross-domain composites use the hull as alpha, so the whole hull region, including the sim background inside it, is pasted.
- The real-crop silhouette targets (stages 2/3, from the EMA teacher) and the colour classes follow the hull-trained network.

**What is re-run.** Stages 1–3 for seeds 5 and 6 (same recipe), the stage-2 inference on det1 train, `chain_x targets` at the clean chain's **frozen w**, the final inference, and the seed average. Then `chain_x deploy` at the frozen w runs with prefix `boxhull`, followed by the val score and the test score (once). The last log line compares the result with the clean PST.

**Held fixed (copied from the clean chain).**
- Detectors det0/det1, and the Module-3 pseudo-labels. Those were built with the clean stage-2 refiner; re-training the self-trained detectors is out of scope.
- The real background bank and the real d0 crops.
- The orientation-student readings, height readings and w.

**Robustness**
- **OOM:** the batch env halves (REF_SIM_BS 8, REF_ST1_BS 128, REF_TRAIN_BS 64, REF_BS 128), with at most 4 retries logged to `logs/retry.log`. A halved training batch is a recipe deviation.
- **Resume:** stage 1 resumes per epoch, and stages 2/3 every 250 steps (`sup_patch`).
- **Skipping:** every step is skipped once its output exists.
- **Workers:** 0 everywhere. GPU steps go through `gpu_slot.sh`, and inference runs under a 0.2 memory fraction.
- **Isolation:** all outputs go under `review1113/boxhull/<combo>/` (`sx.O` / `sx.W` redirected and asserted).

**Estimated runtime** (from the clean chains' logs, GPU alone):

| Target | Total | sim | stage 1 | stage 2 | st2 infer | stage 3 | final infer | CPU |
|---|---|---|---|---|---|---|---|---|
| MAV6D | ~1 h 55 min | 13 min | ~27 | ~15 | ~8 | ~16 | ~18 | ~10 |
| AntiUAV3D | ~1 h 20 min | 4 min | ~13 | ~20 | ~3 | ~18 | ~11 | ~5 |

Both targets together take about 3 h 15 min, or 4–5 h with GPU queueing, starting after the seed chains finish.
