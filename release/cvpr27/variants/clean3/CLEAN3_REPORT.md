# Clean-protocol MAV6D chain (srcx/in2mav, test UDS 60.81): four more items for the paper

2026-11-13.
- CPU only (`CUDA_VISIBLE_DEVICES=""`). No training and no inference was run. No existing file was modified and no process was killed.
- Outputs are in `review1113/clean3/`. The figure also goes to `D:/CVPR27_UAV/figures/fig_qual.pdf`.

**Method.**
- Metrics use the pack `score.py` rules (`real_test` for MAV6D, `outdoor_test` for AntiUAV3D). Every row below is asserted equal to `S.metrics` to 1e-9.
- The bootstrap is a paired sequence bootstrap, B = 2,000, seed 0. Both arms are scored on the same resampled sequences, and AP is recomputed on each resample.
- Bootstrap code: `boot12.py`, which reuses `review1113/clean2/buildup/buildup.py` (`load_list`, `frame_q`, `stats`). Output: `boot12.json` and `boot12.log`.
- Splits: test has 4,800 frames in 10 sequences; val has 2,336 frames in 16 sequences.

---

## 1. Fused vs IR-only depth targets (ω = 0.190 vs ω = 0)

**Point values reproduced.**
- PST (`srcx/in2mav` `pcmd_fused`): test **60.81**, val **59.71**.
- ω = 0 (`review1113/clean2/w0` `w0_fused`): test **59.37**, val **58.77**.

| Split | Arm | AP | ATE (m) | ASE (cm) | AOE (°) | mDR (%) | UDS |
|---|---|---|---|---|---|---|---|
| test | PST, fused targets | 40.93 | 0.306 | 1.75 | 14.48 | 96.17 | **60.81** |
| test | IR-only targets (ω = 0) | 38.53 | 0.319 | 1.82 | 14.48 | 96.17 | 59.37 |
| val | PST, fused targets | 43.69 | 0.320 | 2.30 | 21.11 | 94.74 | **59.71** |
| val | IR-only targets (ω = 0) | 41.98 | 0.331 | 2.26 | 21.11 | 94.74 | 58.77 |

**PST − ω = 0.** Point [95 % CI].

| | test | val |
|---|---|---|
| **ΔUDS (deployed, both with test-time fusion)** | **+1.44 [+0.79, +2.28]** | **+0.95 [+0.42, +1.79]** |
| ΔAP | +2.40 [+1.40, +3.85] | +1.71 [+0.84, +3.14] |
| ΔATE (cm) | −1.28 [−2.03, −0.45] | −1.06 [−1.81, −0.43] |
| sequences where PST has the higher UDS | 9 / 10 | 12 / 16 |
| *ΔUDS without test-time fusion (nofuse arms)* | *+2.22 [+1.41, +2.83]* | *+1.74 [+0.88, +2.97]* |

**Reading.**
- Both intervals exclude 0 on both splits.
- AOE and mDR are identical between the arms (same rotations, same detections), so the whole gain comes from distance: AP and ATE.
- Without test-time fusion the gap is larger: +2.22 test / +1.74 val. Test-time fusion recovers part of what the IR-only targets lose.

---

## 2. RGB-only teachers (`srcx/in2mav_none`)

**Full rows.** Deployed, fused.

| Split | Arm | AP ↑ | ATE (m) ↓ | ASE (cm) ↓ | AOE (°) ↓ | mDR (%) ↑ | UDS ↑ |
|---|---|---|---|---|---|---|---|
| **val** | **RGB-only teachers** | **37.25** | **0.414** | **3.56** | **22.73** | **95.81** | **53.52** |
| val | PST | 43.69 | 0.320 | 2.30 | 21.11 | 94.74 | 59.71 |
| test | RGB-only teachers | 38.04 | 0.348 | 2.58 | 16.32 | 95.98 | 57.38 |
| test | PST | 40.93 | 0.306 | 1.75 | 14.48 | 96.17 | 60.81 |

Val UDS 53.52 matches the expected value. The median symmetric yaw is 15.31° for RGB-only vs 12.97° for PST on val, and 7.28° vs 6.24° on test.

**PST − RGB-only.** Point [95 % CI].

| | **test** | val |
|---|---|---|
| **ΔUDS** | **+3.43 [+1.92, +5.51]** | +6.19 [+4.53, +8.93] |
| **ΔAOE (°)** | **−1.84 [−3.14, −0.66]** | −1.62 [−2.70, −0.59] |
| ΔAP | +2.90 [+0.89, +5.74] | +6.44 [+3.50, +11.43] |
| ΔATE (cm) | −4.17 [−6.60, −2.29] | −9.39 [−13.79, −5.75] |
| ΔASE (cm) | −0.83 [−1.28, −0.50] | −1.27 [−1.70, −0.74] |
| Δmedian yaw (°) | −1.05 [−3.03, +0.22] | −2.34 [−4.07, −0.11] |
| sequences where PST has the higher UDS | 9 / 10 | 16 / 16 |

**Caveat.** AOE is averaged over each arm's own TP set. The detectors are identical across the two arms, so the TP sets differ only through the predicted size and distance. Test mDR is 96.17 vs 95.98, so the TP sets are nearly the same.

---

## 3. Qualitative figure (Fig. 3)

**Files.**
- `D:/CVPR27_UAV/figures/fig_qual.pdf`: 3.33 × 3.80 in, one column. It embeds 18 raster images (12 crops and 6 thumbnails); boxes, text and legend are vector (411 paths).
- `review1113/clean3/fig_qual.png`: 400 dpi copy.
- Scripts: `qual/select_qual.py`, which writes `selection.json`, and `qual/make_fig_qual.py`.
- Per-panel sentences, the selection rule and sequence/frame IDs are in **`FIG_QUAL.md`**.

**Layout.**
- Two blocks of 2 × 3 panels: MAV6D (clean-chain predictions) and AntiUAV3D (`out2antifs` predictions).
- In each block: 3 successes, 1 typical case, 1 position failure and 1 orientation failure.
- Colours: magenta = prediction, cyan = GT. Each panel is labelled with the airframe, range, Δd and AOE; AntiUAV3D panels also give Δd as % of range.
- AntiUAV3D crops carry a full-frame thumbnail, with the crop marked.

**Selection.** The rule was fixed in code before rendering, and no image was inspected.
- Pool: TP frames only.
- Position error and AOE are converted to percentile ranks.
- Success: both ranks ≤ 0.15, spread over range quantiles.
- Typical: both ranks closest to 0.5. It sits at the TP medians: 0.20 m / 11.7° on MAV6D, 7.8 % / 15.2° on AntiUAV3D.
- Failures: range-median frame among the worst 5 % for position, and likewise for AOE.

**Notable cases.**
- MAV6D (e): distance overestimated, 4.98 m vs 4.08 m.
- MAV6D (f): banked, blurred Phantom 4, AOE 44°.
- AntiUAV3D (k): Matrice 400 (0.98 × 0.76 m GT), predicted at roughly the mean small-drone size. The range is therefore 15.9 m instead of 37.5 m (58 % error). This illustrates the size-scale ambiguity behind the outdoor ASE.
- All MAV6D successes happen to be Phantom 4. That is the rule's outcome, and Phantom 4 is 73 % of the test frames.

---

## 4. Retraining the orientation students without the mirror loss: cost estimate (not run)

**Commands.**
- `review1113/clean3/nomir_commands.sh` (not executed).
- It uses lib copies generated by `review1113/clean3/make_nomir_lib.py` into `review1113/clean3/nomir/lib`. I ran the generator, which only writes there, and checked it: syntax, sx paths resolve under `clean3/nomir`, and crop arrays fall back read-only to the deployed work dir.

**Recipe.**
- Same as `run_x.sh`, except `--mir 0`: S6I recipe, gated `pseudo_tr1`, same-seed IR+LiDAR init, 2,250 steps, bs 8 + 4, mem 0.3, seeds 0 and 1.
- Then AdaBN (cap 0), then readings with the test-time mirror **view** kept as deployed (`MIRVIEW=0` drops the view as well).
- Then `chain_x deploy` with the frozen ω = 0.190, val scoring, and **one** test scoring.

**Isolation.**
- Writes go only under `review1113/clean3/nomir/`.
- Small inputs (~20 MB) are copied in. The 5.7 GB crop array and the 2.1 GB background bank are read in place via `OU_CROPS` / `OU_BANK`.

**OOM handling.**
- Each step is retried with its batch halved when its log shows a CUDA OOM:
  - students `--bs-src/--bs-tgt` 8/4 → 4/2 → 2/1;
  - AdaBN / readings `ORI_BS` 16 → … → 1;
  - refiner inference `REF_BS` 128 → … → 1;
  - refiner training `REF_TRAIN_BS` 64 → … → 8.
- Halvings are logged to `nomir/logs/oom.log`. A halved student batch is a recipe deviation.
- Workers: every DataLoader in `ou_train_x` is already `num_workers=0`, and nothing spawns.

**Timings from the clean run** (`in2mav/logs/run_x.log`, `ou_train_s*.log`, file times):

| Step | Clean run | Wall |
|---|---|---|
| copy inputs | – | < 1 min |
| student training, s0 ∥ s1 (2,250 steps each, run concurrently) | s0 06:48:12 → 07:11:56 (1,421 s); s1 06:49:54 → 07:13:16 (1,399 s) | **≈ 25 min** (≈ 47 GPU-min) |
| AdaBN × 2 | 07:13:16 → 07:16:20 | ≈ 3 min |
| readings val1 + te1 (2 students + mirror view) | val1 07:22:54 → 07:24:40, te1 → 07:27:22 | ≈ 4.5 min |
| deploy + score (CPU) | 08:24:18 → 08:24:22 | < 1 min |
| **Total, default (FULL=0)** | | **≈ 35 min wall, ≈ 55 GPU-min** |
| *FULL=1 adds:* tr1 reading (12,866 crops) | 07:16:20 → 07:22:54 | + 6.6 min |
| *FULL=1 adds:* step-3 targets + refiners s5 ∥ s6 (2,000 steps each) | 07:49:36 → 08:06:54 | + 17 min |
| *FULL=1 adds:* refiner inference, train/val/test × 2 seeds | 08:06:54 → 08:24:14 | + 18 min |
| **Total, FULL=1** | | **≈ 76 min wall, ≈ 115 GPU-min** |

**What FULL=0 vs FULL=1 means.**
- **FULL=0 (recommended):** only the students change. Rotations change, and so does the corner-PnP distance used by the test-time fusion (its b_k is re-estimated from unlabeled val images by the deploy rule). The depth refiners, c_k, r_k and s̄ stay frozen. This isolates the effect of the mirror loss on orientation.
- **FULL=1:** also rebuilds the step-3 targets from the new students' corner distances and retrains and re-infers the refiners. ω stays frozen at 0.190: recomputing w_sim needs `wsim_x.py`, which writes into `sx.O`/`sx.W` and has no redirected copy.

**Caveats.**
- These timings were measured while sharing the GPU.
- The OOM fallback, if triggered, makes runs faster but no longer equal to the recipe.
- After the run, add `('PST', 'nomir')` to `boot12.py` to get the paired CI.

---

## Files (`review1113/clean3/`)
- `boot12.py`, `boot12.json`, `boot12.log`: items 1 and 2.
- `qual/select_qual.py`, `qual/selection.json`, `qual/make_fig_qual.py`, `qual/fig_qual_crops.json`, `fig_qual.png`, `FIG_QUAL.md`: item 3. The PDF is at `D:/CVPR27_UAV/figures/fig_qual.pdf`.
- `nomir_commands.sh`, `make_nomir_lib.py`, `nomir/lib/*`: item 4. These are generated copies only; nothing was trained.

**Test-label use.** Labels were used to score existing predictions (items 1 and 2) and to compute per-frame errors for choosing and annotating the figure examples (item 3). No result was used to choose anything about the method.
