# C1 — reviewer diagnostics (CPU only, read-only on all existing files), 2026-11-13

Scripts (this folder): `c1a_bias.py` → `c1a_bias.json`; `c1a_decomp.py` → `c1a_decomp.json`; `c1b_boot_check.py` → `c1b_boot_check.json`.
All Python ran with `CUDA_VISIBLE_DEVICES=""`; nothing outside `review1113/` was written.

## 0. Reproduction check (AntiUAV3D v2 test, 2,624 frames)

Final reported prediction = `srcx/out2antifs/preds/pcmd_fused_test.pkl` (written by `chain_x.py deploy`, scored once by `score_x.py … antifs_test`,
i.e. `score_meta()` → `D:/UAV3D_baselines_pack/scripts/score.py::metrics` with GT `labels/sealed/antifs_test_meta.npz`).
`c1a_bias.py` re-implements `score_meta` verbatim (top-1 box per frame; TP = projected 2-D box IoU ≥ 0.1) and re-scores:

| | AP | ATE | ASE (cm) | AOE (°) | mDR | UDS | TP frames |
|---|---|---|---|---|---|---|---|
| PST (pcmd_fused_test) — reproduced | 27.886 | 9.93 % | 16.297 | 17.108 | 91.92 | 41.735 | 2,412 |
| PST — paper | 27.89 | 9.9 % | 16.3 | 17.1 | 91.9 | 41.74 | |
| Source-only (srconly_test) — reproduced | 16.037 | 13.20 % | 15.363 | 26.058 | 78.54 | 30.552 | 2,061 |
| Source-only — paper | 16.04 | 13.2 % | 15.4 | 26.1 | 78.5 | 30.55 | |

**Exact match.** All C1a statistics below are on these TP frames. "Distance" = ‖t‖ (camera-to-box-centre range, the quantity ATE is
normalised by); log z-component ratios differ from the range ratios by < 0.001 everywhere (`mean_logz` in the json).

**Test-split composition (important context):** only **2 sequences**: `train_seq_0010` = 2,576 frames, M4E, fx = 3,694.7 px;
`train_seq_0022` = 48 frames, M400, fx = 1,908.7 px (47) / 3,393.1 px (1). So aircraft, focal group and sequence are the
same partition on test; the fx 1,500–30,295 px range quoted for the dataset is not represented in test (only 3 distinct fx values).

## 1. C1a — signed distance bias (log(d_pred/d_gt); negative = predicted too close)

### 1a. Overall and by aircraft (= by focal group = by sequence)

| group | n (PST / src) | PST mean log | PST median log | PST median abs rel | src mean log | src median log | src median abs rel |
|---|---|---|---|---|---|---|---|
| all TP | 2412 / 2061 | −0.061 | −0.056 | 7.8 % | −0.083 | −0.062 | 11.5 % |
| M4E (fx 3,695 px) | 2365 / 2014 | −0.046 | −0.054 | 7.6 % | −0.062 | −0.057 | 11.1 % |
| M400 (fx 1,909 px) | 47 / 47 | −0.848 | −0.835 | 56.6 % | −0.997 | −0.989 | 62.8 % |
| fx 3,393 px | 0 / 0 (1 frame, not matched) | – | – | – | – | – | – |

### 1b. By true distance band (all TP; M400 frames all lie in 20–40 m)

| band | frames | n (PST / src) | PST mean / median log | PST med abs rel | src mean / median log | src med abs rel |
|---|---|---|---|---|---|---|
| < 20 m | 855 | 698 / 466 | +0.009 / −0.008 | 6.6 % | −0.121 / −0.130 | 12.9 % |
| 20–40 m | 485 | 464 / 383 | −0.098 / −0.036 | 7.8 % | −0.219 / −0.150 | 14.9 % |
| 20–40 m, M4E only | | 417 / 336 | −0.014 / −0.027 | 6.9 % | −0.110 / −0.124 | 13.2 % |
| 40–60 m | 949 | 916 / 882 | −0.084 / −0.086 | 8.9 % | −0.050 / −0.041 | 10.3 % |
| 60–80 m | 335 | 334 / 330 | −0.094 / −0.078 | 7.5 % | +0.038 / +0.053 | 8.7 % |

### 1c. Size bias, median log(pred/gt) per aircraft (TP frames)

| aircraft | GT l×w×h (m) | PST median pred | PST log l / w / h | src median pred | src log l / w / h |
|---|---|---|---|---|---|
| M4E | 0.307×0.388×0.150 | 0.537×0.533×0.247 | +0.56 / +0.32 / +0.50 (×1.75 / ×1.38 / ×1.65) | 0.532×0.527×0.236 | +0.55 / +0.31 / +0.46 |
| M400 | 0.98×0.76×0.48 | 0.565×0.561×0.260 | −0.55 / −0.30 / −0.61 (×0.58 / ×0.74 / ×0.54) | 0.532×0.527×0.236 | −0.61 / −0.37 / −0.71 |

### 1d. Where the distance comes from — decomposition (median logs over TP frames, `c1a_decomp.py`)

| | size_px | w2d | detector depth | refiner | PnP fusion | final |
|---|---|---|---|---|---|---|
| src-only, M4E | +0.227 | −0.282 | −0.057 | – | – | −0.057 |
| PST, M4E | +0.225 | −0.302 | −0.086 | +0.024 | −0.001 | −0.054 |
| PST, M4E < 20 m | +0.240 | −0.342 | −0.108 | +0.100 | +0.002 | −0.008 |
| PST, M4E 60–80 m | +0.189 | −0.237 | −0.035 | −0.041 | −0.009 | −0.078 |
| src-only, M400 | −0.719 | −0.275 | −0.989 | – | – | −0.989 |
| PST, M400 | −0.734 | −0.238 | −0.970 | +0.126 | +0.009 | −0.835 |

size_px = log(horizontal pixel span of a box of size s̄ at the true pose / span of the true box) — how much wider the assumed body
would look; w2d = log(projected width of the true box / the detector's predicted 2-D width), both at the 512-px input;
detector depth = log(z_det / z_gt) ≈ size_px + w2d; refiner = δ̂ − c_k; fusion = remainder (corner-PnP fusion).

## 2. Plain explanation (verified in code, not guessed)

**How the distance is produced (code path):**
1. Detector (`det_infer_x.py`, `SIZE_SOURCE=known`, `SIZE2D_MODE=width`; decoder `uavdet3d/utils/object_encoder_mav6d.py`
   "几何解深度" block): z_det solves fx · span_x(s̄, R̂) / z = ŵ2d, i.e. **distance = (assumed constant size s̄) / (predicted 2-D width)**,
   iterated on the full 8-corner projection. s̄ = (0.531, 0.527, 0.234) m is the median of the detector's own size head on unlabeled val
   images (self-calibration).
2. PST only (`chain_x.py::base`): z ← z_det · exp(δ̂ − c_k) (δ̂ = mean of 2 stage-3 depth refiners, |δ̂| ≤ log 1.4, c_k per colour class
   ≈ 0.02–0.03). The output size is s̄·exp(0.5(δ̂ − c_k)) — size is **output** from the correction, it does not feed back into distance.
3. PST only (`chain_x.py::cmd_deploy`): log z ← (1 − w) log z + w (log z_PnP − b_k), w = 0.096 (inverse variance on simulated val).
   z_PnP (corner PnP with the same s̄) is biased far by the oversize body: b_k (from unlabeled val images) = 0.53 / 0.99, i.e. ×1.7 / ×2.7,
   and subtracting b_k removes that offset; with w ≈ 0.1 the fusion moves the median by < 0.01.

So the suggested explanation ("distance comes from the apparent width via a refiner, not from the predicted size") is **only partly right**:
the detector's distance *does* use the size — it is size / width. The real reason the 1.4–1.75× size bias does not become a 1.4–1.75×
distance bias is that **numerator and denominator are inflated by about the same factor**: on M4E the oversized s̄ makes the body
≈ +0.22 log (×1.25) wider in pixels at the true range (the horizontal span mixes l and w with the pose, so less than ×1.4–1.75), and the
detector's 2-D width head over-reads the apparent width by ≈ +0.28–0.30 log (×1.33–1.35). The two cancel to −0.06 (src-only) / −0.09
(PST detector). s̄ is self-calibrated from the same network on unlabeled target images, so size and 2-D width are inflated
consistently; their ratio — the distance — is what is constrained. (Why the predicted 2-D width exceeds the GT box projection, e.g.
propellers/blur extending beyond the official box, was not checked here.) The refiner then adds a distance-dependent per-frame correction
(+0.10 below 20 m, −0.04 at 60–80 m) that removes the source-only detector's near-range underestimate (−0.13 → −0.01 below 20 m) and
lowers the median absolute error 11.1 % → 7.6 % on M4E; the remaining PST bias is a small −5 % (too close), growing to −8 % beyond 40 m.

**Where the size bias does propagate:** M400 (47 matched frames, one sequence). One constant s̄ is used for every aircraft; the M400
is 1.8× larger than s̄ in length, the ratio cancellation fails, and distance is −57 % (PST) / −63 % (src-only). ASE and the M400 rows
therefore reflect the single-size assumption; ATE 9.9 % is dominated by the 2,365 M4E frames (98 % of TP).

## 3. C1b — what sits behind each interval in the paper

No interval is reported on AntiUAV3D; every bracketed interval in `D:/CVPR27_UAV` (sec/*.tex, tables/*.tex) is on MAV6D.
Local MAV6D test (`E:/MAV6D/{mavic2,phantom4}/split/test.txt`): **10 sequences, 4,800 frames** (Mavic 2: 3 seqs / 1,290 frames —
01/0110, 02/0206, 02/0211; Phantom 4: 7 seqs / 3,510 frames — 01/0109, 01/0116, 01/0123, 01/0129, 02/0201, 02/0202, 02/0210).
MAV6D val (cmpeval/val_meta.npz): **16 sequences, 2,336 frames**.
AntiUAV3D v2: test **2 sequences** (2,576 M4E + 48 M400 frames); val 2 sequences (831 + 317, both M4E); train 6 sequences.

| interval location | metric | resampling unit (B, rng) | #sequences val / test | #frames val / test | #seeds / models behind each arm |
|---|---|---|---|---|---|
| Sec. 4 "Position" + supp Tab. ablation_a: +0.67 [+0.03, +1.27] val, +1.47 [+0.58, +2.04] test | UDS diff (fused vs IR-only depth-student targets) | whole-sequence cluster bootstrap, paired, B = see below, default_rng(0) (`wf23/mirror/boot_ci.py`, arms `pos_rek_w00` → `pos_z3S6Ik_w00`) | 16 / 10 | 2,336 / 4,800 | one fixed deployment per arm: depth refiner = mean of 3 seeds (5, 6, 7); orientation fixed (PSP × 5 + mirror); dev-version detector ensemble. Seeds are averaged, not resampled |
| Tab. ablation caption, step 1→2: +0.48 [+0.22, +0.74] / +0.61 [+0.40, +0.90] | UDS diff (+ orientation self-training) | same, B = 300 (`wf24/run_ladder.sh`), arms `lad1_w00` → `lad2_w00` | 16 / 10 | 2,336 / 4,800 | refiner 2 seeds (5, 6, IR-only) both arms; orientation 2 sim nets + mirror → 2 students (S6I s0, s1) + mirror; height net 1; detector ensemble. One fixed model per arm |
| Tab. ablation caption, step 2→3: +0.68 [+0.23, +1.55] / +1.48 [+0.54, +2.05] | UDS diff (+ fused targets) | same, B = 300, `lad2_w00` → `simpk_w00` | 16 / 10 | 2,336 / 4,800 | refiner 2 seeds (IR-only → fused, seeds 5, 6); orientation 2 students + mirror; one fixed model per arm |
| Tab. ablation caption, step 3→4: +1.40 [+0.04, +2.72] / +2.07 [+1.35, +3.13] | UDS diff (+ fusion at test) | same, B = 300, `simpk_w00` → `simpk_w15` | 16 / 10 | 2,336 / 4,800 | identical networks in both arms (only w = 0 vs 0.15 changes) |
| Tab. orient caption: color-only − labeler ×9: [−0.24, 1.28] / [−0.12, 1.17] | mean symmetric yaw diff (°) | whole-sequence cluster bootstrap, B = 2,000, default_rng(0); per draw the diff is computed per seed and averaged over seeds (`wf21/equal5/e5_compare.py`; = −(PSA − RA)) | 16 / 10 | 2,137 / 4,660 (frames with an orientation crop) | 3 seeds (0, 1, 2) per arm, single student networks (+BN), paired by seed |
| Tab. orient caption: color-only − IR+LiDAR ×5: [0.82, 1.84] / [−0.40, 0.56] | same | same (= −(T5A − RA)) | 16 / 10 | 2,137 / 4,660 | 3 seeds per arm, single students |
| Sec. 4 L11 (general statement "95 % intervals from a bootstrap over whole sequences") | – | – | – | – | – |

Consequences a reviewer may raise: (i) the chain intervals hold the trained models fixed, so they capture test-sequence sampling only,
not training-seed variance (the "3/3 seeds" statements and the orientation intervals do include seeds); (ii) with 10 test / 16 val
sequences, the interval endpoints are coarse; (iii) on AntiUAV3D v2 a sequence bootstrap is not meaningful — test has 2 sequences, one of
them 48 frames — so any AntiUAV3D interval would have to resample frames or blocks within the M4E sequence and should say so.

### Bootstrap re-check (`c1b_boot_check.py`)
Re-running the paired sequence bootstrap (copy of `wf23/mirror/boot_ci.py`, CPU) for the position interval, val split:
`pos_rek_w00 → pos_z3S6Ik_w00`, 16 sequences / 2,336 frames, stored [+0.030, +1.274]. B = 300 reproduces it exactly; B = 500 gives
[+0.015, +1.266]. So the position interval used **B = 300**, as did the three ladder intervals (`wf24/run_ladder.sh` line 23 calls
`boot_ci.py … 300`). The script hit its 590 s timeout after this first pair, so I did not re-run the other pairs. Their stored JSONs
(`wf23/mirror/boot_lad1_w00__lad2_w00.json`, `boot_lad2_w00__simpk_w00.json`, `boot_simpk_w00__simpk_w15.json`) match the paper's numbers
to two decimals, and every pair resamples the same 16 / 10 sequences and 2,336 / 4,800 frames (`valid` = all frames for every arm).
The orientation intervals come from `wf21/equal5/e5_compare.json` (B = 2,000, per `e5_compare.py`: `NB = 2000`). The paper values are the
sign-flipped RA − PSA and RA − T5A rows of `wf21/equal5/README.md` §6.
