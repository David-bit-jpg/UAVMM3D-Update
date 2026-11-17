# What the LiDAR-trained height ratio buys for box size (review1113/height, 2026-11-13)

CPU only. No training, no GPU, no existing file modified. Labels were read only by the scorers (MAV6D `cmpeval/paper_table.metrics` = pack `score.py` `real_test`; AntiUAV3D pack `score.py` `outdoor_test` on the sealed v2 meta). Nothing was chosen based on the results.

## Setup
- **PST (deployed).** h = r_k · (l+w)/2. r_k is the median height-network reading (LiDAR structure points, KPS seed 0) for the frame's brightness class, computed on unlabeled train. MAV6D r_k = 0.3800 (dark) / 0.7111 (bright). AntiUAV3D r_k = 0.4621 / 0.4664.
- **Constant aspect (no LiDAR).** h = c · (l+w)/2 with c = s̄_h / mean(s̄_l, s̄_w). This is exactly the rule the IR-only / RGB-only arms use (`srcx/run_x.sh` L105–112).
  - MAV6D: s̄ = (0.34967, 0.34855, 0.19960), so **c = 0.5717**. wf24 §2 used the rounded 0.573; it gives the same numbers within ±0.05 cm, see `height_results.json` row `A_const_0573`.
  - AntiUAV3D: s̄ = (0.5307, 0.5268, 0.2344), so **c = 0.4433**.
- **True aspect h / mean(l, w).** Mavic 2 0.389, Phantom 4 0.676, M4E 0.431.
- **Variants.** l, w, position and rotation are unchanged unless stated.
  - **(a) output box only.** h is replaced post hoc in the reproduced deployed predictions.
  - **(b) box + corner PnP.** The chain is rerun with `FE_HTR=0.5717` (the existing switch in `chain_eval_m12.py`). The PnP corner model gets the constant height, so the PnP distance (and b_k, recomputed label-free on val images) changes, and so does the PnP rotation.
  - **(b′)** = (b) position with the deployed rotation, so that only the PnP distance sees the change.
  - On AntiUAV3D, srcx's corner PnP uses the s̄ box (`oc_*.pkl['size']`), never r_k. **So (b) ≡ (a) there.**
- **Height error.** |Δh| on the scorer's TP set: the top-confidence box per frame with 2D IoU ≥ 0.1, which is the same set ASE averages over. "% of GT h" is |Δh| / h_gt.

## 1. MAV6D (reproduced deployed run: test UDS 70.16, val 62.49, identical to M12)

### 1a. Headline metrics
| Split | Height source | AP | ATE (m) | ASE (cm) | AOE (°) | mDR (%) | UDS |
|---|---|---|---|---|---|---|---|
| test | **PST: LiDAR height ratio r_k** | 56.44 | 0.226 | **1.81** | 13.54 | 99.81 | **70.16** |
| test | (a) constant aspect, box only | 56.44 | 0.219 | 2.68 | 13.50 | 99.60 | 69.13 |
| test | (b) constant aspect, box + corner PnP | 56.37 | 0.220 | 2.68 | 13.68 | 99.60 | 69.05 |
| test | (b′) = (b), deployed rotation | 56.37 | 0.220 | 2.68 | 13.50 | 99.60 | 69.09 |
| val | **PST: r_k** | 48.33 | 0.312 | **2.63** | 20.57 | 100 | **62.49** |
| val | (a) box only | 48.33 | 0.312 | 3.87 | 20.57 | 100 | 60.93 |
| val | (b) box + corner PnP | 48.75 | 0.311 | 3.87 | 20.71 | 100 | 61.13 |

- In (a), ATE and mDR move only because 10 test frames drop out of the IoU ≥ 0.1 TP set when the box gets taller or shorter. Those frames had large position errors, so ATE falls slightly; AP does not change.
- In (b), the PnP distance with a constant height changes AP by −0.07 (test) / +0.41 (val), which is noise-level, and AOE by +0.15°. **The height ratio matters almost only through box size.**

### 1b. Size error breakdown (cm, mean over TPs)
| Split | Aircraft (TPs) | Height source | ASE | \|Δl\| | \|Δw\| | **\|Δh\| mean** | \|Δh\| median | \|Δh\| % of GT h (mean / median) |
|---|---|---|---|---|---|---|---|---|
| test | all (4791 / 4781) | r_k | 1.81 | 2.54 | 1.12 | **1.77** | 1.66 | 9.3 / 8.2 |
| test | all | constant | 2.68 | 2.53 | 1.12 | 4.41 | 3.25 | 27.8 / 14.1 |
| test | Mavic 2 (1289) | r_k | 3.39 | 6.85 | 1.84 | **1.49** | 1.48 | 12.4 / 12.4 |
| test | Mavic 2 | constant | 5.59 | 6.84 | 1.83 | 8.10 | 8.15 | 67.5 / 67.9 |
| test | Phantom 4 (3502) | r_k | 1.23 | 0.95 | 0.86 | **1.87** | 1.73 | 8.1 / 7.5 |
| test | Phantom 4 | constant | 1.62 | 0.95 | 0.86 | 3.06 | 3.08 | 13.3 / 13.4 |
| val | all (2336) | r_k | 2.63 | 4.32 | 1.58 | **1.99** | 1.66 | 12.2 / 10.7 |
| val | all | constant | 3.87 | 4.32 | 1.58 | 5.73 | 7.40 | 42.4 / 61.7 |
| val | Mavic 2 (1259) | r_k | 3.50 | 6.93 | 1.92 | **1.66** | 1.43 | 13.8 / 11.9 |
| val | Mavic 2 | constant | 5.67 | 6.93 | 1.92 | 8.15 | 8.08 | 67.9 / 67.3 |
| val | Phantom 4 (1077) | r_k | 1.61 | 1.28 | 1.18 | **2.38** | 2.03 | 10.4 / 8.8 |
| val | Phantom 4 | constant | 1.78 | 1.28 | 1.18 | 2.89 | 2.88 | 12.6 / 12.5 |

Variants (b) and (b′) have the same size errors as (a) to two decimals.

### 1c. Differences (constant − r_k) with sequence-bootstrap 95 % intervals
Sequences are resampled with replacement, B = 2000, seed 0. Variant (a); (b) and (b′) differ by ≤ 0.05.

| Split | Group (sequences) | ΔASE (cm) | Δ mean \|Δh\| (cm) |
|---|---|---|---|
| test | all (10) | +0.87 [+0.41, +1.44] | +2.64 [+1.23, +4.35] |
| test | Mavic 2 (3) | +2.20 [+2.16, +2.22] | +6.61 [+6.58, +6.67] |
| test | Phantom 4 (7) | +0.39 [+0.31, +0.47] | +1.19 [+0.93, +1.40] |
| val | all (16) | +1.24 [+0.57, +1.76] | +3.74 [+1.71, +5.27] |
| val | Mavic 2 (7) | +2.17 [+2.04, +2.22] | +6.49 [+6.13, +6.67] |
| val | Phantom 4 (9) | +0.17 [−0.21, +0.37] | +0.51 [−0.64, +1.11] |

The Mavic 2 test interval rests on only 3 clusters, so it is optimistic. The effect itself is large and consistent across splits.

### 1d. Side by side with the 10-18 cross-frame version (same definition)
The 10-18 numbers are from `release_20261018h/code/size_iv.py`: **median** |Δh| over all frames with a prediction (top box), with no IoU filter. The row below uses that same definition.

| Box-height error, median \|Δh\| (cm) | Mavic 2 val / test | Phantom 4 val / test |
|---|---|---|
| 10-18 before (release_20261015kps) | 7.7 / 7.6 | 3.0 / 3.1 |
| 10-18 LiDAR keypoints, per-clip rolling median over past frames (cross-frame), coefficient 0.843 | 1.1 / 1.4 | 3.2 / 1.8 |
| **now: constant aspect of s̄ (no LiDAR)** | 8.08 / 8.16 | 2.88 / 3.08 |
| **now: PST r_k, strictly single frame** | **1.43 / 1.49** | **2.03 / 1.74** |

(The mean over TPs is in 1b: Mavic 1.66 / 1.49, Phantom 2.38 / 1.87.) The single-frame per-class constant matches the cross-frame rolling median on Mavic 2 and beats it on Phantom 4 val. The 10-18 "before" row is numerically close to the constant-aspect row, but I did not check which height rule release_20261015kps used.

**Why it works.** One aspect cannot fit both airframes: 0.389 vs 0.676, and the constant is 0.572. The constant over-estimates Mavic 2 height by about 68 %. The LiDAR-taught ratio gives 0.380 / 0.711, close to each true aspect. Diagnostic only: the brightness class matches the airframe on 98.5 % / 94.5 % (val Mavic / Phantom) and 99.6 % / 98.4 % (test) of frames.

## 2. AntiUAV3D v2 test (out2antifs, deployed UDS 41.74, reproduced)
| Height source | AP | ATE (rel.) | ASE (cm) | AOE (°) | mDR (%) | UDS | \|Δl\| | \|Δw\| | **\|Δh\| mean / median** | \|Δh\| % GT | mean predicted l / w / h (m, TPs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **PST r_k** (0.462 / 0.466) | 27.89 | 0.099 | 16.30 | 17.11 | 91.9 | 41.735 | 23.71 | 15.01 | 10.17 / 9.80 | 66.1 | 0.541 / 0.537 / 0.249 |
| (a) = (b) constant 0.4433 | 27.89 | 0.099 | 15.97 | 17.11 | 91.9 | 41.735 | 23.71 | 15.01 | 9.20 / 8.79 | 59.5 | 0.541 / 0.537 / 0.239 |
| IR-only arm (out2antifs_ir) | 28.08 | 0.099 | 16.03 | 17.19 | 91.9 | 41.84 | 23.77 | 15.07 | 9.23 / 8.78 | 59.7 | 0.542 / 0.538 / 0.239 |
| RGB-only arm (out2antifs_none) | 23.39 | 0.111 | 15.97 | 16.45 | 90.9 | 38.92 | 23.68 | 14.98 | 9.25 / 8.94 | 59.8 | 0.541 / 0.537 / 0.240 |

Per aircraft for PST r_k → constant (TPs; 2 test sequences, so these are point values):
- **M4E** (GT 0.307 × 0.388 × 0.150, 2365 TPs): ASE 16.07 → 15.73; |Δl| 23.36, |Δw| 14.91; |Δh| 9.94 → 8.93.
- **Large airframe** (0.98 × 0.76 × 0.48, 47 TPs): ASE 27.81 → 28.17; |Δh| 22.00 → 23.05.

**Why the nopriv table shows no ASE difference (16.3 / 16.0 / 16.0):**
1. **Scale, not aspect, dominates.** Predicted l and w are about 0.54 m against the true 0.307 / 0.388 (+75 % / +38 %). This follows the 1.4–1.7× enlarged outdoor sim assets inherited through s̄. |Δl| + |Δw| account for about 80 % of ASE. Because h = ratio × mean(l, w), the height inherits the same 1.55× scale error (0.24–0.25 m vs 0.150 m), whatever the ratio.
2. **The ratio is already right for the M4E, under both rules.** The true M4E aspect is 0.431. The LiDAR r_k is 0.462 and the constant is 0.443, which differ by only 4 %. With true l and w, either rule would give h within about 1 cm. The constant looks slightly better here only because its smaller ratio partly cancels the inflated scale.
3. **There is nothing for the brightness classes to separate.** The test set is 98 % one airframe (the large one has 48 frames). The two classes get nearly identical ratios (0.462 / 0.466), unlike MAV6D's two airframes with very different aspects (0.39 vs 0.68). So the height network cannot add anything over a constant on this test set.

The −0.3 cm ASE of the IR-only / RGB-only arms comes entirely from the height term. |Δl| and |Δw| are equal across arms. Corner PnP on AntiUAV3D never sees r_k, so AP, ATE and AOE are unchanged.

## Files (all in `gproto/review1113/height/`)
- `height_eval.py`: all tables, bootstrap and per-aircraft splits.
- `height_results.json`: every number above, plus the 0.573 sensitivity row and the brightness-class diagnostic.
- `mav_rk/`: reproduced deployed run.
- `mav_htr/`: variant (b) rerun. Each has `results.json` and `fe_deploy_fused_{val,test}_P.pkl`.
- `logs/`.
- Commands (bash, from `E:/Open3DUAVDet/tools`, env `PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet`):
  - Deployed run: `NL_ARM=nl075 FE_UNC=z2 FE_ORI=ouS6I2M FE_NMEM=4 M12_WS=w M12_OUT=<height>/mav_rk python review1113/M12/fuse_eval_m12.py`
  - Variant (b): the same command plus `FE_HTR=0.5717326440954079 M12_OUT=<height>/mav_htr`.
  - Tables: `python height_eval.py`.
- AntiUAV3D inputs: `srcx/out2antifs{,_ir,_none}/preds/pcmd_fused_test.pkl` (deployed, unchanged) and `srcx/out2antifs/deploy_pcmd.json` (s̄, r_k).
