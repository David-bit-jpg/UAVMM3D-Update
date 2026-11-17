# Clean-protocol MAV6D chain (srcx/in2mav, test UDS 60.81): analyses for the main text

2026-11-13. Everything comes from the clean chain `srcx/run_x.sh in mav` (outputs `srcx/in2mav/`, crops `D:/S2MUAV3D/srcx_work/in2mav/`).
- No training. No existing file was modified and no process was killed. All outputs are in `review1113/clean2/`.
- Items 1 and 3 ran on CPU only and reuse predictions. Item 4 ran GPU inference through `gpu_slot.sh light`, with memory fraction 0.15 and batch 4.
- Labels were read only by scorers. Metrics come from the pack `score.py` `real_test` rules: every build-up row is asserted equal to `S.metrics` to 1e-9, and the three re-built rows also match `srcx/lib/score_x.score_meta`.
- **Median yaw** is the quadrotor-symmetric yaw min_k |yaw(R_gt^T R_pred R_z(90k))| on the scorer's TP set. This is `ang_metrics` 'hs' in `wf21/orient_uda/eval_orient.py`, the definition behind tab_orient. tab_orient itself reports the *mean*, so both are given.

---

## 1. Step-by-step build-up (clean chain)

**Rows.**
- (a) Sim-only detector ensemble: `preds/srconly`.
- (b) + Step-1 self-trained detector ensemble, `recs/det1.pkl` as is. It uses its own rotation R_det, and its size is det1's s̄ box, so h = s̄_h.
- (c) **(new)** The det1 box with its rotation replaced by the step-2 student ensemble (S_ab0 + S_ab1 + mirror view, corner PnP with s̄). This is exactly the `chain_x.cmd_deploy` rule: `q[6:9] = R_student` where a reading exists, which is all 4,800 / 2,336 frames. Position and size are unchanged. Check: the rotations equal the deployed `pcmd_nofuse` rotations to within 3e-6°.
- (c′) Diagnostic: (c) plus the deploy height h = r_k·(l+w)/2, with the refiner correction forced to 0.
- (d) + Step-3 depth students: `pcmd_nofuse`. This is the refined distance, the α-coupled l, w and r_k height, with no fusion.
- (e) + Test-time fusion: `pcmd_fused`.

**Script.** `buildup/buildup.py`, which writes `buildup.json` and the row pickles `buildup/{b_det1,c_students,cp_students_rk}_{val,test}.pkl`.

### MAV6D test (4,800 frames, 10 sequences)
| Step | AP ↑ | ATE (m) ↓ | AOE (°) ↓ | median yaw (°) ↓ | UDS ↑ | (mean yaw) | (ASE cm) | (mDR %) |
|---|---|---|---|---|---|---|---|---|
| (a) sim-only detector | 26.87 | 0.283 | 20.32 | 10.14 | 49.51 | 14.25 | 1.74 | 69.8 |
| (b) + self-trained detector (R_det) | 34.04 | 0.356 | 25.44 | 11.44 | 53.38 | 15.30 | 2.54 | 96.2 |
| (c) + orientation students | 34.04 | 0.355 | **14.48** | **6.24** | 55.78 | 9.48 | 2.54 | 96.2 |
| (d) + depth students (no fusion) | 33.74 | 0.350 | 14.48 | 6.24 | 56.67 | 9.49 | 1.75 | 96.2 |
| (e) + test-time fusion = PST | **40.93** | **0.306** | 14.48 | 6.24 | **60.81** | 9.48 | 1.75 | 96.2 |
| *(c′) (c) + LiDAR height r_k, depth unchanged* | *34.04* | *0.355* | *14.48* | *6.24* | *57.00* | *9.49* | *1.56* | *96.2* |

### MAV6D val (2,336 frames, 16 sequences)
| Step | AP ↑ | ATE (m) ↓ | AOE (°) ↓ | median yaw (°) ↓ | UDS ↑ | (mean yaw) | (ASE cm) | (mDR %) |
|---|---|---|---|---|---|---|---|---|
| (a) sim-only detector | 11.85 | 0.368 | 27.84 | 18.59 | 35.53 | 21.02 | 2.20 | 44.3 |
| (b) + self-trained detector (R_det) | 36.93 | 0.372 | 34.55 | 21.71 | 51.29 | 21.81 | 3.48 | 94.8 |
| (c) + orientation students | 36.93 | 0.372 | **21.12** | **12.98** | 54.22 | 15.97 | 3.48 | 94.9 |
| (d) + depth students (no fusion) | 41.42 | 0.327 | 21.11 | 12.97 | 58.51 | 15.97 | 2.30 | 94.8 |
| (e) + test-time fusion = PST | **43.69** | **0.320** | 21.11 | 12.97 | **59.71** | 15.97 | 2.30 | 94.7 |
| *(c′) (c) + LiDAR height r_k, depth unchanged* | *36.93* | *0.371* | *21.11* | *12.97* | *55.92* | *15.97* | *2.13* | *94.8* |

### Step differences, paired sequence bootstrap (B = 2,000, seed 0; point estimate [95 % CI])
Both arms are scored on the same resampled sequences, and AP is recomputed on each resample.

| Step | test ΔUDS | test ΔAOE (°) | val ΔUDS | val ΔAOE (°) |
|---|---|---|---|---|
| (b) − (a) self-trained detector | +3.87 [−2.20, +11.19] | +5.12 [+2.29, +9.23] | +15.75 [+11.40, +20.32] | +6.71 [+2.45, +10.39] |
| (c) − (b) orientation students | **+2.40 [+1.58, +3.22]** | **−10.96 [−14.72, −7.24]** | **+2.93 [+2.26, +3.49]** | **−13.43 [−16.01, −10.34]** |
| (d) − (c) depth students | +0.89 [−1.31, +3.28] | 0.00 [0.00, 0.00] | **+4.29 [+2.48, +5.82]** | −0.01 [−0.03, 0.00] |
| (e) − (d) test-time fusion | **+4.14 [+2.70, +5.71]** | −0.01 [−0.01, 0.00] | +1.21 [−0.52, +3.20] | 0.00 [−0.03, +0.02] |
| (e) − (a) whole chain | **+11.30 [+4.89, +19.65]** | −5.84 [−11.28, +0.14] | **+24.18 [+20.16, +28.11]** | **−6.73 [−11.50, −2.27]** |
| *(d) split:* (c′) − (c) height r_k | +1.22 [+0.80, +1.82] | 0.00 | +1.70 [+1.10, +2.21] | −0.01 |
| *(d) split:* (d) − (c′) distance only | −0.33 [−2.41, +1.75] | 0.00 | +2.59 [+0.85, +4.27] | 0.00 |

Median-yaw differences, same bootstrap:

| Step | test | val |
|---|---|---|
| (b) − (a) | +1.29 [−1.30, +3.75] | +3.12 [−3.20, +6.12] |
| (c) − (b) | −5.20 [−11.56, −1.18] | −8.74 [−13.95, −3.56] |
| (e) − (a) | −3.91 [−11.92, +0.58] | −5.62 [−14.84, −0.14] |

All other median-yaw differences are within ±0.06.

**Reading the table.**
- **AOE / yaw across (a)→(b) are on different TP sets.** Sim-only detects 69.8 % of test frames; det1 detects 96.2 %, and the extra frames are harder.
- R_det itself is worse than sim-only. The step-1 self-training uses **no rotation loss** (CLEAN_REPORT §5), so its R_det is a drifted byproduct. The students then lower AOE by 11.0° (test) / 13.4° (val), with 10 / 10 and 16 / 16 sequence intervals excluding 0.
- **Step (d) bundles two things:** the refined distance, and the size coupling with r_k height. That is why (c′) is shown.
  - The r_k height alone gives +1.22 test / +1.70 val UDS (ASE 2.54 → 1.56 cm).
  - The distance part is +2.59 [+0.85, +4.27] on val but −0.33 [−2.41, +1.75] on test.
  - The test cause is the +4.6 % global distance over-estimate traced in CLEAN_REPORT §4c. The refiner removes only c_k, so it cannot fix the scale it inherited from det1. Its per-frame spread does improve: 9.5 % → 8.1 % on test.
- **(e) − (d), fusion:** +4.14 test (CI excludes 0), +1.21 val (CI crosses 0).
- **(b) − (a) on test crosses 0:** +3.87 [−2.20, +11.19]. Only 10 sequences, and the AP gain is sequence-dependent.

---

## 2. Cost of retraining step-3 depth students with ω = 0 (estimate only, not run)

**What changes.** `chain_x.py targets 0` gives δ = log(z_pl / z_det0) on the same 4,751 crops, with no corner term. Seeds 5 and 6 are retrained from the same stage-2 weights, with 2,000 steps, the same sim batches and REF_REAL = d0. Deployment is unchanged: the fused arm keeps the frozen ω = 0.190, and the nofuse arm is also written.

**Timings from the clean run** (`run_x.log` and `ref/train_st3_s*.log`, file times):

| Stage | Clean run | Wall time |
|---|---|---|
| targets (CPU) | 07:49:36 → ~07:50 | < 1 min |
| stage-3 training, s5 ∥ s6 concurrent | step 1800 at 817 s / 818 s → 2,000 steps ≈ 905 s each; `ref_st3_s5.pt` 08:05, `s6` 08:06 | **≈ 16 min wall (≈ 30 GPU-min)** |
| refiner inference, 2 seeds × train / val / test, sequential (train is needed for c_k) | 08:06 → 08:24 (`fin_s5_train` 08:10, `s6_train` 08:14, val 08:16 / 08:17, test 08:22 / 08:24) | **≈ 18 min** |
| deploy + scoring (CPU) | 08:24:19 → 08:24:21 | < 1 min |
| **Total** | 07:49:36 → 08:24:21 | **≈ 35 min wall, ≈ 50 GPU-min** |

**Caveats.**
- These timings were measured while sharing the GPU with other jobs.
- The `review1113/clean/lib` copy (used below so that nothing under `srcx/in2mav` is written) runs refiner inference at batch 8 instead of 128. Allow ≈ 45 min wall in that case. Training batches are unchanged.
- Add ≈ 2 min to copy the ~1.3 GB of inputs.

**Exact commands.** They are in `cost/w0_commands.sh` (not executed):
```bash
export SX_SRC=in SX_TGT=mav SX_TEACH=full PYTHONUTF8=1 TORCH_HOME=E:/torch_home PYTHONPATH=E:/Open3DUAVDet UAV_PNP_EPNP_FALLBACK=1
export CL_AO=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/clean2/w0      # clean/lib copies write only here
# copy read-only inputs: recs/det0,det1.pkl; ref/{sim_meta.pkl,sim_rgb.npy,sim_sil.npy,bank.npy,real_d0_meta.pkl,real_d0_rgb.npy,ref_st2_s5.pt,ref_st2_s6.pt,inf_st2_det1_train.pkl};
#   ou/read_{tr1,val1,te1}_stu.pkl; kps_{tr1,val1,te1}.pkl; work/oc_{tr1,val1,te1}.pkl   (see the script)
$PY $L/chain_x.py targets 0
for j in 5 6; do ( REF_REAL=d0 REF_DP=$CL_AO/ref/dp_fused.npy $SLOT light $PY $L/refiner_x.py train 3 $j ) & sleep 30; done; wait
for sp in train val test; do for j in 5 6; do $SLOT light $PY $L/refiner_x.py infer $CL_AO/ref/ref_st3_s$j.pt $CL_AO/recs/det1.pkl $sp $CL_AO/ref/fin_s${j}_$sp.pkl; done
  $PY srcx/lib/avg_inf.py $CL_AO/ref/inf_fin_det1_$sp.pkl $CL_AO/ref/fin_s5_$sp.pkl $CL_AO/ref/fin_s6_$sp.pkl; done
$PY $L/chain_x.py deploy 0.190282282581008 w0
$PY $L/score_x.py $CL_AO/preds/w0_{fused,nofuse}_val.pkl val ; $PY $L/score_x.py $CL_AO/preds/w0_{fused,nofuse}_test.pkl real_test   # test once
```
Here `L = review1113/clean/lib`, `PY = D:/Miniconda3/envs/city/python.exe` and `SLOT = bash gproto/gpu_slot.sh`.

---

## 3. The two distances on the clean chain

**Quantities.**
- z_IR is the refined IR distance: z of `pcmd_nofuse`, i.e. t_det1·e^(δ̂−c_k) with phase-3 refiners s5 + s6.
- z_cor is the corner-PnP distance of the student ensemble (+ mirror, s̄ box), `chain_x.zL_table` on the det1 crops, before b_k.

**Method.**
- e = log(z / z_gt) on the TP frames of PST that have a corner reading. That is every TP frame: test 4,616, val 2,213.
- Global bias b = median e, per quantity and split.
- Spread = 1.4826·median |e − b|, in %.
- Correlation = Pearson of the de-biased errors. Correlation within a class does not depend on which offset is removed.
- Brightness class is the deploy rule: phase-3 silhouette brightness, nearest centre.

**Script.** `dist/dist_clean.py`, which writes `dist_clean.json`.

**Main numbers.** Spreads use each class's own offset, matching the per-class β_k the method uses. The global-offset version is below.

| Split | Class (frames) | IR spread | corner spread | corr (Pearson) | corr (Spearman) |
|---|---|---|---|---|---|
| test | dark (1,148) | 10.6 % | 20.6 % | 0.46 | 0.47 |
| test | bright (3,468) | 7.5 % | 9.6 % | 0.11 | 0.19 |
| test | overall (4,616) | 8.1 % | 15.1 % | 0.28 | 0.22 |
| val | dark (1,153) | 10.5 % | 20.5 % | 0.58 | 0.45 |
| val | bright (1,059) | 6.8 % | 14.4 % | 0.09 | −0.12 |
| val | overall (2,213) | 8.7 % | 24.2 % | 0.41 | 0.22 |

**Global offsets removed** (b over all frames, as asked):

| Split | Global bias IR / corner | IR spread dark / bright / all | corner spread dark / bright / all | class residual median IR / corner (dark; bright) |
|---|---|---|---|---|
| test | +6.6 % / +28.5 % | 10.7 / 7.6 / 8.1 % | 36.0 / 12.5 / 15.1 % | +1.5 / +24.2 %; −0.5 / −4.4 % |
| val | +2.0 % / +43.8 % | 10.1 / 6.9 / 8.7 % | 23.5 / 25.0 / 24.2 % | +1.0 / +13.4 %; −0.4 / −13.5 % |

**Replacement for the development sentence** (dev was IR 8.3 / 5.1 %, corner 15.5 / 8.6 %, corr 0.36 / 0.00, dark / bright), test, per-class offsets:
- IR 10.6 / 7.5 %;
- corner 20.6 / 9.6 %;
- correlation 0.46 / 0.11.

The pattern is the same as in the development version:
- IR is tighter in both classes.
- The corner distance is noisier on dark airframes.
- The two errors are nearly independent on bright frames and moderately correlated on dark ones.

All levels are higher in the clean chain.

**Caveats.**
- The raw corner distance has a large, class-dependent offset: +53 % dark / +24 % bright on test. A single global offset therefore inflates the dark corner spread to 36 %. This is why the method uses per-class β_k.
- On val, Pearson and Spearman disagree (0.41 vs 0.22 overall; bright 0.09 vs −0.12), because a few large-error frames dominate Pearson.

---

## 4. ShaSpec re-evaluated on the clean chain's orientation crops

**Feasible, and done: inference only, seeds 0 and 1, about 6 min of GPU in total.**

**Sources.**
- Checkpoints: `output/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_shaspec/SHASPEC_s{0,1}/ckpt/checkpoint_epoch_8.pth`, config `cfgs/.../sim_full_r34_kp_w12_zoom_shaspec.yaml`. These are the ones `cmpeval/auto_eval.sh` evaluated.
- Evaluation code: `cmpeval/cmp_eval.py`, kz kind.

**Why the main-table number is development-bound.**
- The main-table ShaSpec value, 20.65° = mean of 20.95 and 20.39, is `test_adabn`.
- It was computed on `cmpeval/test_crops_u8.npy`, which are development-detector windows (`st2p_*_kzt7q`).
- Its AdaBN used the development `kzst6/data/crops_train_u8`.
- It averages over all 4,800 frames, not only true positives.

**What I ran.** Script `shaspec/shaspec_clean.py`, which imports `cmp_eval` unchanged for model build, input norm and PnP.
- **Crops:** the clean `oc_{val1,te1}.npy`, i.e. the det1 windows the clean students read. They have the same 512×288 uint8 format and channel order as the cmpeval crops; I checked this on overlapping frames.
- **Reading:** the 5×5 heat-map peak around the det1 centre projected into the crop, `uc, vc`. Corners are read as `uv + kp·64` (the cmp_eval rule), then PnP with K_crop and the clean s̄ box (the same box the students use).
- **AdaBN:** 2,000 uniformly spaced clean unlabeled train crops `oc_tr1`.
- Batch size is 4; it was 16 in cmp_eval.
- **Sanity:** the raw seed-0 reading of the original dev crops gives 24.76° / 23.58° (mean / median geodesic) vs 24.74° / 23.54° in `res_SHASPEC_s0.json`. The 0.02° gap is batch-size numerics.

**Evaluation.** Script `shaspec/eval_shaspec.py`, which writes `eval_shaspec.json`.
- Frame set: the **same 4,616 test / 2,213 val TP frames of the clean PST**.
- AOE uses the pack `sym`; yaw is defined as in item 1.
- The paired sequence bootstrap is ShaSpec (seed mean) − PST, with B = 2,000.

### Test (4,616 TP frames of clean PST)
| Orientation source (same det1 crops) | AOE (°) | median yaw (°) | mean yaw (°) | < 15° (%) | AOE all 4,800 frames | UDS with this rotation in PST boxes |
|---|---|---|---|---|---|---|
| **PST students (2 + mirror)** | **14.48** | **6.24** | **9.48** | **64.8** | **15.05** | **60.81** |
| ShaSpec, no AdaBN (mean s0, s1) | 24.09 | 14.50 | 17.48 | 33.1 | 24.30 | 58.70 |
| ShaSpec, AdaBN on clean train crops (mean s0, s1) | 24.36 | 12.29 | 16.07 | 39.9 | 24.71 | 58.65 |
| per seed, AdaBN s0 / s1 | 25.03 / 23.68 | 12.13 / 12.45 | 16.12 / 16.02 | 40.4 / 39.3 | 25.40 / 24.03 | 58.51 / 58.80 |
| *ref: main table, dev crops + dev AdaBN, all frames (s0 / s1)* | – | – | – | 42.3 / – | *20.95 / 20.39 (mean 20.65)* | – |
| *ref: dev crops, no AdaBN (s0 / s1)* | – | – | – | – | *24.74 / 24.51* | – |

### Val (2,213 TP frames)
| Orientation source | AOE (°) | median yaw (°) | mean yaw (°) | < 15° (%) | AOE all frames | UDS |
|---|---|---|---|---|---|---|
| **PST students** | **21.11** | **12.97** | **15.97** | **39.0** | **21.94** | **59.71** |
| ShaSpec, no AdaBN | 28.06 | 20.59 | 21.25 | 20.3 | 28.20 | 58.19 |
| ShaSpec, clean AdaBN | 27.66 | 18.94 | 20.35 | 25.2 | 27.63 | 58.27 |

### ShaSpec − PST, paired sequence bootstrap
| | test ΔAOE | test Δmedian yaw | val ΔAOE | val Δmedian yaw |
|---|---|---|---|---|
| no AdaBN | +9.61 [+8.19, +10.67] | +8.26 [+5.95, +9.59] | +6.95 [+4.79, +9.00] | +7.62 [+3.81, +10.69] |
| clean AdaBN | +9.88 [+7.15, +12.36] | +6.05 [+3.57, +7.98] | +6.55 [+3.88, +9.26] | +5.97 [+2.48, +8.49] |

**Notes.**
- **Paper value.** For the clean main table, I suggest ShaSpec **AOE 24.4° / median yaw 12.3°** (clean AdaBN, the analogue of the old protocol) on the clean TP frames, against PST 14.48° / 6.24°. On all frames it is 24.7°.
- **The clean ShaSpec AOE is worse than the old 20.65°.** Without AdaBN, clean and dev crops give nearly the same AOE (24.30 vs 24.63, all frames). So the old 20.65 came mainly from AdaBN on the *development* train crops.
  - On the clean crops, AdaBN improves median yaw and the share below 15°, but not the mean AOE. I did not separate how much of that is the different AdaBN crop set (all 12,866 det1 train windows) and how much is batch 4 vs 16, which changes the cumulative BN variance estimate.
  - Either way, ShaSpec stays 6–10° behind the students on both splits, and every interval excludes 0.
- Per-frame rotations: `shaspec/pf_s{0,1}_{val,test}_{raw,adabn}.npz`. Inference log: `shaspec/shaspec_infer.log`.

---

## Files (all under `review1113/clean2/`)
- `buildup/buildup.py`, `buildup.json`, `buildup.log`, and the row pickles (b), (c), (c′) for val and test. This is item 1.
- `cost/w0_commands.sh`: item 2. Not executed.
- `dist/dist_clean.py`, `dist_clean.json`: item 3.
- `shaspec/shaspec_clean.py`, `eval_shaspec.py`, `eval_shaspec.json`, `pf_*.npz`: item 4.

**Test-label use.**
- Scoring of existing predictions.
- Scoring of the three re-built rows (b), (c), (c′).
- The item 3 diagnostics.
- One evaluation of the ShaSpec readings, plus the seed-0 sanity reading on the dev crops.

No result was used to choose anything.
