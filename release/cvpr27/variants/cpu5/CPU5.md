# CPU5: five CPU-only checks for the paper (2026-11-14)

- CPU only (`CUDA_VISIBLE_DEVICES=""`). Existing predictions only: no training, no inference. No process was killed. Writes only to `review1113/cpu5/`.
- Script: `cpu5.py`. Outputs: `cpu5.json`, `cpu5.log`, and the rebuilt AntiUAV3D row pickles `anti_{b_det1,c_students,cp_students_rk}_{val,test}.pkl`.
- **Metrics.**
  - MAV6D: `review1113/clean2/buildup/buildup.py`, pack `score.py` `real_test` rules.
  - AntiUAV3D: pack `score.py` `outdoor_test` rules. ATE is relative to GT distance. The AP code is the same as in `A3/a_block_boot.py`.
  - Every row is asserted equal to `S.metrics` to 1e-9. Where a stored `.score_*.json` exists, the row also equals it to 1e-6.
- **Bootstraps.**
  - MAV6D: paired sequence bootstrap, B = 2,000, seed 0. Test has 10 sequences / 4,800 frames; val has 16 sequences / 2,336 frames.
  - AntiUAV3D: paired circular moving-block bootstrap within each sequence, B = 2,000, seed 0. Block 176 on test (2,624 frames: 2,576 + 48). Block 53 on val (1,148 frames: `test_seq_0003` 831 + `test_seq_0005` 317).
  - AP is recomputed on every resample.
- Paths are relative to `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/`.

---

## 1. PST − RGB-only teachers on MAV6D, pooled over 3 seed-run pairs

**Arms (deployed, fused):**

| Pair | PST run | RGB-only run | test UDS PST / RGB | val UDS PST / RGB |
|---|---|---|---|---|
| 1 (shift 0, reported) | `srcx/in2mav` | `srcx/in2mav_none` | 60.81 / 57.38 | 59.71 / 53.52 |
| 2 (shift 10) | `review1113/sup3/run` | `review1113/seeds/mav_none_s10` | 59.28 / 58.02 | 59.54 / 51.44 |
| 3 (shift 20) | `review1113/seeds/mav_full_s20` | `review1113/seeds/mav_none_s20` | 62.70 / 58.85 | 59.87 / 50.87 |
| **3-run mean** | | | **60.93 / 58.08** | **59.71 / 51.95** |

**Pooling.** In each draw, sequences are resampled once. ΔUDS is computed for each of the 3 pairs on the same frames, and the 3 values are averaged.

**ΔUDS, PST − RGB-only.** Point [95 % CI]; seqs = sequences where PST has the higher UDS.

| | test | seqs | val | seqs |
|---|---|---|---|---|
| pair 1 | +3.43 [+1.92, +5.51] | 9/10 | +6.19 [+4.53, +8.93] | 16/16 |
| pair 2 | +1.26 [−3.24, +7.14] | 6/10 | +8.10 [+5.14, +11.97] | 16/16 |
| pair 3 | +3.85 [+1.39, +6.68] | 9/10 | +8.99 [+5.76, +13.14] | 15/16 |
| **pooled (mean of 3)** | **+2.85 [+0.61, +5.68]** | 9/10 | **+7.76 [+5.98, +10.43]** | 16/16 |

The fraction of draws with pooled Δ ≤ 0 is 0.007 on test and 0.000 on val.

**Pooled components:**

| | test | val |
|---|---|---|
| ΔAP | +1.83 [−1.34, +6.07] | +8.88 [+5.83, +13.52] |
| ΔATE (m) | −0.033 [−0.068, −0.007] | −0.107 [−0.145, −0.076] |
| ΔAOE (°) | −1.83 [−3.86, −0.37] | −2.09 [−3.93, −0.70] |

**Notes.**
- Pair 1 reproduces `clean3/CLEAN3_REPORT.md` §2 exactly (+3.43 [+1.92, +5.51]).
- Pair 2's test interval crosses 0, because PST run 2 is the weakest PST run (59.28).
- These intervals cover sequence resampling given the 3 runs. They do not add the seed variance as a separate term.

---

## 2. IR⇄LiDAR check off (students on all frames) − with check, clean chain

**Source.** `review1113/sup2/nogate/`, run by `review1113/sup2/nogate_commands.sh` with FULL = 0 and MIRVIEW = 1.
- Log: `sup2/nogate/logs/nogate.log`, which ends with "ALL DONE: test UDS 62.07 … | val UDS 59.50".
- Predictions: `sup2/nogate/preds/nogate_fused_{val,test}.pkl`. Re-scored here: test **62.067**, val **59.503**.

**What differs between the arms.**
- Only the orientation students differ. They are retrained with `--tgt all`: 10,293 frames, an 80 % bag of all 12,866 pseudo-oriented crops, instead of the gated 4,301.
- Everything else is the same: mirror loss 12, 2,250 steps, init kzm_s{0,1}, followed by AdaBN.
- Detectors (det1), depth refiners (s5, s6), c_k, r_k and s̄ are frozen copies from `srcx/in2mav`. ω = 0.190 is frozen.
- b_k is re-estimated from unlabeled val images by the deploy rule. The fusion's corner-PnP distance changes because the rotations change.

**Seed counts.**
- **Both arms: 2 orientation-student seeds (0 and 1), deployed as one 2-student ensemble with the mirror view.** These are `sup2/nogate/ou/runs/NOGATE_s{0,1}` and `srcx/in2mav/ou/runs/S6I_s{0,1}`.
- Both arms also share the same 3 detector seeds and refiner seeds 5 and 6.
- This is **one run per arm**, not 2 independent runs. The paper's "(2 seeds)" means 2 student seeds inside one ensemble.

**Results, all frames − check.**

| | test | val |
|---|---|---|
| ΔUDS | **+1.26 [+0.45, +2.09]** (9/10 seqs) | **−0.21 [−0.95, +0.49]** (7/16 seqs) |
| ΔAP | +2.24 [+0.88, +3.65] | −0.25 [−1.51, +1.08] |
| ΔATE (m) | −0.017 [−0.026, −0.009] | +0.003 [−0.004, +0.009] |
| ΔAOE (°) | +0.33 [−0.65, +1.24] | +0.21 [−1.00, +1.66] |

**Reading.**
- On test, training on all frames is significantly better, and the gain comes through distance (AP, ATE) from the fusion's corner depth. Orientation is not better: AOE is 14.81° vs 14.48°.
- On val the difference is null.
- The +1.26 test gain lies inside the PST seed-run spread of item 1 (test 59.28–62.70).
- The supplement's wording, "check gives no gain; extra frames mainly improve corner depth", is consistent with this. A main-text claim that the check is a "conservative filter, not a gain" is fair on val. On test the check costs 1.26 UDS, and that cost excludes 0 under sequence resampling.

---

## 3. Fused vs IR-only depth-student targets: provenance confirmed

**Source.**
- Numbers: `review1113/clean3/CLEAN3_REPORT.md` item 1, from `clean3/boot12.py` → `clean3/boot12.json`.
- Values: ΔUDS test +1.44 [+0.79, +2.28], val +0.95 [+0.42, +1.79]. The arms are `srcx/in2mav/preds/pcmd_fused_*` (60.81 / 59.71) and `review1113/clean2/w0/preds/w0_fused_*` (59.37 / 58.77).
- The paper text matches: `sec/4_experiments.tex` line 30 and `sec/X_suppl.tex` line 273.

**Seeds.**
- Stage-3 depth refiners: seeds 5 and 6 in both arms (`clean2/w0/ref/ref_st3_s{5,6}.pt`).
- `w0/logs/infer.log` shows "avg_inf … from 2 files" for train, val and test, so the two seeds are averaged.
- This is one retrained control, with 2 refiner seeds averaged.

**Control.** The control is the reported chain (`in2mav`) with only the stage-3 targets changed (`chain_x.py targets 0`). MD5 checks:
- **Identical** between in2mav and w0:
  - `recs/det0.pkl` and `recs/det1.pkl` (detectors);
  - `ou/read_{tr1,val1,te1}_stu.pkl` (student rotations);
  - `kps_{tr1,val1,te1}.pkl` (height readings);
  - `ref/ref_st2_s{5,6}.pt` (stage-3 init);
  - `ref/inf_st2_det1_train.pkl`, `ref/real_d0_meta.pkl`, `ref/sim_meta.pkl`.
- **Different, as expected:**
  - `ref/dp_fused.npy`: same 4,751 finite crops, mean |Δδ| = 0.026;
  - `ref/ref_st3_s{5,6}.pt`.
- `targets_info.json` is identical apart from w (0 vs 0.190): same colour centres and b_train, 4,751 targets.

**Deploy constants** (`deploy_pcmd.json` vs `deploy_w0.json`):
- **Identical:** ω = 0.190282 in both fused arms, α = 0.5, and s̄ = (0.3478, 0.3485, 0.1955).
- **Re-derived by the frozen label-free rule from the new refiner outputs:**
  - colour centres (56.29, 156.96) → (57.09, 157.26);
  - c_k {0: 0.0615, 1: 0.0907} → {0: 0.0551, 1: 0.0957};
  - b_k {0: 0.528, 1: 0.318} → {0: 0.533, 1: 0.326};
  - r_k class 0: 0.37934 → 0.37948, via the class assignment. Class 1 is unchanged.
- These constants depend on the refiner's silhouette brightness and corrections, so re-deriving them is part of "changing stage 3", not a separate change.
- Rotations and detections are the same: AOE and mDR are equal in both arms (CLEAN3).

---

## 4. Single model (no TTA) vs ensemble, validation (and test)

**Sources.**
- Single-model predictions: MAV6D `review1113/clean/single/preds/single_fused_*.pkl`; AntiUAV3D `review1113/M12/anti/single/preds/single_fused_*.pkl`.
- Ensemble predictions: `srcx/in2mav` and `srcx/out2antifs` `pcmd_fused_*`.
- AntiUAV3D ensemble val 34.98 is also in `srcx/out2antifs/logs/run_x.log` ("val sanity: UDS 34.98199…").
- AntiUAV3D single val 33.44 was computed here from the saved predictions, and equals the stored `.score_val.json`.

| Dataset | Split | Single UDS | Ensemble UDS | Single − ensemble [95 % CI] | Bootstrap |
|---|---|---|---|---|---|
| MAV6D | **val** | 57.73 | 59.71 | **−1.98 [−3.21, −0.76]** (single better in 2/16 seqs) | sequence |
| MAV6D | test | 58.82 | 60.81 | −1.99 [−3.30, −1.39] (0/10 seqs) | sequence |
| AntiUAV3D | **val** | 33.44 | 34.98 | **−1.55 [−4.78, +1.26]** | block 53 |
| AntiUAV3D | test | 42.85 | 41.73 | +1.12 [−0.46, +2.69] | block 176 |

**Component rows.**
- AntiUAV3D val, single / ensemble: AP 18.14 / 20.30; ATE 10.2 / 9.7 %; AOE 20.58 / 20.83°; mDR 71.6 / 73.8 %.
- MAV6D val: AP 41.30 / 43.69; ATE 0.352 / 0.320 m; AOE 22.48 / 21.11°.

**Reading.** Validation favours the ensemble on both datasets, so choosing it on validation is justified.
- On MAV6D the preference is significant and agrees with test.
- On AntiUAV3D val (2 sequences, within-sequence resampling only) the direction favours the ensemble, but the CI crosses 0. The test reversal (+1.12) also crosses 0.

---

## 5. AntiUAV3D (`srcx/out2antifs`): refined distance vs the detector's distance

**Feasible on CPU.** All components were saved.

**Rows.** (c) and (c′) are rebuilt exactly as in `clean2/buildup/buildup.py` `build_c`, on out2antifs.
- (b) det1 as is: `recs/det1.pkl`, with the detector's rotation.
- (c) det1 box (the detector's own distance and s̄ size) with the rotation replaced by the student ensemble (S_ab0 + S_ab1 + mirror, `chain_x.zL_table`). The students read every frame: 2,624 / 1,148.
- (c′) (c) plus the deploy height r_k·(l+w)/2, with depth unchanged.
- (d) `preds/pcmd_nofuse`: the depth students' refined distance plus size coupling, no fusion. Orientation is identical to (c).
- (e) `preds/pcmd_fused`, the deployed PST.

| Split | Row | AP | ATE (rel.) | ASE (cm) | AOE (°) | mDR (%) | UDS |
|---|---|---|---|---|---|---|---|
| test | (b) det1 | 23.49 | 11.69 % | 15.24 | 22.05 | 91.9 | 37.58 |
| test | (c) + students, detector distance | 23.49 | 11.69 % | 15.24 | 17.11 | 91.9 | 38.66 |
| test | (c′) (c) + r_k height | 23.49 | 11.69 % | 15.56 | 17.11 | 91.9 | 38.66 |
| test | **(d) + depth students, no fusion** | 27.83 | 9.96 % | 16.30 | 17.11 | 91.9 | **41.69** |
| test | (e) PST fused | 27.89 | 9.93 % | 16.30 | 17.11 | 91.9 | 41.73 |
| val | (b) det1 | 16.93 | 11.58 % | 14.93 | 26.03 | 73.8 | 31.22 |
| val | (c) + students, detector distance | 16.93 | 11.58 % | 14.93 | 20.83 | 73.8 | 32.36 |
| val | (c′) | 16.93 | 11.58 % | 15.26 | 20.83 | 73.8 | 32.36 |
| val | **(d)** | 19.34 | 9.80 % | 15.89 | 20.83 | 73.8 | **34.45** |
| val | (e) PST fused | 20.30 | 9.69 % | 15.89 | 20.83 | 73.8 | 34.98 |

**Paired differences.** Point [95 % CI], block bootstrap.

| Step | test ΔUDS | test ΔATE (rel. pts) | val ΔUDS | val ΔATE (rel. pts) |
|---|---|---|---|---|
| **(d) − (c): refined distance alone vs detector distance** | **+3.04 [+0.72, +5.74]** | **−1.73 [−3.0, −0.5]** | **+2.09 [−0.14, +4.77]** | **−1.77 [−3.3, −0.4]** |
| ΔAP within (d) − (c) | +4.34 [+0.96, +8.53] | | +2.41 [−0.69, +6.47] | |
| (c) − (b): orientation students | +1.08 [+0.79, +1.38] (AOE −4.95°) | 0 | +1.14 [+0.78, +1.49] (AOE −5.20°) | 0 |
| (e) − (d): test-time fusion | +0.04 [−0.61, +0.46] | −0.03 [−0.2, +0.2] | +0.53 [+0.03, +1.09] | −0.11 [−0.3, +0.1] |

**Notes.**
- (d) − (c′) is identical to (d) − (c) in UDS, AP and ATE.
- On AntiUAV3D, ASE (15–16 cm) is above the 10 cm cap, so the size term of UDS is saturated. The r_k height (ASE +0.32 cm) and the size coupling in (d) (ASE +1.05 cm vs (c)) therefore have zero UDS effect. **The whole (d) − (c) UDS change is the refined distance.**
- On AntiUAV3D, most of the gain from distance comes from the depth students, not from fusion. On MAV6D test it is the opposite (CLEAN2: depth +0.89, CI crosses 0; fusion +4.14).
- The block bootstrap covers within-sequence variation only: 1 long M4E flight on test, 2 sequences on val (A3 caveat).
