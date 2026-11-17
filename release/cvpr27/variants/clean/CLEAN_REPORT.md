# Clean-protocol MAV6D chain (srcx/in2mav, test UDS 60.81): recomputed numbers on one protocol

2026-11-13. Everything here works from the predictions and intermediate files of the clean chain `srcx/run_x.sh in mav` (outputs `srcx/in2mav/`, log `srcx/in2mav/logs/run_x.log`).
- No training. No existing file was modified, and no process was killed.
- All outputs are in `review1113/clean/`.
- Item 1 needed GPU inference. It went through `gpu_slot.sh light` with `torch.cuda.set_per_process_memory_fraction(0.2)` (via `M12/memcap.py`) and batch size 8. Everything else ran CPU-only.
- Labels were read only by scorers: `srcx/lib/score_x.py`, and `cmpeval/paper_table.metrics`, which is the same metric code as pack `real_test`.
- Metric order: AP, ATE (m), ASE (cm), AOE (°, quadrotor-symmetric), mDR (%), UDS.

Reference rows, from the chain log and identical when re-scored here:

| MAV6D | AP | ATE | ASE | AOE | mDR | UDS |
|---|---|---|---|---|---|---|
| test, clean PST (run_x.log:442) | 40.93 | 0.306 | 1.75 | 14.48 | 96.17 | **60.81** |
| test, sim-only detector (run_x.log:438) | 26.87 | 0.283 | 1.74 | 20.32 | 69.77 | 49.51 |
| val, clean PST (run_x.log:429, sanity) | 43.69 | 0.320 | 2.30 | 21.11 | 94.74 | 59.71 |

---

## 1. Single model, no TTA (clean chain)

**Recipe.** M12's recipe, as used for AntiUAV3D in `M12/run_m1_anti.sh`, applied to `in2mav`. Script: `clean/run_single.sh`, with copies in `clean/lib`.

**Networks.**
- Detector: the self-trained detector `srcx/st_in2mav/DET1_s0` (epoch 3), without flip TTA.
- Depth refiner: phase-3 seed 5 only (`ref_st3_s5.pt`).
- Orientation student: `S_ab0` only, without the mirror view.
- Height network: unchanged (KPSs0). It is one network in the full chain too.
- ω = 0.190 is frozen from the full chain's `w_sim.json`, as in M12.

**Constants.** All were recomputed by the unchanged `chain_x.py deploy` rule from this model's own unlabeled images:
- s̄ (val images, this detector, conf ≥ 0.7: 519 boxes) = (0.3474, 0.3483, 0.1953);
- brightness centres 57.5 / 157.3;
- c_k = 0.050 / 0.086 (−1: 0.070);
- r_k = 0.378 / 0.702;
- β′_k on val images = 0.511 / 0.292.

**Reused readings.**
- As in M12, the height net's BN is re-estimated on the full chain's train crops (det1 ensemble windows, `tr1`). Val/test crops, readings and refiner outputs (train/val/test) were recomputed on the single detector's windows.
- The re-run tr1 height median is 0.447 vs 0.451 in the full chain. The BN statistics differ slightly with batch size 8 vs 16.

**Scoring.** Test was scored once.

| MAV6D | AP | ATE (m) | ASE (cm) | AOE (°) | mDR (%) | UDS |
|---|---|---|---|---|---|---|
| test, clean PST full (3 det + flip, 2 refiners, 2 students + mirror) | 40.93 | 0.306 | 1.75 | 14.48 | 96.17 | 60.81 |
| **test, clean PST single, no TTA** | **38.21** | **0.325** | **1.75** | **15.20** | **94.23** | **58.82** |
| test, clean single, ω = 0 (reference) | 32.30 | 0.368 | 1.75 | 15.21 | 94.31 | 55.34 |
| val, clean full (sanity) | 43.69 | 0.320 | 2.30 | 21.11 | 94.74 | 59.71 |
| val, clean single (sanity) | 41.30 | 0.352 | 2.30 | 22.48 | 94.14 | 57.73 |
| (70.16 run: full / single, test) | 56.44 / 54.44 | 0.226 / 0.236 | 1.81 / 1.79 | 13.54 / 14.39 | 99.8 / 99.75 | 70.16 / 68.86 |

**Cost of slimming.**
- Clean chain: −1.99 test UDS and −1.98 val UDS.
- 70.16 run: −1.30 test UDS.

**Fusion and frame counts.**
- Single-model fusion gain: +3.48 test, +0.32 val.
- Fused frames: test 4,741 / 4,800, val 2,275 / 2,336. By the deploy rule, a frame is not fused if it has no corner reading or its class has no β′_k; I did not split the count between those two cases.

**Outputs.** `single/deploy_single.json` and `single/preds/single_{fused,nofuse}_{val,test}.pkl(.score_*.json)`.

---

## 2. Test-time fusion gain of the two distances (clean chain)

The chain's log scores fused val (:429), nofuse val (:433) and fused test (:442), but **it never scored nofuse test**: `pcmd_nofuse_test.pkl` exists, but run_x.sh only scores `srconly_test` and `pcmd_fused_test`. I scored it here once with the same `score_x.py` (`clean/fusion/`). The other three re-scores are identical to the log.

| Split | Fusion | AP | ATE | ASE | AOE | mDR | UDS |
|---|---|---|---|---|---|---|---|
| test | with fusion (ω = 0.190) | 40.93 | 0.306 | 1.75 | 14.48 | 96.17 | 60.81 |
| test | without (ω = 0) | 33.74 | 0.350 | 1.75 | 14.48 | 96.19 | 56.67 |
| val | with fusion | 43.69 | 0.320 | 2.30 | 21.11 | 94.74 | 59.71 |
| val | without | 41.42 | 0.327 | 2.30 | 21.11 | 94.82 | 58.51 |
| **gain** | test / val | +7.19 / +2.27 AP | | | | | **+4.14 / +1.21** |

The 70.16 run gave +1.90 / +0.98. On test the clean chain gains more: its distances before fusion carry a +6.6 % scale bias, and the corner distance pulls the median back by −2.6 % (item 4).

Fused frames: test 4,800 / 4,800 and val 2,333 / 2,336. The 3 val frames are class −1, which has no b_k.

---

## 3. LiDAR height ratio vs alternatives (clean chain)

**Setup.**
- Only h changes: h := r · (l+w)/2. Length, width, position, rotation and confidence are unchanged.
- In the srcx chain, corner PnP uses the s̄ box (`oc_*.pkl['size']`) and never r_k. So this post-hoc swap is the whole effect of the height source, which is variant (a) of `review1113/height`. AP is therefore identical in every row.

**Arms.**
- **PST r_k = 0.379 (dark) / 0.705 (bright).** This is the median height-network reading (KPSs0, BN re-estimated on train crops) per brightness class on unlabeled train (`deploy_pcmd.json`). Test has no class −1 frames; val has 3, and those get their per-frame reading.
- **s̄ constant aspect = 0.1955 / mean(0.3478, 0.3485) = 0.561.** This uses the chain's own s̄.
- **Sim-box constant = 0.400.** This is the mean h/((l+w)/2) over all indoor-sim train boxes (0.3999, `A3/b_sim_aspect.json`).

**Method.**
- Per-frame quantities come from `A3/b_swap.frame_q` / `stats`, asserted equal to `paper_table.metrics` to 1e-9.
- |Δh| is the mean over the scorer's TP set: top box, 2D IoU ≥ 0.1.
- CIs use a sequence bootstrap: B = 2,000, seed 0, both arms on the same resampled sequences, 10 test / 16 val sequences.
- Script: `clean/height/height_clean.py`, which writes `height_clean.json`.

**Test (10 sequences)**

| Height source | r dark / bright | UDS | ASE (cm) | mean \|Δh\| cm (% GT h) | Mavic 2 \|Δh\| cm (%) | Mavic 2 ASE | Phantom 4 \|Δh\| cm (%) | Phantom 4 ASE |
|---|---|---|---|---|---|---|---|---|
| **PST: LiDAR height ratio r_k** | 0.379 / 0.705 | **60.81** | **1.75** | **1.62 (8.2)** | **1.25 (10.4)** | **3.18** | **1.73 (7.5)** | **1.30** |
| s̄ constant aspect | 0.561 | 59.65 | 2.68 | 4.39 (26.4) | 7.61 (63.4) | 5.30 | 3.36 (14.6) | 1.84 |
| sim-box constant | 0.400 | 58.44 | 3.65 | 7.31 (33.7) | 1.97 (16.4) | 3.42 | 9.01 (39.2) | 3.73 |

**Val (16 sequences)**

| Height source | r dark / bright | UDS | ASE (cm) | mean \|Δh\| cm (% GT h) | Mavic 2 \|Δh\| cm (%) | Mavic 2 ASE | Phantom 4 \|Δh\| cm (%) | Phantom 4 ASE |
|---|---|---|---|---|---|---|---|---|
| **PST: LiDAR r_k** | 0.379 / 0.705 | **59.71** | **2.30** | **1.58 (9.7)** | **1.36 (11.3)** | **3.20** | **1.82 (7.9)** | **1.34** |
| s̄ constant aspect | 0.561 | 58.04 | 3.62 | 5.57 (39.8) | 7.62 (63.5) | 5.28 | 3.39 (14.7) | 1.86 |
| sim-box constant | 0.400 | 58.14 | 3.57 | 5.41 (27.6) | 1.98 (16.5) | 3.40 | 9.03 (39.3) | 3.74 |

**Differences (alternative − r_k), mean of the bootstrap with its 95 % interval**

| Split | Height source | Group (seqs) | ΔUDS | ΔASE (cm) | Δ mean \|Δh\| (cm) | Δ mean \|Δh\| (% GT h) |
|---|---|---|---|---|---|---|
| test | s̄ constant | all (10) | −1.17 [−1.79, −0.71] | +0.93 [+0.57, +1.42] | +2.79 [+1.70, +4.27] | +18.4 [+7.8, +32.8] |
| test | s̄ constant | Mavic 2 (3) | −2.66 [−2.68, −2.62] | +2.12 [+2.10, +2.13] | +6.36 [+6.29, +6.38] | +53.0 [+52.4, +53.2] |
| test | s̄ constant | Phantom 4 (7) | −0.68 [−0.79, −0.58] | +0.55 [+0.47, +0.63] | +1.64 [+1.40, +1.90] | +7.1 [+6.1, +8.3] |
| test | sim-box 0.400 | all (10) | −2.36 [−2.99, −1.51] | +1.89 [+1.21, +2.39] | +5.66 [+3.63, +7.17] | +25.3 [+17.4, +31.2] |
| test | sim-box 0.400 | Mavic 2 (3) | −0.30 [−0.30, −0.30] | +0.24 [+0.24, +0.24] | +0.72 [+0.71, +0.72] | +6.0 [+5.9, +6.0] |
| test | sim-box 0.400 | Phantom 4 (7) | −3.03 [−3.13, −2.95] | +2.43 [+2.36, +2.51] | +7.28 [+7.07, +7.52] | +31.7 [+30.8, +32.7] |
| val | s̄ constant | all (16) | −1.66 [−2.21, −1.04] | +1.31 [+0.83, +1.75] | +3.93 [+2.47, +5.23] | +29.7 [+15.7, +42.2] |
| val | s̄ constant | Mavic 2 (7) | −2.64 [−2.74, −2.55] | +2.09 [+2.02, +2.12] | +6.26 [+6.07, +6.36] | +52.1 [+50.6, +53.0] |
| val | s̄ constant | Phantom 4 (9) | −0.65 [−0.84, −0.40] | +0.52 [+0.32, +0.67] | +1.55 [+0.96, +2.02] | +6.8 [+4.2, +8.8] |
| val | sim-box 0.400 | all (16) | −1.60 [−2.45, −0.84] | +1.30 [+0.68, +1.98] | +3.89 [+2.05, +5.93] | +18.2 [+10.8, +26.2] |
| val | sim-box 0.400 | Mavic 2 (7) | −0.23 [−0.30, −0.11] | +0.20 [+0.14, +0.24] | +0.61 [+0.42, +0.71] | +5.1 [+3.5, +5.9] |
| val | sim-box 0.400 | Phantom 4 (9) | −2.98 [−3.16, −2.75] | +2.40 [+2.20, +2.54] | +7.19 [+6.61, +7.62] | +31.3 [+28.7, +33.1] |

- Every overall interval excludes 0, on both splits.
- Per aircraft, each constant fits at most one airframe:
  - s̄ (0.561 ≈ the sim Phantom 4 box) over-estimates Mavic 2 height by about 63 %.
  - 0.400 (≈ the Mavic 2 aspect 0.389) under-estimates Phantom 4 height by about 39 %.
- The Mavic 2 test intervals rest on only 3 sequence clusters, so they are optimistic.
- The conclusion matches the 70.16 run (`A3_REPORT.md` B3). The clean test effect is slightly larger: ΔUDS is −1.17 here vs −1.03 there for s̄, and −2.36 vs −2.25 for 0.400.
- Diagnostic only: the brightness class matches the airframe on 86.8 % / 99.1 % (test, Mavic 2 / Phantom 4) and 89.7 % / 97.6 % (val).

---

## 4. Why is the clean test ATE (0.306 m) worse than sim-only (0.283 m)?

**Diagnostic only, not a deployable fix.**
- e = log(z_pred/z_gt) is computed on each arm's own TP set.
- Bias b = median e. Spread = 1.4826 · MAD(e − b).
- "After −b" rescales every predicted position by e^−b, using the arm's own global b on that split, and re-scores everything (AP too).
- Script: `clean/diag/ate_diag.py`, which writes `ate_diag.json`.

**Arms.**
- Clean stages: det0 = sim-only detector (3 seeds + flip, `srconly`), then det1 = self-trained detector ensemble (`recs/det1.pkl`), then + refiner (`pcmd_nofuse`), then + corner fusion (`pcmd_fused`).
- 70.16 run: raw nl075 ensemble (`nl075_ens/ens_*.pkl`), then w = 0, then fused (`M12/M1/repro_full`, reproduced exactly).

### 4a. Bias / spread and de-biased scores, test (4,800 frames)
| Arm | UDS | AP | ATE | mDR | TPs | bias b | spread | bias Mavic 2 / Phantom 4 | bias dark / bright class | ATE after −b | AP after −b | UDS after −b |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean: sim-only (det0) | 49.51 | 26.87 | 0.283 | 69.8 | 3349 | −3.3 % | 9.7 % | +8.8 / −3.5 % | +6.9 / −3.5 % | 0.275 | 28.00 | 50.19 |
| clean: self-trained detector (det1) | 53.38 | 34.04 | 0.356 | 96.2 | 4619 | +5.9 % | 9.5 % | +7.8 / +5.3 % | +7.5 / +5.4 % | 0.288 | 42.31 | 58.37 |
| clean: + depth refiner (ω = 0) | 56.67 | 33.74 | 0.350 | 96.2 | 4617 | +6.6 % | 8.1 % | +8.5 / +6.1 % | +8.2 / +6.1 % | 0.255 | 45.94 | 63.96 |
| **clean: PST (+ fusion)** | **60.81** | 40.93 | **0.306** | 96.2 | 4616 | **+4.6 %** | 7.7 % | +7.5 / +3.9 % | +7.0 / +3.9 % | **0.252** | 48.19 | **65.11** |
| 70.16: self-trained detector (raw) | 59.93 | 46.13 | 0.276 | 99.6 | 4780 | +2.9 % | 8.3 % | +6.2 / +1.9 % | +6.3 / +1.9 % | 0.260 | 48.57 | 61.35 |
| 70.16: + refiner (w = 0) | 68.26 | 52.97 | 0.239 | 99.8 | 4790 | +2.6 % | 6.1 % | +7.0 / +1.8 % | +7.2 / +1.8 % | 0.221 | 56.39 | 70.18 |
| 70.16: PST | 70.16 | 56.44 | 0.226 | 99.8 | 4791 | +1.4 % | 5.8 % | +6.5 / +0.4 % | +6.5 / +0.4 % | 0.221 | 57.15 | 70.58 |

Val (2,336 frames), same columns:

| Arm | UDS | ATE | TPs | bias b | spread | bias Mavic 2 / Phantom 4 | ATE after −b | UDS after −b |
|---|---|---|---|---|---|---|---|---|
| clean: sim-only | 35.53 | 0.368 | 1034 | −6.9 % | 8.8 % | −3.1 / −7.2 % | 0.305 | 37.90 |
| clean: det1 | 51.29 | 0.372 | 2215 | +1.8 % | 10.2 % | +1.8 / +1.8 % | 0.363 | 51.88 |
| clean: + refiner | 58.51 | 0.327 | 2215 | +2.0 % | 8.7 % | +2.3 / +1.8 % | 0.313 | 59.73 |
| clean: PST | 59.71 | 0.320 | 2213 | +1.7 % | 8.3 % | +2.4 / +1.1 % | 0.309 | 60.60 |
| 70.16: PST | 62.49 | 0.312 | 2336 | −0.2 % | 7.4 % | −0.0 / −0.2 % | 0.312 | 62.50 |

### 4b. Selection: ATE is averaged over different frames
ATE averages only over each arm's own TP frames. Sim-only detects 69.8 % of test frames, PST 96.2 %.

| Test arm | ATE on the 3,349 frames that are TP for sim-only | bias there | spread there | extra TP frames | ATE on those extra frames |
|---|---|---|---|---|---|
| sim-only | 0.283 | −3.3 % | 9.7 % | – | – |
| clean det1 | 0.306 | +5.5 % | 8.8 % | 1,270 | 0.487 |
| clean + refiner | 0.312 | +6.1 % | 7.3 % | 1,268 | 0.449 |
| **clean PST** | **0.251** | +3.8 % | 6.7 % | 1,267 | 0.451 |
| 70.16 PST | 0.148 | +0.5 % | 4.6 % | 1,442 | 0.406 |

On the same frames, clean PST is **better** than sim-only: 0.251 vs 0.283 m, with per-frame spread 6.7 % vs 9.7 %. The headline 0.306 > 0.283 has two sources:
- PST adds 1,267 hard frames that sim-only misses. Their ATE is 0.451.
- A **+4.6 % global over-estimate of distance** remains. Removing it gives ATE 0.252 and UDS 65.11 (+4.3), against sim-only 0.275 / 50.19 after its own de-biasing. The 70.16 run gains only +0.42 from the same oracle, because its bias is +1.4 %.

### 4c. Where the bias comes from (median log-ratio between consecutive stages, test frames TP in clean PST)
| Stage | clean (all / Mavic 2 / Phantom 4) | 70.16 run |
|---|---|---|
| self-training: det1 / det0 | **+9.8 %** / +0.0 % / **+10.4 %** | +6.1 % / −0.5 % / +6.9 % |
| refiner applied, = δ̂ − c_k | +1.3 % / +0.9 % / +1.3 % | −0.2 % |
| corner fusion: fused / nofuse | −2.6 % / −1.3 % / −2.8 % | −1.5 % |

**Refiner.**
- Raw δ̂ on test has median 0.098 (dark class 0.070, bright 0.104). The train constants are c_k = 0.062 / 0.091, so the refiner adds only +1.3 %.
- ω = 0.190 and the corner distance **reduce** the bias. The corner estimate log z_L − b_k is itself −6.6 % on test (Phantom 4 −7.7 %), with spread 11.2 %. Its correlation with the refined error is 0.28.
- So neither the refiner correction (c̄, c_k) nor ω/b_k causes the over-estimate.

**Self-trained detector.** It copies its pseudo labels. On train, label-free:
- pseudo-label / det0 = +9.75 %;
- det1 / det0 = +9.38 %;
- det1 / pseudo label = −0.05 %.

**Pseudo labels.**
- The pseudo labels are t_det0 · e^(δ̂ − λc̄), with λ = 0.75 and c̄ = 0.099.
- c̄ is the median of δ̂ over **all** 12,865 train crops.
- The labels themselves are kept only for frames with det0 confidence ≥ 0.5: 5,227 of 12,866 (40.6 %). On those frames the stage-2 refiner's median δ̂ is **0.172**; on the dropped frames it is 0.095.
- So the kept labels retain 0.172 − 0.074 ≈ +9.8 % instead of the intended ≈ 0.25·δ̂.
- In the development chain 95 % of frames were kept (12,257 / 12,866), so c̄ (0.126) matched the kept set.

**Val vs test.**
- On val, det0 is −6.9 % biased, so the same +9.7 % shift lands at +1.8 % and PST improves ATE (0.320 vs 0.368).
- On test, det0 Phantom 4 is only −3.5 % biased, so the shift overshoots to +5.3 %.
- The global scale is not identified without labels (paper Sec. "Global scale").

**One-sentence cause:** the clean chain's pseudo boxes keep only the 40.6 % confident frames, but λc̄ is computed over all train frames, so on the kept frames the refiner's offset is under-removed (median δ̂ 0.172 vs c̄ 0.099). The self-trained detector copies the resulting +9.8 % depth inflation, which on test (where the sim-only detector was only −3.3 % short) becomes a +4.6 % global over-estimate. Together with the 1,267 hard extra detections, this makes ATE 0.306 > 0.283, even though on the same frames PST is better (0.251 vs 0.283 m).

---

## 5. Lineage of the clean chain (format of Tab. lineage, `X_suppl.tex`)

None of these networks is a development network. The chain uses **no bootstrap or initial network**: no first-round detector, no crop detector, no first-generation refiners, and no development students. It starts from the simulation-only detectors and uses only single-frame material.

Sources: `run_x.sh`, `lib/sx.py` SOURCES['in'] / TARGETS['mav'], `in2mav/logs/*`, `in2mav/ref/train_st*.log`, `deploy_pcmd.json`, `targets_info.json`, `w_sim.json`, `pseudo_srcx_in2mav.json`.

| Network | Initialised from | Trained on (data / labels) | Seeds | Cross-frame steps |
|---|---|---|---|---|
| simulation-only detector (×3) | ImageNet (ResNet-34) | indoor sim (`E:/mmcache/full_v2`), sim labels; recipe `sim_full_r34_geo_tr_mix_sym8_mblur_ramp`, 5 epochs (= `release_20260920/ramp_s{0,1,2}`) | 0, 1, 2 | none |
| depth refiner phase 1 (×2) | ImageNet ResNet-18 (DCNetU) | 8,965 sim samples (ramp_s0 on indoor-sim train, every 2nd frame, matched within 25 % of distance; δ* = log z_gt/z_det; IR silhouettes 8,964) composited onto a real background bank of 4,976 windows (det0 detections, conf ≥ 0.5, width 6–170 input px); 30 epochs | 5, 6 | none |
| depth refiner phase 2 (×2) | phase 1, same seed | + 5,227 real crops of det0 (conf ≥ 0.5) without labels: EMA consistency, silhouette BCE, adversary; 2,000 steps | 5, 6 | none |
| self-trained detector (×3) | same-seed sim-only detector ramp_s{k} | 5,227 pseudo boxes (`index_pseudo_srcx_in2mav.pkl`): t_det0·e^(δ̂−λc̄) from det0 (3 seeds + flip) and phase-2 refiners (mean of s5, s6); size s̄_det0; rotation R_det0; 3 epochs, LR 2e-4, flips only, no rotation loss | 0, 1, 2 | none |
| teacher-shaped orientation nets (6) and height net | simulation (IR+LiDAR `kzm_s{0,1}`, IR-distilled `KZKDir_s{0,1}` ep 30, LiDAR `kzl_s{0,1}`; height net `KPSs0` ep 8) | indoor sim; only BN re-estimated on 3,695 real train crops (conf ≥ 0.5, ≤ 80 per sequence) cut at **det1** windows | 0, 1 (height: 0) | none |
| orientation students S6I (×2) | same-seed IR+LiDAR teacher `kzm_s{0,1}` (sim weights) | sim crops (background swap 0.5 from a 4,420-window single-frame real bank) + real det1 crops with gated pseudo orientations (IR-distilled ⇄ LiDAR ≤ 10°: 4,301 of 12,866 crops); random 80 % bag → 3,441 frames per student; mirror loss 12; 2,250 steps; then BN re-estimated on 8,261 real train crops | 0, 1 | none |
| final depth refiner, phase 3 (×2) | phase 2, same seed | the 5,227 det0 crops; fused pseudo targets on 4,751 (4,741 with a corner term): log(z_pl/z_det0) + ω(log z_L/z_pl − β_k); 2,000 steps; sim batches continue | 5, 6 | none |

**Counts asked for, with the development chain alongside**

| Step | clean chain | development chain | Source (clean) |
|---|---|---|---|
| (a) pseudo boxes: frames kept at conf ≥ 0.5 / train frames | **5,227 / 12,866** (40.6 %) | 12,257 / 12,866 | `pseudo_srcx_in2mav.json`; run_x.log:4 ("≥0.5 40.6 %") |
| (b) orientation self-training: crops / passing IR⇄LiDAR / used per student | **12,866 / 4,301 / 3,441** | 4,289 / 1,525 / 1,220 | run_x.log ("crops tr1: 12866 (train every 1)"; "gate … passes 4301 (33.4 %)"); `logs/ou_train_s{0,1}.log` DONE `n_tgt` 3441 |

Clean crops are every train frame (`ori_x.py crops … train 1 tr1`), not every third. The gate here is the IR-distilled ⇄ LiDAR check only; the development version also had a teacher ⇄ RGB check.

**(d) Orientation crops.**
- They are cut at the windows of **det1**: the clean chain's own self-trained detector ensemble (`st_in2mav/DET1_s{0,1,2}`, 3 seeds + flip, from `recs/det1.pkl`), on raw MAV6D frames undistorted on the fly.
- The teacher and height-net BN windows, the pseudo-orientation crops, the student training crops, the student BN and the val/test reading crops all come from det1.
- **No crop detector, bootstrap or initial network is involved.** The only pre-trained inputs are simulation-trained nets: ramp detectors, the 6 orientation teachers and the height net.

**(c) Constants**

| Constant | Value | Derivation | Source |
|---|---|---|---|
| λ | 0.75 | frozen method constant (set on MAV6D in development) | make_pseudo_x.py `LAM` |
| c̄ | 0.0992 | median phase-2 refiner δ̂ (mean of s5, s6) over det0 train crops with \|δ̂\| ≤ log 1.4 (all 12,865 crops, not only the kept ones) | pseudo_srcx_in2mav.json |
| γ (size coupling, α in code) | 0.5 | frozen | chain_x.py `ALPHA`; deploy_pcmd.json |
| ω | 0.1903 | σ²_IR/(σ²_IR+σ²_cor) on 1,344 sim-val boxes (ramp_s0, conf ≥ 0.5, matched): σ_IR = 0.0574 (phase-2 refiners s5+s6), σ_cor = 0.1185 (students + mirror, corner PnP with detected size); used for training targets and deployment | w_sim.json; run_x.log:363 |
| brightness centres | train targets 56.1 / 158.2; deployment 56.3 / 157.0 | 1-D two-means of silhouette mean grey on unlabeled train crops (phase-2 refiners on det1 for targets; phase-3 refiners on det1 for deployment) | targets_info.json `col_centres`; deploy_pcmd.json |
| β_k, training targets | 0.387 (dark) / 0.211 (bright) | median log(z_L/z_pl) on unlabeled train frames whose det1 box agrees with the pseudo box within 14.3 % of distance (5,216 frames) | targets_info.json `b_train` |
| β′_k, deployment | 0.528 (dark) / 0.318 (bright) | median log(z_L/z_refined) on unlabeled **val** images | deploy_pcmd.json `b_k_val_images` |
| c_k | 0.0615 (dark) / 0.0907 (bright) / 0.0781 (all, class −1) | median phase-3 δ̂ (mean of s5, s6) on unlabeled train det1 crops per class, \|δ̂\| ≤ log 1.4 | deploy_pcmd.json `c_k` |
| r_k | 0.379 (dark) / 0.705 (bright) | median height-net ratio on train crops per class (class −1: per-frame reading) | deploy_pcmd.json `r_k`; kps_tr1.pkl |
| s̄ (deployment) | (0.3478, 0.3485, 0.1955) m | median lwh of det1 (3 seeds + flip) boxes with conf ≥ 0.7 on unlabeled val images, every 2nd frame (470 boxes) | run_x.log:31; deploy_pcmd.json |
| s̄ (pseudo boxes, refiner) | (0.3474, 0.3475, 0.1954) m | the same rule on det0 (175 boxes) | run_x.log:3; index_pseudo calib_lwh |
| thresholds | conf ≥ 0.5 (pseudo boxes, refiner crops, teacher/student BN), conf ≥ 0.6 (orientation bank), conf ≥ 0.7 (s̄); guard \|δ̂\| ≤ log 1.4; IR⇄LiDAR gate 10°; agreement 0.5/3.5 = 14.3 % of distance; width 6–170 input px (raw/3.75); crop zoom 95/w2D in [1, 4]; sim match 25 % of distance; mirror loss 12; bag 0.8; BN cap 80 per sequence | | run_x.sh, sx.py, chain_x.py, refiner_x.py, ori_x.py, ou_train_x.py |

**MonoCD +PL conversion of the clean pseudo labels** (checked, not run):
- **Position and size.** `E:/mmcache/mav6d_cn_obj/train/index_pseudo_srcx_in2mav.pkl` (5,227 frames, tag `srcx_in2mav`, written 11-13 05:44). These are the same pseudo boxes that trained the self-trained detectors and define the refiner's phase-3 targets. The refiner's own phase-3 targets are `srcx/in2mav/ref/dp_fused.npy`, per det0 crop.
- **Orientation.** `D:/S2MUAV3D/srcx_work/in2mav/pseudo_tr1.npz` has 12,866 crops with fields seq/frame/R/gate…, and gate passes 4,301. These are the students' pseudo orientations.
- **Frame list.** `D:/cmpfull/real_train/labels.pkl` (12,866 frames; all 5,227 match).
- **Conversion is not a literal one-liner.** `cmpfull/make_pseudo_pcmd.py <tag> <out>` takes the position tag, so `srcx_in2mav` works. But its orientation file is **hard-coded** to the development `wf21/orient_uda/pseudo/pseudo_train_t6il.npz`. Running it unchanged would silently mix development orientations with clean positions, and it would also restrict frames to t6il's every-third-frame set.
- **Needed change.** A copy with one line changed (`T = np.load('D:/S2MUAV3D/srcx_work/in2mav/pseudo_tr1.npz', …)`). The output is the same MonoCD `labels.pkl` format.
- **Frames.** 5,227 (all position frames), or 2,126 with `--gate-only`. The development version had about 4,289 / 1,525 candidates.
- **Check.** Look at the printed detector-vs-teacher rotation angle (should be ≪ 90°). If it is not, apply the 4-fold representative swap of `make_pseudo_pcmd_out.py`.

---

## Files (all under `review1113/clean/`)
- `lib/`: inference-only copies of the current `srcx/lib`, made by `lib/make_copies.py`. The changes are: sx O/W redirected to `$CL_AO`; ori_x `_wf` fallback to the deployed crops; batch size 8; `M12_NOFLIP`.
- `run_single.sh`, `single/`: item 1, with `single/logs/`, `single/deploy_single.json` and `single/preds/*.score_*.json`.
- `fusion/`: item 2 re-scores, including the first scoring of `pcmd_nofuse_test`.
- `height/height_clean.py`, `height_clean.json`: item 3.
- `diag/ate_diag.py`, `ate_diag.json`: item 4.
- `make_tables.py`, `tables.md`: raw tables.
- **Test-label use.** Re-scores of existing chain predictions (fused / nofuse / sim-only), one scoring of `pcmd_nofuse_test`, the single-model fused and nofuse test predictions scored once each, and the item 3 and 4 diagnostics. No result was used to choose anything.
