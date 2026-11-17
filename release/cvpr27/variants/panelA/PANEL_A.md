# Panel A: pseudo-label quality, second depth, intervals, copying, facts, size prior (2026-11-14)

**Ground rules.**
- CPU only (`CUDA_VISIBLE_DEVICES=""`). Nothing was trained or newly inferred, and no process was killed.
- Inputs are existing predictions and intermediate files only. All writes are in `review1113/panelA/`.
- Labels are used only to evaluate. No number below was used to choose anything.
- Clean MAV6D chain = `srcx/in2mav` (PST, test UDS 60.81). RGB-only control = `srcx/in2mav_none`.
- Error e = log(z / z_gt). The chain scales t uniformly, so e is also the log range ratio.
- "med |rel|" = median |e^e − 1|. "Bias" = median e. "Spread" = 1.4826 · MAD(e − bias).
- MAV6D intervals are paired whole-sequence bootstraps (B = 2,000, seed 0; 10 test / 16 val sequences, or 61 train sequences, of which 48 have kept frames). AntiUAV3D intervals are circular moving-block bootstraps within each sequence (A3 code).

**Scripts → outputs.**

| Script | Writes | Content |
|---|---|---|
| `a1_a8_labels.py` | `a1_a8_labels.{json,log}` | A1, A8 |
| `a1b_spread.py` | `a1b_spread.{json,log}` | A1 bias vs spread |
| `a8b_inherit.py` | `a8b_inherit.{json,log}` | A8 inheritance regression |
| `a2_second_depth.py` | `a2_second_depth.{json,log}` | A2 |
| `a6_intervals.py` | `a6_intervals.{json,log}` | A6 |
| (A10) | `A10_FACTS.md` | A10 |
| `a11/a11_size.py` | `A11_SIZE.md`, `A11_size_sweep.{png,pdf}` | A11 |

Shared helpers are in `pa_common.py`. It reuses `clean2/buildup/buildup.py`, which is asserted equal to the pack `score.py`. Corner-PnP readings are cached in `cache/`.

**Reproduction checks (all asserted).**
- The pseudo-label reconstruction equals `index_pseudo_srcx_{in2mav,in2mav_none}.pkl`: same frames, max |Δlog z| < 1e-5.
- The recomputed 14.3 % gate gives exactly the NaN crops of `dp_fused.npy`.
- The deploy function gives exactly `pcmd_fused_val` for both chains.
- The A2 re-fusion with the deployed reader gives exactly `pcmd_fused_{val,test}`, with recomputed β′_k equal to `deploy_pcmd.json`.
- Every A6 row equals `S.metrics`.

---

## A1. Position pseudo-label quality, stage by stage (MAV6D train, 12,866 frames; GT for evaluation only)

The sim-only detectors (det0) are bit-identical in both chains (asserted). The two chains' pseudo labels therefore differ only through the stage-2 refiner δ̂ and its constant c̄.

### A1a. Stages on the 5,227 kept frames (det0 conf ≥ 0.5; same frames for both chains)

| Stage | PST: med \|rel\| / bias / spread | RGB-only: med \|rel\| / bias / spread |
|---|---|---|
| 0. raw det0 distance | 7.6 % / −5.3 % / 9.7 % | (same) |
| 1a. det0 · e^δ̂ (refiner, no offset removed) | 11.7 % / +10.8 % | 14.0 % / +12.9 % |
| **1. pseudo label** z_det0 · e^(δ̂ − λc̄) (λ = 0.75, c̄ over **all** crops) | **5.9 % / +3.4 % / 7.4 %** | **8.8 % / +7.5 % / 8.1 %** |
| 1′. counterfactual: c̄ over the kept crops | 5.2 % / −2.1 % / 7.4 % | 5.5 % / −0.6 % / 8.1 % |
| 2. fused stage-3 target (4,751 / 3,899 gate-kept crops) | 5.6 % / +3.9 % / 6.7 % | 9.3 % / +8.4 % / 7.9 % |
| ↳ pseudo label on the same gate-kept crops | 6.0 % / +3.5 % / 7.4 % | 9.2 % / +7.9 % / 8.3 % |
| 3a. student detector det1 | 5.6 % / +3.3 % / 7.3 % | 8.4 % / +7.0 % / 8.1 % |
| 3b. det1 + phase-3 refiner (deploy `base`, c_k) | 4.6 % / +2.4 % | 7.6 % / +6.8 % |
| 3c. + deployed fusion (ω, β′_k from val images) | 4.3 % / +0.7 % | 4.9 % / +1.8 % |
| *all 12,866 train frames:* det0 → det1 → 3b → 3c | 12.2 → 7.5 → 6.2 → 5.6 % (bias −6.1 → +3.5 → +2.7 → +1.6 %) | 12.2 → 10.3 → 8.7 → 7.0 % (bias −6.1 → +7.4 → +6.3 → +3.7 %) |

**The +4.6 % story, now with labels.**
- c̄ = 0.0992 is the median δ̂ over all 12,865 crops. On the kept frames, the median δ̂ is 0.172; on the dropped frames it is 0.095.
- So the labels keep 0.172 − 0.75 · 0.0992 = **+9.7 %** of offset instead of the intended (1 − λ) · 0.172 = +4.3 %.
- Against GT this turns det0's −5.3 % into a **+3.4 %** label bias. The student copies it: det1 is +3.3 % on its own training frames.
- On test, det0 is only −3.3 % short. The same shift therefore becomes the +4.6 % over-estimate of deployed PST (det1 +5.9 %, + refiner +6.6 %, + fusion +4.6 %; `clean/CLEAN_REPORT.md` §4a/4c).
- On val, det0 is −6.9 %, and PST ends at +1.7 %.
- With c̄ computed on the kept set (row 1′), the PST label bias would be −2.1 % and med |rel| 5.2 %.

**PST vs RGB-only labels.**
- PST labels are better: 5.9 % vs 8.8 % med |rel|.
- About two thirds of that gap is the same c̄ selection effect. RGB-only has a smaller c̄ (0.072) against a kept-set median of 0.180, so it retains **+12.6 %** offset vs +9.7 % for PST.
- The per-frame spread advantage is smaller but real: 7.4 % vs 8.1 % (and 6.7 % vs 7.9 % for the fused targets).
- Under the counterfactual (row 1′) the labels nearly tie: 5.2 % vs 5.5 %.

### A1b. The 14.3 % agreement gate (stage-3 refiner targets): crops it drops vs crops it keeps

| Chain | Crops | n | det0: med \|rel\| / bias | pseudo label: med \|rel\| / bias | median δ̂ | label better than det0 | det0 > 1 m from GT |
|---|---|---|---|---|---|---|---|
| PST | **dropped** | 476 (9.1 %) | 13.5 % / −13.8 % | **5.1 % / +2.4 %** | 0.239 | **83 %** | 3.8 % |
| PST | kept | 4,751 | 7.1 % / −4.5 % | 6.0 % / +3.5 % | 0.166 | 57 % | 1.7 % |
| RGB-only | dropped | 1,328 (25.4 %) | 10.8 % / −11.3 % | 7.5 % / +6.1 % | 0.226 | 64 % | 2.7 % |
| RGB-only | kept | 3,899 | 6.6 % / −3.2 % | 9.2 % / +7.9 % | 0.165 | 40 % | 1.6 % |

**Reading.**
- The gate compares the det0 box with the pseudo box. Both share the det0 position up to the depth factor, so the gate removes exactly the crops whose label correction is large (|δ̂ − λc̄| ≳ 13 %).
- Those are det0's worst under-estimates, and **the labels fix them best** (13.5 % → 5.1 % for PST). The gate is not removing bad labels; it removes the crops where the label disagrees most with det0.
- This should not be described as "dropping unreliable labels". The second gate (det1 box vs pseudo box, for the corner term) passes 5,216 / 5,227 PST frames, so it is nearly inactive.

### A1c. Mediator: does better label quality go with better students?

**Val labels for RGB-only do not exist** among the saved files. They would need its stage-2 refiners (`in2mav_none/ref/ref_st2_s{5,6}.pt`) run on det0's 835 val crops: one GPU inference pass, for example `N1clean/n1c_refiner_infer.py` with the chain path changed. It was not run, under the brief. So the mediator test is done:
- (i) in-sample on train, where both chains' labels exist;
- (ii) on held-out val for the student side, with PST's held-out val labels (N1clean reconstruction) as context.

Both use a paired sequence bootstrap. Differences are PST − RGB-only in med |rel| (percentage points).

| Frames | PL difference | Student difference | Across-sequence association (per-sequence ΔPL vs Δstudent) |
|---|---|---|---|
| train, kept 5,227 frames, 48 seqs: student = det1 | **−2.8 [−3.3, −2.3]** | **−2.8 [−3.5, −2.3]** | r = **0.81 [0.61, 0.93]**, slope 0.76 [0.51, 0.95]; 41/48 seqs better on both |
| train: student = det1 + phase-3 refiner | −2.8 | −3.0 [−3.7, −2.1] | r = 0.70 [0.43, 0.86], slope 0.82 [0.45, 1.10] |
| train: student = deployed (+ fusion) | −2.8 | −0.6 [−1.4, +0.1] | r = 0.63 [0.32, 0.81], slope 0.73 [0.31, 1.06]; 30/48 both better |
| **val (held out)**, 835 det0-conf frames, 14 seqs: det1 | (PST PL 5.5 % vs det0 8.1 %; RGB-only n/a) | **−2.4 [−3.6, −1.4]** | – |
| val: deployed (+ fusion) | | −0.7 [−2.1, −0.1] | – |

Frame level on train, sequence-demeaned: ΔPL vs Δdet1 gives r = 0.79 (slope 0.67). For the deployed output the same r is 0.33.

**Reading.**
- Yes, label quality and student quality move together. The detector student reproduces the label difference one-for-one in aggregate (−2.8 vs −2.8 points). Sequences with a larger label advantage also have a larger student advantage (r = 0.81).
- The held-out det1 gap on val (−2.4 points) is the same size as the train label gap.
- The deployed output keeps only a fraction of the distance gap (−0.6 train, −0.7 val), because test-time fusion pulls both chains toward the val-calibrated scale (A2).
- Caveat: much of the label difference is the c̄ bias (A1a). The mediation is therefore largely a **scale** mediation, plus a 7.4 vs 8.1 % spread difference.

---

## A2. Second depth from another reader: does the fusion gain need the LiDAR-shaped corner reader?

**Rule.**
- This is unchanged `chain_x.cmd_deploy`: log z = (1 − ω) log z_PST + ω (log z₂ − β′_k), with ω = 0.190 frozen.
- β′_k is the per-PST-colour-class median log(z₂ / z_PST) on unlabeled val images, re-estimated for each source exactly as the deploy rule does.
- z_PST is `pcmd_nofuse`; rotation, size and confidence are unchanged.
- Corner depths for other chains come from their own students on their own det1 crops (corner PnP with their own s̄).
- Fused frames: 2,333 / 2,336 val and 4,800 / 4,800 test for every source.

| Second depth z₂ | Test UDS | Test fusion gain [95 % CI] | Val fusion gain | corr(e₂, e_PST), test | spread of e₂, test |
|---|---|---|---|---|---|
| none (ω = 0) | 56.67 | – | – (58.51) | | |
| **PST students, deployed (seed 0)** | **60.81** | **+4.14 [+2.70, +5.71]** | +1.21 [−0.52, +3.20] | 0.11 | 0.106 |
| PST students, seed run +10 | 60.40 | +3.73 [+2.33, +5.32] | +1.14 [−0.65, +3.14] | 0.18 | 0.105 |
| PST students, seed run +20 | 60.87 | +4.21 [+2.50, +5.81] | +0.75 [−0.98, +2.83] | 0.14 | 0.104 |
| PST students without mirror loss | 60.66 | +3.99 [+2.65, +5.52] | +1.07 [−0.82, +3.25] | 0.04 | 0.121 |
| PST students without IR⇄LiDAR check | 62.14 | +5.47 [+3.68, +7.26] | +1.04 [−0.95, +3.47] | 0.08 | 0.114 |
| **RGB-only students (seed 0)** | **61.17** | **+4.51 [+2.49, +6.40]** | +0.45 [−1.85, +2.82] | 0.04 | 0.132 |
| RGB-only students, seed run +10 | 61.86 | +5.20 [+3.18, +7.12] | +0.87 [−1.39, +3.35] | 0.07 | 0.132 |
| RGB-only students, seed run +20 | 59.44 | +2.77 [−0.04, +5.27] | −0.35 [−2.38, +1.56] | 0.09 | 0.116 |
| control: sim-only detector depth (det0) | 57.56 | +0.89 [−0.33, +1.91] | +0.59 [−0.44, +1.49] | 0.33 | 0.120 |
| control: PST det1 depth (before refiner) | 56.90 | +0.24 [−0.35, +0.66] | −0.33 [−0.92, +0.11] | 0.90 | 0.095 |
| control: RGB-only chain final depth | 56.73 | +0.06 [−0.35, +0.50] | +0.21 [−0.63, +0.54] | 0.83 | 0.082 |
| control: RGB-only chain det1 depth | 56.21 | −0.46 [−0.90, +0.25] | −0.43 [−1.19, +0.27] | 0.80 | 0.096 |

Means over the three seeds:
- **Test:** PST readers +4.03, RGB-only readers +4.16 UDS.
- **Val:** PST readers +1.03, RGB-only readers +0.32 UDS.
- Using the seed-0 RGB-only reader instead of PST's changes UDS by +0.36 [−0.79, +1.20] on test and −0.76 [−2.21, +0.19] on val (`vs_pst` in the json).

**Reading.**
- **The test-time fusion gain does not need the LiDAR-shaped corner reader.** Any corner-PnP student depth gives about the same gain on test, including RGB-only students. Its error is nearly uncorrelated with the refined distance (r ≈ 0.04–0.18) and has a similar spread (0.10–0.13).
- Second depths from the detector family (r = 0.8–0.9) give nothing, and det0 (r = 0.33) gives an insignificant +0.9.
- On val the PST readers are slightly ahead (+1.0 vs +0.3), but every val interval spans 0.
- What matters is an **independent geometric (corner-PnP) depth**, not the privileged teacher behind it. The privileged contribution to distance is upstream: the IR-refined labels and depth students (A1; clean3 item 1, ω = 0 vs fused targets).
- Part of the test gain is scale: fusion moves PST's +6.6 % pre-fusion bias toward the val-calibrated β′_k (CLEAN_REPORT §4c, −2.6 %). Any reader with a val-calibrated offset does this.

---

## A6. Missing intervals

### A6a. MAV6D, PST − variant, paired sequence bootstrap

| Split | Pair | ΔUDS | ΔAP | ΔAOE (°) | Δmedian yaw (°) | seqs PST better |
|---|---|---|---|---|---|---|
| test | PST − no mirror loss (`clean3/nomir`, 60.57) | +0.24 [−0.05, +0.47] | +0.28 [−0.35, +0.73] | −0.42 [−1.13, +0.05] | +0.01 [−0.41, +0.37] | 7/10 |
| val | PST − no mirror loss (59.52) | +0.20 [−0.27, +0.80] | +0.21 [−0.58, +1.23] | −0.28 [−0.76, +0.23] | −0.07 [−1.28, +1.16] | 10/16 |
| test | **PST − no IR⇄LiDAR check (`sup2/nogate`, 62.07)** | **−1.26 [−2.09, −0.45]** | −2.24 [−3.65, −0.88] | −0.33 [−1.24, +0.65] | −0.08 [−0.53, +0.47] | 1/10 |
| val | PST − no check (59.50) | +0.21 [−0.49, +0.95] | +0.25 [−1.08, +1.51] | −0.21 [−1.66, +1.00] | −0.21 [−2.35, +2.56] | 9/16 |
| test / val | same pairs without test-time fusion (rotation only) | nomir +0.09 [−0.01, +0.25] / +0.06 [−0.05, +0.17]; nogate +0.07 [−0.14, +0.27] / +0.05 [−0.22, +0.36] | 0 | as above | | |

**Reading.**
- The mirror loss gives a consistent but non-significant AOE gain: −0.42° test, −0.28° val. Neither interval excludes 0.
- **The IR⇄LiDAR check has no measurable orientation benefit:** ΔAOE −0.33 [−1.24, +0.65] test, −0.21 [−1.66, +1.00] val.
- On test, removing the check is better by 1.26 UDS. All of that comes from AP/ATE via its corner depth in the fusion (A2: +5.47 vs +4.14). On val the sign is reversed and n.s.
- Neither component can be claimed as significant on its own.

### A6b. AntiUAV3D v2 test: block-length sensitivity (B = 2,000, seed 0)

| L (test frames) | PST − Sim-only UDS | PST − Sim-only AP | PST − RGB-only UDS | PST − RGB-only AP |
|---|---|---|---|---|
| 50 | +11.18 [+7.68, +14.38] | +11.85 [+6.91, +16.43] | +2.81 [+1.35, +3.99] | +4.50 [+2.25, +6.35] |
| 100 | [+7.31, +14.85] | [+6.55, +16.91] | [+1.33, +4.07] | [+2.14, +6.40] |
| **176 (main)** | [+7.08, +15.44] | [+6.16, +17.42] | [+1.53, +3.96] | [+2.53, +6.23] |
| 300 | [+6.30, +16.33] | [+5.28, +18.55] | [+1.78, +3.81] | [+2.95, +5.99] |

- No interval reaches 0 at any L. The fraction of resamples ≤ 0 is 0.000, except PST − RGB-only at L = 100 (0.001).
- The 50 / 100 / 176 rows reproduce `A3/a_block_boot.json`.
- These remain within-flight intervals: 2 test sequences (A3 caveat).

### A6c. Is the gain larger at 40–60 m? Interaction test (AntiUAV3D test, PST − RGB-only per GT-range band, L = 176)

Per band:

| Band | n | ΔUDS | ΔATE (rel.) |
|---|---|---|---|
| < 20 m | 855 | +1.09 [−4.07, +3.91] | −0.001 [−0.014, +0.020] |
| 20–40 m | 485 | +2.37 [+0.52, +4.21] | −0.010 [−0.018, −0.003] |
| 40–60 m | 949 | +3.98 [+2.00, +5.75] | −0.018 [−0.028, −0.008] |
| ≥ 60 m | 335 | +3.39 [−0.70, +5.93] | −0.018 [−0.031, −0.002] |

Between-band differences of ΔUDS (two-sided bootstrap p, Holm over the 6 pairs):

| Pair | Δ of ΔUDS | 95 % CI | p | p (Holm) |
|---|---|---|---|---|
| 40–60 minus < 20 | +2.89 | [−0.75, +8.35] | 0.13 | 0.77 |
| 40–60 minus 20–40 | +1.61 | [−0.79, +3.95] | 0.19 | 0.93 |
| 40–60 minus ≥ 60 | +0.59 | [−2.17, +5.55] | 0.67 | 1.00 |
| ≥ 60 minus < 20 | +2.30 | [−3.29, +7.99] | 0.35 | 1.00 |
| 20–40 minus < 20 | +1.28 | [−1.88, +6.31] | 0.50 | 1.00 |
| ≥ 60 minus 20–40 | +1.02 | [−3.88, +4.18] | 0.54 | 1.00 |

- For AP, every Holm p ≥ 0.84.
- For relative ATE, the smallest is 40–60 vs < 20: Δ −0.018 [−0.043, −0.001], raw p = 0.042, Holm p = 0.25.

**Reading.**
- **No band difference is significant.** The per-band gains are significant from 20 m on, but "the gain is larger at 40–60 m" is not supported as an interaction.
- Suggested wording: "the gain is significant from 20 m on (20–40 m +2.4 [+0.5, +4.2], 40–60 m +4.0 [+2.0, +5.8] UDS); band differences are not significant (Holm-corrected p ≥ 0.77)".

---

## A8. Copying without the 1 m filter (held-out MAV6D val; student = det1; pseudo label = N1clean held-out reconstruction)

**Frames:** all 835 det0-conf ≥ 0.5 val frames, of which 9 have det1 ≥ 1 m from GT. The 1 m filter of N1clean keeps 803. CIs use a sequence bootstrap (B = 2,000; Theil–Sen B = 500).

| Group (n / seqs) | gap student−PL | gap student−det0 | student−GT | PL−GT | det0−GT | bias student / PL / det0 | OLS slope e_s on e_PL | Theil–Sen slope | Pearson / Spearman (s, PL) | Spearman (s, det0) |
|---|---|---|---|---|---|---|---|---|---|---|
| **all (835 / 14)** | **3.0 % [2.5, 3.7]** | **10.3 % [8.5, 11.1]** | 5.2 % | 5.5 % | 8.1 % | +2.7 / +2.8 / −7.0 % | 0.71 [0.39, 0.82] | 0.78 [0.63, 0.83] | 0.82 [0.52, 0.93] / 0.78 | 0.80 |
| Phantom 4 (792 / 9) | 3.0 % [2.4, 3.6] | 10.5 % [9.1, 11.2] | 5.5 % | 5.6 % | 8.2 % | +2.8 / +2.9 / −7.1 % | 0.71 [0.38, 0.83] | 0.79 [0.65, 0.84] | 0.82 / 0.79 | 0.83 |
| Mavic 2 (43 / 5) | 3.7 % [1.7, 4.4] | 5.8 % [3.6, 13.3] | 3.4 % | 3.8 % | 7.3 % | −0.2 / +2.1 / −2.9 % | 0.60 [0.36, 0.88] | 0.55 [0.47, 0.95] | 0.78 / 0.62 | 0.63 |
| *with 1 m filter, all (803 / 14)* | 3.0 % | 10.3 % | 5.0 % | 5.3 % | 7.8 % | +2.7 / +2.8 / −7.0 % | 0.78 [0.65, 0.83] | 0.78 | 0.83 [0.72, 0.90] / 0.76 | 0.79 |

**How much of the label's per-frame correction is inherited** (`a8b_inherit`). The model is e_s = a + b₀·e_det0 + b₁·c, where c = log(z_PL/z_det0) is the applied correction (its SD is 0.052).

| Group | b₁ (correction) | b₀ (det0 error) | b₁ after sequence-demeaning |
|---|---|---|---|
| all | **0.32 [0.11, 0.61]** | 0.69 [0.38, 0.81] | 0.29 [0.04, 0.63] |
| Phantom 4 | 0.30 [0.10, 0.58] | 0.69 [0.36, 0.84] | 0.27 [0.02, 0.64] |
| Mavic 2 (43 frames) | 0.26 [0.18, 1.40] | 0.53 [0.35, 0.85] | 0.49 [0.42, 1.71] |

**Reading.**
- The 1 m filter does not drive the result. Without it, the gap is still 3.0 % to the label vs 10.3 % to raw det0. The correlation CI widens (0.82 [0.52, 0.93]), and the OLS slope drops from 0.78 to 0.71 because of 9 misdetections. The robust Theil–Sen slope is unchanged at 0.78.
- **Correlation does not separate copying from shared difficulty:** Spearman(student, det0) 0.80 ≥ Spearman(student, PL) 0.78. Magnitude and bias separate them:
  - the student carries the label's +2.8 % scale (det0 is −7.0 %);
  - it inherits only about **a third** of the label's frame-by-frame correction (b₁ ≈ 0.3);
  - it keeps about 0.7 of det0's per-frame error.
- Suggested wording: "the student inherits the pseudo labels' global scale and about a third of their per-frame correction (held-out val: 3.0 % from the label vs 10.3 % from the raw detector)".
- Mavic 2 has 43 frames in 5 sequences. Its rows are indicative only.

---

## A10. Facts: AntiUAV3D v2 test scoring history and test-label reads

Full table: `A10_FACTS.md`.

**Scorings, in chronological order (machine time = Michigan + 13 h).**

| When | Variant | UDS |
|---|---|---|
| 11-12 05:04 | `out2antifs` sim-only | 30.55 |
| 11-12 05:04 | `out2antifs` PST (reported) | 41.74 |
| 11-12 05:33 | `in2antifs` sim-only / PST | 10.46 / 15.82 |
| 11-12 10:55 | `out2antifs_ir` | 41.84 |
| 11-12 11:07 | `out2antifs_lid` | 37.52 |
| 11-12 13:31 | `out2antifs_none` (RGB-only) | 38.92 |
| 11-12 13:34 | local Sim-Indoor baselines GUPNet / MonoCD / MonoDGP | 17.25 / 18.48 / 20.44 |
| 11-13 03:57 | `mix2antifs` sim-only / PST | 23.24 / 33.25 |
| 11-13 04:11 | M12 reproduction | 41.74 |
| 11-13 04:36 | M12 single model, no TTA | 42.85 |
| 11-13 07:43 | height-swap variant | – |
| 11-13 15:32 | MonoCD trained on outdoor sim | 10.62 |
| 11-14 02:57 | cpu5 rows; first scoring of `pcmd_nofuse_test` | 41.69 |
| 11-14 03:14 | seed run `antifs_full_s10` | 41.38 |

- `antifs_none_s10`, `antifs_full_s20` and `antifs_none_s20` were not finished at writing.
- The paper's outdoor-source baselines (e.g. MonoDGP 20.80 / 27.78) were scored on the advisor's server. There is no local score file for them.
- Disclose as: 7 chain variants plus 4 local baselines scored on this test split.

**Test-label reads.**
- **Split construction (11-12 01:28–01:54):** `make_subset_fs.py` used GT distance (the < 80 m pool and a near/mid/far balance term) and the M400 placement. This contradicts its own "no GT" comment.
- **Duplicate check and sealing:** `seal_and_check.py` (01:55); `antifs_test_meta.npz` written at 01:57, with no writer script on disk.
- **Diagnostics after scoring:** C1, R4, N2 (includes a test-fitted "oracle" constant orientation), height, A3, Fig. 3 selection, breakdown, orient_dist, cpu5, and this panel.
- **The only test-derived analysis setting** is A3's block length L = 176. A6b shows the conclusions do not depend on it.
- **Prior exposure:** 98.2 % of v2 test frames are the v1 **validation** sequence, which was seen with labels during v1 development.

---

## A11. Size prior (full tables: `A11_SIZE.md`; figure: `A11_size_sweep.pdf`)

**Decoding.** The decoder is z_det = f_x · span_x(s̄, R_det) / ŵ₂D. The identity was checked to hold exactly. The detector's log distance error is the sum of three terms:
- pure size error;
- rotation footprint;
- width read.

On AntiUAV3D M4E, PST: size **+0.54** (s̄ is ×1.72 too large), rotation footprint −0.31, width read **−0.30** (over-read ×1.35), final −0.05. C1's "size +0.22" is size + rotation footprint combined.

**Perfect size** (GT size in the decoder, same apparent width and rotation; test UDS):

| Arm | MAV6D | AntiUAV3D |
|---|---|---|
| PST | 60.81 → 61.83 | 41.73 → 19.99 |
| RGB-only | 57.38 → 59.40 | 38.92 → 19.80 |
| Sim-only | 49.51 → 48.37 | 30.55 → 13.32 |

- On AntiUAV3D, AP drops to 0 and distance is ×0.55. The oversized s̄ is what cancels the two −0.30 errors.
- On MAV6D, PST − RGB-only goes from +3.43 to +2.43.

**Sweep.** Distance error tracks size error one-for-one: ±10 % size gives ≈ ±0.10 log distance, which costs 8–13 UDS on MAV6D. PST stays ahead of RGB-only at every bias on MAV6D (+1.5 to +2.4 UDS).

**AntiUAV3D with both biases removed** (oracle, UDS gaps):

| Gap | Deployed | Per-frame size + global width | Global size + global width | Each arm's own global bias removed |
|---|---|---|---|---|
| PST − Sim-only | +11.18 | +13.07 | +13.19 | +14.12 |
| PST − RGB-only | +2.81 | +1.01 | −0.18 | +0.39 |
| PST − RGB-only, M4E only | +2.85 | +0.88 | +0.13 | +0.53 |

- The gain over sim-only survives and grows; the per-frame error falls from 11 % to 6.5–7 %.
- **Most of the AntiUAV3D gain over RGB-only is global scale:** final bias is −5.6 % for PST vs −8.6 % for RGB-only.
- On MAV6D the same oracle keeps PST − RGB-only at +3.38 (deployed +3.43): the gain there is per-frame.
- M400 (48 frames, s̄ ×0.54 of true size) is reported separately, never pooled.

---

## Cross-cutting conclusions for the paper

1. **Distance gains are largely scale on both targets.** The clean chain's +4.6 % test over-estimate comes from a c̄ / selection mismatch in the pseudo labels (A1a). The AntiUAV3D PST − RGB-only gain is mostly global scale (A11). The test-time fusion gain mostly does not depend on which teacher shaped the corner reader (A2).
2. **Where the privilege shows per frame:**
   - MAV6D label spread is 7.4 vs 8.1 % (fused targets 6.7 vs 7.9 %);
   - the MAV6D per-frame gain survives scale removal (+3.38 UDS);
   - PST − Sim-only on AntiUAV3D survives (+13 UDS).
3. **Claims to soften:**
   - "larger at 40–60 m" (interaction n.s.);
   - "the agreement gate drops unreliable labels" (it drops the most-corrected and best-fixed crops);
   - the IR⇄LiDAR orientation check (no measurable AOE benefit; test UDS favours removing it);
   - the mirror loss (n.s.).
