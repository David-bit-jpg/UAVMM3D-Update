# SENSES_REPORT: evidence for LiDAR's role and the IR⇄LiDAR check (review 11-13)

Everything here was read-only, except the two CPU diagnostics in this folder (`agree_diag.py`, `boot_sel.py`). Paths are relative to `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/` unless they start with a drive letter. "OU" means `wf21/orient_uda/`. The diagnostics read the sealed MAV6D **train** labels only to score pseudo-labels, the same way `OU/pseudo_sets.py --diag` does. Nothing they produce feeds training or selection.

## Q0. How the check is wired in the code

**MAV6D final chain (S6I students)**
- **Where d_IL is computed.** `OU/pseudo_sets.py:52`: `d_il = PB.sym(out['IR']['R'], out['LiD']['R'])`.
  - `out[g]['R']` is the group's own symmetric ensemble (SA-TM), built from that group's members only (`pseudo_sets.py:46-51`).
  - The groups are defined at `:25-29`: IR = IR_s0,s1; LiD = LiD_s0,s1.
  - So the IR-distilled pair's own rotation is compared with the LiDAR pair's own rotation. Neither is compared against the six-network average.
- **Distance and threshold.** `sym()` is defined at `OU/pseudo_build.py:34-35`: the full 3-D geodesic distance, minimised over the four R_z(90°·q) rotations, i.e. folded by 90°. The threshold is `TH = 10.0` (`pseudo_build.py:30`).
- **Label vs check.** The label comes from `T6` = IRLiD s0,s1 + IR s0,s1 + LiD s0,s1 (`pseudo_sets.py:29`), via SA-TM and then PnP (`:56-60`).
  - For the `t6il` config the check against the colour-only group is off: `rg=None` gives `d_tr=0` (`:31, :61`).
  - The gate is therefore `d_il<=10` only (`:71`).
  - Order of operations: (a) the IR-pair rotation is compared with the LiD-pair rotation to give pass/fail; (b) passing frames get the 6-network label. The label itself is never checked.
- **What the IR+LiDAR group does.** It enters only the 6-network label average. It is not part of the check. It also initialises each student: `ou_train.py:467` loads `irlid/release/kzm_s{seed}`, and `wf24/logs/train_S6I_s0.log:7` confirms this.
- **The filter is applied in the final run.**
  - `wf24/run_students.sh:8` runs `--tgt gate --pseudo pseudo_train_t6il`.
  - `ou_train.py:454-456`: `pool = np.where(P['gate'])`, then a random 80% of it.
  - `wf24/logs/train_S6I_s{0,1,2}.log:5`: "帧池 1220（gate 1525 / 全部 4289）" (frame pool 1,220; gate 1,525 / all 4,289).
  - `OU/pseudo/sets_info.json` (t6il): n_gate 1525 / 4289.
  - Target batches therefore draw only from passing frames.
- **Caveat 1.** The label-free rerun (`nolabel/`) did **not** retrain the orientation students. It reuses S6I s0,s1 (`nolabel/PREREG.md:9`; `nolabel/run_chain_nl.sh:78-79`).
- **Caveat 2.** The teacher readings behind the 1,525 frames come from the `wanb_train_kz` crop windows (`pseudo_build.py:25`). The wanb detector's own training pseudo-labels contained cross-frame information (`OU/README.md:283`, honesty note).

**AntiUAV3D chain (srcx)**
- **Where d_IL is computed.** `srcx/lib/ori_x.py:269`: `d_il = sym(out['IR'][0], out['LiD'][0])`.
  - The groups come from `run_x.sh:90`: T = irlid0,1 + kdir0,1 + lid0,1; IR = kdir0,1; LiD = lid0,1.
  - Each group gets its own ensemble (`ori_x.py:221-256`).
  - That ensemble is a simpler yaw-folded trimmed mean, not the MAV6D SA-TM code, but it serves the same purpose.
- **Distance, threshold and order.** `sym()` (`ori_x.py:32-33`) is the 3-D geodesic distance folded by 90°. The gate is `d_il <= 10.0` (`:271`). The label is `out['T']`, the 6-network ensemble (`:270`). The order is the same as on MAV6D: the two group rotations are checked, and the T label is used for passing frames.
- **The filter is applied.**
  - `ou_train_x.py:456` keeps only passing frames.
  - Students start from the same-seed IR+LiDAR net (`ou_train_x.py:469`; `run_x.sh:79`).
  - `srcx/out2antifs/ou/runs/S6I_s0/train_*.log:4`: pool 692 (gate 865 / 3293).
  - The mix run logged 524 passing frames (15.9%), d_il median 23.0 (`srcx/mix2antifs/logs/run_x.log:165`).
- **Single-modality arms.** In the `ir`, `lid` and `none` arms, the "gate" is the agreement between the two seeds of one group (`run_x.sh:91`).

## Q1. Is the 10° IR⇄LiDAR filter useful for anything we can show?

**Verdict: label quality is supported. A training benefit, and "cross-sensor beats colour-only", have no evidence.**

| Possible benefit | Evidence | Source |
|---|---|---|
| Pseudo-label quality on passing frames | 16.52° on passing frames vs 20.64° on all frames (6-network label). Symmetric yaw: mean 11.89° vs 15.70°; median 8.14° vs 12.42°. | `OU/pseudo/sets_diag.json` (t6il); `review1113/senses/agree_diag.json` |
| Student yaw, mean, 5 seeds (before slimming, two-level check) | No filter vs filter: −0.08° val, −0.28° test. Cluster interval: +0.04 [−1.07, +0.79] / −0.14 [−0.54, +0.54]. | `OU/README.md:262, :173` |
| Median yaw, 3-member ensemble | Filter vs no filter: val 14.06 vs 13.67; test 6.34 vs 6.05. No filter is better on both. | `OU/README.md:110,112` |
| Median yaw, 5-member deployment | Val 13.38 vs 13.85 (filter better); test 6.09 vs 5.82 (filter worse). Mixed. | `OU/README.md:139-145` |
| Tilt | Test tilt −0.04 for no filter. Deployment tilt: val 8.69 vs 8.78, test 6.52 vs 6.60. Within noise. | `OU/README.md:262, :139-145` |
| Full chain UDS | Filter vs no filter vs filter + prototype: 62.78/68.38 vs 62.77/68.40 vs 62.76/68.39. A tie. | `OU/README.md:192` |
| Spread across seeds (5 seeds; my arithmetic from per-seed numbers) | Filter vs no filter, SD of the delta: test 0.23 vs 0.50 (filter tighter); val 0.74 vs 0.55 (filter looser). Mixed, so not a claim. | `OU/README.md:122-125` |
| Frames used | Final: 1,525 of 4,289 (35.6%); before slimming: 1,088 (25.4%). Accuracy is the same. | `OU/README.md:262` ("用 1/4 的帧达到同样效果", same result with a quarter of the frames); `sets_info.json` |
| Training compute | Not saved. Steps are fixed at 2,250 and all frames must be labelled anyway, so only the real-frame pool shrinks. | `ou_train.py` args; `wf24/logs/train_S6I_s0.log:1` |
| Final configuration (slimmed) | No ungated run exists. The only comparison adds the colour check: S6R (937 frames) 16.35/9.83 vs S6I (1,525 frames) 15.64/9.32. Adding the colour check made it worse (fewer frames). | `wf24/README.md:41-46`; `tab_orient.tex:28-29` |
| "Check vs colour-only" row | This row adds a second check; it does not replace d_IL. It cannot show that cross-sensor agreement beats colour agreement. | as above |
| Matched-count selection (new) | Keeping the 1,525 lowest-scoring frames, the error of the 6-network label is: d_IL 16.52°; colour pair s0,1 vs s2,3 (seed-only diversity) 16.98°; label vs colour pair 16.43°. Random equal count: 20.64 [20.18, 21.13]. | `agree_diag.json` |
| Matched-count differences (new, cluster bootstrap) | d_IL − colour pair: −0.46° [−1.01, +0.09], crosses 0. d_IL − label vs colour: +0.09° [−0.66, +0.67]. The d_IL and colour-pair selections share only 50% of frames. Spearman correlation of score with label error: 0.30 / 0.28 / 0.32. | `boot_sel.json`; `agree_diag.json` |
| Flip / quadrant errors | Not measurable. The metric and `sym()` fold 90° (and so 180°), and the paper evaluates under the same symmetry. Proxy, share of gross errors: geodesic >30° is 14.1% on passing frames vs 24.5% on all; yaw >20° is 20.9% vs 32.8%. A colour-pair selection gives 16.2% / 22.2%. | `agree_diag.json` |
| AntiUAV3D | No run with the filter off. Passing rates: out 26.3%, mix 15.9%, in 11.7%. No label errors were computed: the AntiUAV3D train labels are sealed, so I did not open them. | `srcx/*/logs/run_x.log` (lines above) |

**Recommendation.** Describe the filter honestly; do not present it as a component that adds accuracy. Defensible wording: "the check keeps 36% of frames, whose pseudo-labels are 20% more accurate (16.5° vs 20.6°), and the student trained on them matches one trained on all frames." Do **not** write "cross-sensor agreement beats agreement with a colour-only labeler":
- At matched count it is indistinguishable from seed agreement between two colour-only pairs (−0.46°, interval crosses 0).
- It is also indistinguishable from label-vs-colour agreement (+0.09°).

This also closes the TODO at `sec/X_suppl.tex:277`, at the label-quality level (no student training).

## Q2. LiDAR contributions

**(a) Height network: weak, but the number is real.**
- **The number.** Removing the height network: UDS 65.70 → 64.20 val (−1.50) and 72.31 → 71.08 test (−1.23); test orientation error 12.86 → 13.08 (`wf24/README.md:18,25,27`).
- **Version.** Pre-slimming final: detector 3, distance 5, orientation 5 + mirror, height network 2.
- **Seeds.** It is a **re-evaluation without retraining**, so it has no seeds or interval. The distance students' training targets still used the height network.
- **What replaced it.** Every frame gets the constant aspect `FE_HTR` = h/mean(l,w) of s̄ = 0.200/0.349 = 0.573 (`wf23/mirror/chain_eval_m.py:147-149`; s̄ at `wf24/README.md:82`).
- **What the network actually supplies.** It provides only **two per-colour constants**: the median of its readings per colour cluster, r_k = 0.380 (dark) / 0.711 (bright) (`chain_eval_m.py:89-114, :150-152`; `nolabel/chain_eval_m_nl.py:114,152`; `wf24/README.md:82`). Per frame, its output is not used.
- **Median aspect of the 7 simulated models.** Never measured (no file found).
- **AntiUAV3D (confounded).** The `ir` arm uses a constant aspect and no LiDAR teachers, and scores test UDS 41.84 vs 41.74 for the full arm (single run; `srcx/out2antifs_ir/logs/run_x.log:273` vs `srcx/out2antifs/logs/run_x.log:377`).

**(b) Fused corner distance: supported on MAV6D.**
- **Student alone, 3 seeds (5–7), no fusion at deployment.** Orientation fixed at PSP×5 + mirror. Fused vs IR-only targets: val +0.67 [+0.03, +1.27], test +1.47 [+0.58, +2.04]; test AP +2.74 [+1.20, +3.75] (`wf24/README.md:79`).
- **Earlier version.** +0.74 [+0.05, +1.24] / +1.40 [+0.57, +2.02] (`wf23/fusetrain/README.md:76`).
- **Development-version ladder (tab_ablation), 2 seeds each.** +0.68 [+0.23, +1.55] / +1.48 [+0.54, +2.05] (`wf24/README.md:77`).
- **Does the corner PnP box height use r_k? Yes.**
  - The box height is set to `r_k·mean(l,w)` before PnP: `nolabel/ztrain_nl.py:33` (training targets) and `wf23/mirror/chain_eval_m.py:162-166` (deployment).
  - The PnP geometry takes lwh from that prediction: `night1003/scratch/kzst1/kzst_common.py:154`.
  - `wf24/README.md:68` states the same.
- **AntiUAV3D.**
  - Fusion at deployment helps on val: 34.98 vs 34.45 (`out2antifs` `score_val.json`).
  - The no-LiDAR `ir` arm, whose corner distance is read by IR-initialised students, ties the full arm (41.84 vs 41.74 test).
  - So the benefit there comes from a second measurement, not specifically from LiDAR.

**(c) Compositing matte: no evidence.**
- The only comparison is inference-time: IR-only matte vs IR∩LiDAR matte, 18.61° vs 18.65° (`wf20/design/README.md:120-121, 127`).
- The LiDAR veto removes a median of 1.3% of the silhouette (p90 4.9%), and only 2.0% of samples lose more than 10% (`:132-135`, `:21`).
- The planned training arm with an IR-only matte (`PI`) was never run (`wf20/design/README.md:181,197`; no results).

**(d) LiDAR-taught orientation group: weak or no evidence.**
- **Labeler accuracy on all frames (train diagnosis).** LiD pair 21.76°, IR pair 22.15°, colour pair 21.98–22.49°; IR+LiD pair 21.64°.
  - Labelers: T4 (IR+LiD pairs) 21.15°; T6 20.64° (`agree_diag.json`; `OU/README.md:66-70`).
  - Labeler sets: IRLiD×5 20.32° vs colour×5 21.03° (`equal5/README.md:61-64`).
- **Students, equal-count control (pre-registered).** IRLiD×5 − colour×5: val −1.43 [−1.84, −0.82], test −0.05 [−0.56, +0.40]. Fails the registered criterion (`equal5/README.md:82-83,97`).
- **Students, 9 labelers vs colour×5.** Val −0.52 [−1.28, +0.24], test −0.47 [−1.17, +0.12], not significant (`:86-87`).
- **Balanced labeler set.** The set with LiD pair + IR pair (T5B) is worse than IRLiD×5 on val, +0.92 [+0.47, +1.31] (`:91`).
- **Removing the IR+LiD group from the labelers (T4 vs T6).** Label error on passing frames rises 16.52 → 17.06° (`wf24/README.md:35-36`). No student was trained.
- **Initialisation.** No MAV6D run starts the students from IR-only or colour-only nets.
  - On AntiUAV3D, the single-modality arms change initialisation, labelers and gate together. Test AOE / UDS: full 17.11 / 41.74; ir 17.19 / 41.84; lid 16.73 / 37.52; none 16.45 / 38.92 (single runs; `srcx/out2antifs*/logs/run_x.log` final `all` lines).
  - The colour-only `none` arm has the best orientation. The `lid` drop comes from the position chain (no IR silhouettes).
