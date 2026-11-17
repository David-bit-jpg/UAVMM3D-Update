# cleanfix Part A: CPU recomputes on the clean chain (11-14, machine time ~15:00)

Clean chain = `gproto/srcx/{in2mav, in2mav_none, out2antifs}` (run_x.sh). All outputs live under `gproto/review1113/cleanfix/`; no run dir, shared lib or paper file was modified.
Real labels are used only for post-hoc error measurement, never for a choice. Each test variant was scored **once**. Calibration (former val) labels were never read.

| Item | Paper line | Old (dev) | **Clean value** | Split, n | Source |
|---|---|---|---|---|---|
| A1 | 3_method.tex:64 | ≤1.8 % vs 20.6–71.9 % | **≤0.09 %** of bank windows hold ≥10 % of a UAV, against **21.5 % / 69.2 %** for random windows (refiner / orientation size) | MAV6D train frames; banks 4,976 (refiner) / 4,420 (orientation); random 5,225 / 6,873 | `A1/A1_RESULT.md`, `A1/a1_contam.json` |
| A1 | X_suppl.tex:88 | 0.10 % (dev bank) | **0.04 %** (2 of 4,976), refiner bank | MAV6D train | same |
| A1 | X_suppl.tex:276 | halo 1.3 % | **0.7 %** median of IR-silhouette pixels removed by the LiDAR veto (p90 3.8 %); outdoor crops 3.1 % | simulation crops, 5,911 (in) / 5,196 (out) | `A1/a1_halo.json` |
| A2 | X_suppl.tex:139 | 70.16→69.03 (AP 56.44→54.40); +1.15 | β′_k from unlabeled **training** images instead of calibration images: **MAV6D test UDS 60.81→58.07, AP 40.93→36.24**; **AntiUAV3D test UDS 41.73→40.02, AP 27.89→25.24** (s̄ unchanged) | test, 4,800 / 2,624 frames | `A2/*/preds/btrain_fused_test.pkl.score_*.json`, `A2/*/deploy_btrain.json` |
| A3 | X_suppl.tex:162 | width 12–15 % narrow (11.6 %) | detector's 2-D width is **16.3 % narrower** than the projected GT box (median; train 16.8 %); in **68 %** of test detections (87 % of train) the GT rotation's span is wider than the detector rotation's; detector rotation head AOE 25.5° mean / 24.0° median | test detections within 1 m of GT, n 4,396 (train 11,725) | `A3/a3_result.json` |
| A3 | X_suppl.tex:162 | 69.51→55.89 | GT rotation in the width-depth formula. **Without normalisation:** refined chain before fusion (pcmd_nofuse) **56.67→35.37**, PST (fused) **60.81→40.55**. **With unlabeled-style normalisation** (÷ median s, here from train, as orient2pos did with the other split): nofuse **56.67→57.52** (+0.85), fused **60.81→58.78** (−2.03) | test, 4,800 | `A3/preds/*.score_real_test.json` |
| A3 | X_suppl.tex:162 | \|ρ\| ≤ 0.09 | **Does not hold on the clean chain.** Spearman of depth error log(z/z_gt) with log s_GT: **−0.38** (nofuse) / **−0.27** (PST); |depth error| vs AOE: **0.17** / **0.32**; signed depth error vs AOE 0.05 / 0.25 | test TP (centre < 1 m), n 4,430 / 4,463 | `A3/a3_result.json` |
| A4 | X_suppl.tex:276 | RGB pair −0.46 [−1.01, +0.09] | Matched count (4,301 frames, smallest RGB⇄RGB disagreement KZs0 vs KZs1), same deployed 6-net label: **15.87° vs 15.40°** for the IR⇄LiDAR check, **+0.47° [+0.05, +0.90]**. With the RGB pair's own label: 16.39°, +0.99° [+0.09, +2.04]. RGB pair's own 10° rule (2,046 scorable frames): 15.61°, +0.21° [−0.23, +0.62] | train, scorable (det0 on target) 6,047 frames / 52 seq; IR⇄LiDAR passes 2,424, RGB matched 2,429 | `A4/a4_result.json` |
| A4 | X_suppl.tex:276 | drop IR+LiDAR group 16.52→17.06 | Label from IR-distilled×2 + LiDAR×2 only: **15.40°→15.53°** (+0.13° [−0.08, +0.38]) on the same passing frames | train, 2,424 | same |
| A5 | tab_main.tex GUPNet MAV6D row | comment cites PAPER_TABLE_adabn.md | **Simulation-only (no AdaBN).** `to_pf.py` docstring: the `_adabn` file name is only so paper_table reads it; whole-image methods get no AdaBN; predict.py uses `model.eval()`. Rescored with score_x: AOE 22.16, ATE 1.165, ASE 12.67, AP 11.35, UDS 22.13, mDR 70.3: identical to the row | test, 4,800 | `A5/gupnet_s0_real_test.score.json` |
| A6 | X_suppl.tex:205–211 disclosure | "six PST chain variants …" | see list below: **14 PST-chain variant families** (15 when the running none s20 seed finishes) + 4 local baselines (+ sim-only rows) | AntiUAV3D v2 test | below |

## Details and caveats

**A1.** Replays of `wf24/diag_randwin.py` and `wf21/strict/diag_bank.py` with the clean bank rules (window counts assert-equal to the saved banks). The clean composites do **not** use the same code or crops as the dev 1.3 % figure (`wf20/design/prep_sim.py` on 552 test crops): they use `kzpaste_train.matte()` alphas of `kz/crops_6000` (same rules, separate copy, different minimum counts), hence the recompute. The LiDAR veto concerns only the orientation composites; the refiner composites use the IR silhouette without a veto. AntiUAV3D banks not measured (the original measured MAV6D only).

**A2.** `A2/lib/chain_a2.py` = `srcx/lib/chain_x.py` with one change: b_k (= β′_k) from the train detections (student corner-PnP depth on `tr1` vs the single-frame chain's depth), n per class 6,943 / 5,903 (MAV6D), 1,084 / 1,699 (AntiUAV3D). Reproduction check: with `A2_BSRC=val` the copy reproduces the deployed `pcmd_fused_test.pkl` bit-exactly for both targets (max |Δ| 0.0). β′_k MAV6D (dark, bright) 0.528/0.318 (calibration) → 0.491/0.235 (train); AntiUAV3D 0.530/0.992 → 0.736/1.134. ω, c_k, r_k, s̄ unchanged. Clean reading: estimating β′_k on training images is worse on both targets (−2.74 / −1.71 UDS), i.e. the constant does not transfer from training to held-out sequences; supports "global scale is not identified without labels". The "+1.15 (version before slimming)" has no clean counterpart without GPU (needs s̄ from train → re-inference); delete it.

**A3.** Script `A3/a3_gtrot_decode.py` replicates `wf21/orient2pos/orient2pos.py` (resolve: same ray, same s̄, z′ with projected width under R_gt = detector w2d, 12 perspective iterations). For the fused PST the scale enters as s^(1−ω) (the fusion's log-linear weight) and β′_k stays as deployed (re-estimating it would need GT rotations on calibration images). The dev "69.51→55.89" was the normalised arm (orient2pos README: "已做无标签归一"); its clean counterpart on the refined chain is **+0.85 (57.52 vs 56.67), i.e. no longer worse**; only the final PST worsens (−2.03), and without normalisation both collapse (−21 / −20 UDS). The paper's "worsens an intermediate chain's test UDS" therefore needs rewording; and the clean depth error **is** correlated with orientation (ρ −0.27 to −0.38 with the GT span ratio; |err| vs AOE 0.32 for PST, partly because the fused z_cor comes from the orientation students). Suggested: keep the collapse + narrow-width facts (16 %, 68 %), drop "|ρ| ≤ 0.09", and state that inserting GT rotation into the width formula without the matching width correction lowers PST from 60.81 to 40.55 (58.78 after rescaling by the median ratio).

**A4.** Script `A4/a4_select.py`; ensembles = `ori_x.ensemble`; reproduction: 6-net label vs deployed `pseudo_tr1.npz` max 3e-6°, d_il exact, gate 4,301; label errors reproduce the paper's 15.4° / 18.8°. The clean chain has only two RGB-only nets read on train crops (`in2mav_none`, KZs0/KZs1, on that control's own det1 crops), so the "RGB pair" is two single nets differing only by seed, not two pairs. Clean reading: cross-sensor agreement selects slightly better labels than seed agreement at matched count (+0.47° [+0.05, +0.90]), small; the old wording "about as accurate … we do not claim that cross-sensor agreement is special" should be softened to "slightly more accurate (0.5°)". The irlid pair adds little to the label (0.13°, CI spans 0).
Side finding: `figures/data/README_figdata.md` l.63 says the RGB-only control's labels are "17.4° on its own RGB⇄RGB-gate frames"; from the CSV, 17.4° is the RGB-only label on the **IR⇄LiDAR** gate frames (2,424); on its own gate frames (2,046) it is 16.1°.

**A5.** No change needed to the numbers; fix the tab_main.tex comment (the source table PAPER_TABLE_adabn.md is titled "AdaBN 后", but GUPNet's whole-image prediction never had AdaBN).

**A6. Every scoring of the AntiUAV3D v2 test split** (A10_FACTS.md #1–#16 compiled 11-14 03:20, plus the score files written since; mtimes of `*.score_antifs_test.json` / `pred_outdoor_v2_test.pkl.score.json`):

| # | Machine time | Variant | UDS (AP) |
|---|---|---|---|
| 1–2 | 11-12 05:04 | out2antifs: sim-only det0; **PST (reported)** | 30.55 (16.04); 41.74 (27.89) |
| 3–4 | 11-12 05:33 | in2antifs: sim-only; PST indoor source | 10.46; 15.82 |
| 5 | 11-12 10:55 | out2antifs_ir (IR-distilled teachers only) | 41.84 (28.08) |
| 6 | 11-12 11:07 | out2antifs_lid (LiDAR teachers only) | 37.52 (21.31) |
| 7 | 11-12 13:31 | out2antifs_none (RGB-only teachers, reported control) | 38.92 (23.39) |
| 8 | 11-12 13:34 | local baselines (Sim-Indoor): GUPNet / MonoCD / MonoDGP | 17.25 / 18.48 / 20.44 |
| 9–10 | 11-13 03:57 | mix2antifs: sim-only; PST mixed source | 23.24; 33.25 |
| 11 | 11-13 04:11 | M12 reproduction of #2 (bit-identical) | 41.74 |
| 12 | 11-13 04:36 | M12 PST single model, no TTA | 42.85 (30.07) |
| 13 | 11-13 07:43 | height swap on #2 (in memory) | HEIGHT_REPORT §2 |
| 14 | 11-13 15:32 | local baseline MonoCD outdoor sim | 10.62 |
| 15 | 11-14 02:57 | cpu5 rows (b) det1, (c) +students, (c′) +r_k, (d) pcmd_nofuse | 37.58 / 38.66 / 38.66 / 41.69 |
| 16 | 11-14 03:14 | seed run antifs_full_s10 (PST) | 41.38 (27.25) |
| **17** | 11-14 05:57 | seed run antifs_none_s10 (RGB-only teachers) | 33.49 (15.09) |
| **18** | 11-14 09:39 | box-hull instead of IR silhouette (out2antifs): fused; nofuse | 41.01 (26.75); 41.25 (27.17) |
| **19** | 11-14 13:05 | seed run antifs_full_s20 (PST) | 42.01 (28.27) |
| **20** | 11-14 14:55 | **cleanfix A2**: β′_k from training images (this task) | 40.02 (25.24) |
| pending | launched 11-14 13:05 | seed run antifs_none_s20 (still running) | – |

Sim-only rows of #16, #17, #19 are re-scores of the identical det0 (30.55). Count for the disclosure: PST-chain variants scored on this test = (1) out (reported), (2) in, (3) mix, (4) ir, (5) lid, (6) none, (7) M12 single, (8) height swap, (9) cpu5 intermediate rows (det1 / +students / +r_k / nofuse), (10) box hull (fused + nofuse), (11) seed full s10, (12) seed full s20, (13) seed none s10, (14) A2 β′-from-train → **14 variant families** (15 once none s20 finishes; 19 PST-type prediction sets counting the 4 cpu5 rows and box-hull nofuse separately), plus 4 local baselines (GUPNet, MonoCD, MonoDGP Sim-Indoor; MonoCD outdoor sim) and the sim-only detectors (out/in/mix). Server baselines (advisor pack) were scored off-machine. The one test-derived analysis parameter remains A3's block length L = 176. "six PST chain variants" in X_suppl.tex:205–211 is stale.
