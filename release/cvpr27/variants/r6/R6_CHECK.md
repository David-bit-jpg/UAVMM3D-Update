# R6_CHECK: checking reviewer R6's factual claims (11-13)

Read-only, CPU only. Paths are relative to `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/` unless they start with a drive letter. "OU" means `wf21/orient_uda/`.

No labels were read anew. Every pseudo-label error below already exists in a JSON written by `--diag` or `agree_diag.py` (MAV6D train labels, diagnosis only). AntiUAV3D labels were not opened.

## Q1. Sec. 4.3 orientation numbers: R6's version-mixing claim (C5) is CORRECT

Paper sentence (`D:/CVPR27_UAV/sec/4_experiments.tex:34`): "Teacher labels err by 20.3° vs 21.0° for five plain color labelers ..., and frames passing the IR⇄LiDAR check by 16.5° vs 20.6° overall. With a small label gain (0.7°) and a third of the frames kept..."

Where each figure comes from:

| Figure | Labeler set | Version | Source |
|---|---|---|---|
| 20.3° | IRLiD×5 (all 5 seeds of the IR+LiDAR group); arm T5A of the equal-count control | before slimming (11-08 21:21) | `wf21/equal5/premise_t5.json`: geo_err_mean T = 20.316; `wf21/equal5/README.md` §5 |
| 21.0° | plain colour KZ×5 (RGB_s0..s4), the labels of arm RA | before slimming (11-08 14:33) | `OU/pseudo/premise.json`: group_err_mean RGB = 21.031 (same value in `premise_t5.json`) |
| 16.5° | final 6-labeler label T6 = IRLiD×2 + IR×2 + LiD×2, frames with d_IL ≤ 10° | final (slimmed; S6I training label) | `OU/pseudo/sets_diag.json`: t6il err_pass = 16.517 (n = 1,491 scorable passing frames) |
| 20.6° | same T6 label, all scorable frames | final | `sets_diag.json`: t6il err_all = 20.644 (n = 4,131) |
| 0.7° | 21.03 − 20.32, i.e. IRLiD×5 vs colour×5 | before slimming | arithmetic |

So the first pair is the pre-slimming equal-count comparison (5 vs 5), and the second pair is the final T6 label. The final T6 label is not IRLiD×5.

### Correct final-version numbers

Same teacher readings for every row: `OU/teach/teach_train.npz`, dated 11-08 14:32 and unchanged since. Same 4,131 scorable train frames. Metric: mean symmetric geodesic error.

| Quantity | Value | Source |
|---|---|---|
| Teacher label T6 (6 labelers; the S6I training label), all frames | **20.64°** (median 17.76; yaw mean 15.70, median 12.42) | `sets_diag.json` t6il; `review1113/senses/agree_diag.json` labels_all_frames.T6 |
| Five plain colour labelers (RGB×5), all frames | **21.03°** (median 18.41; yaw mean 16.20, median 13.18) | `OU/pseudo/premise.json`; `wf21/equal5/premise_t5.json` |
| Difference | **0.39°** (R6's 0.36 subtracts from the rounded 21.0) | arithmetic |
| T6, passing vs all frames | **16.52°** (1,491 scorable of 1,525 passing, i.e. 35.6% of 4,289) **vs 20.64°** (4,131); median yaw 8.14 vs 12.42 | `sets_diag.json`; `sets_info.json` (n_gate 1525 / 4289) |

Notes:
- **Like-for-like scoring.** RGB×5 is scored as its group SA-TM rotation, while T6 is scored as its PnP label. The two scorings agree to all digits for IRLiD×5 (both 20.316 in `premise_t5.json`), so the 0.39° compares like with like.
- **Other colour baselines.**
  - Two-network colour pairs score 22.49° and 21.98° (`agree_diag.json` RGBa / RGBb).
  - At matched count, d_IL selection minus colour-pair selection is −0.46° [−1.01, +0.09] (`senses/boot_sel.json`).
- **The student half of the sentence is also pre-slimming.** "Teacher labels beat plain color labels on validation but not on test" is T5A − RA: val −1.43 [−1.84, −0.82], test −0.05 [−0.56, +0.40] (`equal5/README.md` §6). No final-version (S6I) student was ever trained on colour labels.
- **"A third of the frames kept" is correct for the final version** (35.6%). Before slimming, the two-level check kept 25.4% (1,088 frames) at 14.20° vs 20.23° (`sets_diag.json`, "full").

Suggested rewrite:

> "In the final configuration the 6-network teacher label errs by 20.6° on all frames vs 21.0° for five plain colour networks (0.4° better), and by 16.5° on the 36% of frames passing the check. In an equal-count comparison before slimming (IR+LiDAR×5 vs colour×5: 20.3° vs 21.0°), students trained on teacher labels were better on validation but not on test."

## Q2. "5 seeds" in Tab. lineage: R6's contradiction claim (C27) is WRONG, but the wording is ambiguous

What the sources say:
- **Tab. 7.** The row for the first-round detector reads "pseudo boxes of the simulation-only detector (5 seeds, flip)". Here "5 seeds" describes the ensemble of the **sim-only detector** that produced the pseudo boxes.
- **`wf21/MODULES_EN.md` §1.2.**
  - Row 0: sim-only detector, "two weight sets of the same recipe, **5 seeds each**".
  - Row 1: "first-generation self-trained detector **×5**", trained from the sim-only "5-seed ensemble + flip".
  - Module 3 input (`MODULES_EN.md:186`) also says the first-generation self-trained detector has 5 seeds.
- **Files on disk.** `E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/st2_mav6d_r34_geo_full_tr/` holds `ST2_s0..s4`. `nolabel/make_nolabel_pseudo.py:8` uses the 5-model first-round ensemble predictions (`real_train_preds_st2nb.pkl`).
- **`nolabel/PREREG.md`** does not state these seed counts. It only fixes "first two seeds" for the students.

So both detectors have 5 seeds, and L359 ("first-round ... five seeds with flip") is consistent with Tab. 7.

Fix: write "(two weight sets of one recipe, 5 seeds each)" in the sim-only detector row, and "first-round detector (×5)" in the next row.

## Q3. Protocol: what the reported MAV6D run (nl075, test UDS 70.16) reused from development

Sources: `nolabel/PREREG.md`, `nolabel/run_chain_nl.sh`, `nolabel/make_nolabel_pseudo.py`, `nolabel/adabn_all_nl.py`, `nolabel/kps_nl.py`, `wf24/README.md`.

`PREREG.md` was last saved on 11-10 at 11:12, before the first label-free result (nl100 at 17:06), and that save includes the λ = 0.75 decision. "Fixed before any of its scores existed" is therefore literally true for the run itself.

However, the following parts were **reused from the development version, where test scores had been seen**:

1. **Orientation students S6I s0 and s1.**
   - Weights are reused unchanged; only BN is re-estimated (`run_chain_nl.sh:78-79`; `adabn_all_nl.py:2`; PREREG: the orientation students are not retrained).
   - S6I was adopted after comparing it with PSPM and S6R on both val and test (`wf24/README.md` §4).
   - The development chain with S6I had test AOE **13.51°** and UDS 72.48 (`wf24` §6). The reported run has 13.54° (`nolabel/chain_nl075.out:40`).
   - Their training inputs come with them:
     - teacher pseudo-orientations `OU/pseudo/pseudo_train_t6il.npz` (11-09);
     - teacher BN banks `OU/teach/teach_train.npz` (11-08);
     - crops cut by the **crop detector** (`wanb_train_kz`), whose own pseudo-labels used temporal background, a running median and the ±3-frame filter (`MODULES_EN.md` §1.2, row 3).
2. **Height network (KPS seed 0).**
   - Weights are reused; BN is re-estimated and r_k is recomputed (`kps_nl.py:2`).
   - "One network, seed 0" was decided in wf24 §2 with val and test both in view.
3. **Bootstrap networks for the position pseudo-labels** (`make_nolabel_pseudo.py:8,19-20`):
   - first-round detector ×5 (`ST2_s0..4`; predictions `st/real_train_preds_st2nb.pkl`);
   - first-generation depth refiners ×5 (`st2m/delta_IR_nb.pkl`), which are the IR teachers that correct z;
   - the sim-only detector release, used as initialisation.
4. **Refiner stage 1** `V2P_s{5,6}_st` (simulation only) is copied, not retrained (`run_chain_nl.sh:62`).
5. **Teacher-shaped orientation networks** (sim-trained) enter only through the reused S6I labels and initialisation.
6. **Constants and design choices settled with test labels in view:**
   - λ = 0.75 and γ (α) = 0.5 (PREREG, the user's final decision of 11-10);
   - the component set, each component adopted after improving both val and test;
   - the slimmed configuration (2 students, 1 height network);
   - the use of validation images for β′_k and s̄.

   By contrast, the 10° gate, mirror weight 12, 0.5 m window threshold and step counts were preset (PREREG rule 6).

**Recomputed in the run itself:**
- position pseudo-labels (λ·median, confidence ≥ 0.5);
- the detector (3 seeds);
- refiner stages 2–3 (seeds 5 and 6);
- the background bank and real crops;
- ω = 0.158, from the simulated validation set;
- the constants;
- all BN statistics.

Suggested one-sentence disclosure:

> "For the reported MAV6D run, pseudo labels, detector, depth-refiner stages 2–3, constants and ω were recomputed after the rules were fixed, but the orientation students and height network keep their development weights (batch norm re-estimated) and the bootstrap networks are reused, so their test performance had been seen during development (AOE 13.51° vs 13.54° reported); only the run as a whole was scored once."

## Q4. UDS = "UAV Detection Score", proposed by AntiUAV3D (name and ingredients only)

- **AntiUAV3D paper** (local PDF `C:/Users/jiang/Downloads/33632_AntiUAV3D_A_Large_Scale_.pdf`, p. 6, Metrics paragraph). Inspired by NDS, it proposes "the UAV Detection Score (UDS)". UDS combines mAP, normalised ATE, AOE, ASE and mean detection recall (mDR). The paper gives **no formula and no normalisers**. It matches predictions to ground truth by projected-2D IoU ≥ 0.1.
- **`E:/AntiUAV3D`** contains data only (train / val / test / trainval); there is no README and no PDF.
- **Our code.**
  - `geo1004/cmpeval/paper_table.py:6` notes that the paper gives no formula and implements UDS = ½·AP + ½·mean(1−min(1, ATE/1 m), 1−min(1, ASE/10 cm), 1−min(1, AOE/57.3°), mDR), ×100.
  - `D:/UAV3D_pack_update_1112/scripts/score.py:6,67` uses the same formula, with a = 1 m on MAV6D and a = 0.25 (relative) on AntiUAV3D.
- **Our paper.** `sec/4_experiments.tex:10` says only "UDS, a composite score in the form of nuScenes NDS that we define because no formula is published". It never expands the name and never credits AntiUAV3D for it. The bib entry `antiuav3d` reads "Anonymous, under review".
- **Prior verdict.** The same conclusion is in `D:/CVPR27_UAV/notes/review/CLAIMS_1113_verdicts.md:88-95` (Q13).
- **Transcripts.** Grepping them only turned up our own notes ("UDS (我们自定义，仿 nuScenes NDS 的形式)", i.e. "UDS (our own definition, modelled on nuScenes NDS)"); the PDF is the primary source.

Suggested wording:

> "UDS (UAV Detection Score), proposed by AntiUAV3D~\cite{antiuav3d} as an NDS-style composite of AP, ATE, AOE, ASE and mDR without a published formula; we instantiate it as ... (normalisers chosen by us)."

Caveat: the PDF is a numbered under-review copy (ID 33632). Check that it may be cited and paraphrased.

## Q5. Scorings on the AntiUAV3D v2 test split (`antifs`, 2,624 frames)

**PST chains.** Each `srcx/<run>/logs/run_x.log` has exactly one test block: the two `all` lines after "val sanity", first the sim-only detector and then fused PST. The no-fusion variant is scored only on val.

| Run | Started | Test UDS, sim-only | Test UDS, PST | Log lines |
|---|---|---|---|---|
| out2antifs | 11-12 01:58 | 30.552 | 41.735 | 373, 377 |
| out2antifs_ir | 11-12 10:11 (the 08:20 attempt restarted before any scoring) | 30.552 (same det0) | 41.839 | 269, 273 |
| out2antifs_lid | 11-12 08:24 | 30.552 (same det0) | 37.515 | 286, 290 |
| out2antifs_none | 11-12 08:28 | 30.552 (same det0) | 38.922 | 283, 287 |
| in2antifs | 11-12 02:05 | 10.460 | 15.817 | 430, 434 |
| mix2antifs | 11-13 00:50 | 23.236 | 33.254 | 448, 452 |

**Further PST-side test scorings** beyond the six chains:
- One post-hoc single-network PST deployment, `review1113/M12/anti/single/logs/run_m1_anti.log` (11-13 04:36): test UDS 42.849, AP 30.07. This is a seventh PST-side test scoring.
- Post-hoc diagnostics that re-read existing test predictions together with the test labels: `review1113/c1a_bias.py`, `c1a_decomp.py` (11-13 02:14–02:26) and `R123/r4_diag.py` (03:07).
- An unchanged re-score of out2antifs (`srcx/STATUS.md:56`).

**Our sim-only detectors.** There are 3 distinct models (outdoor, indoor, mix), scored 6 times in total. The outdoor detector was scored 4 times with identical predictions.

**Baselines.** There are 3: GUPNet, MonoCD and MonoDGP, all trained on Sim-Indoor. Their scores are in `D:/cmpfull_runs/{gupnet,monocd,monodgp}_s0/pred_outdoor_v2_test.pkl.score.json`, all dated 11-12 13:34. The other `pred_outdoor_test*.pkl` files belong to the voided v1 split. The `out2anti` and `in2anti` runs (11-11) are also on v1.

**Verdict.** R6 (C19) is right that Supp H's "five PST variants and three baselines" undercounts. The accurate count is:
- 6 PST chains, plus 1 single-network deployment;
- 3 of our sim-only detectors;
- 3 baselines.

Each was scored once, except the shared outdoor sim-only detector, which was re-scored with identical output.
