# Reviewer items R1–R4 (11-13, CPU only, read from existing files)

Scripts and raw outputs in this folder: `r1_table.py` → `r1_seeds.json`; `r2_overlap.py` → `r2_overlap.json`; `r3_bk_train.py` → `r3_results.json`; `r4_diag.py` → `r4_diag.json`, `r4_sizes.py` → `r4_sizes.json`. No GPU, no training, nothing outside this folder written.

---
## R1. "Transferring or aligning features changed accuracy by no more than seed noise"

### (a) Whole-image detector distillation (MAV6D real test, 4,800 frames, single models)
Columns: pos = median position error; det = detection rate; ADS = ADS-indoor. Δ = paired same-seed difference vs the no-transfer reference. "better" = number of seeds that improved.

No-transfer seed spread (single models, 5 seeds):
- **RAMP:** pos 30.0 cm [26.5, 34.5], det 89.1 % [88.3, 90.4], ADS 45.27 [44.64, 45.98].
- **P0** (same recipe, retrained for the IR experiments): pos 30.3 [25.8, 34.1], det 88.0 [85.9, 90.0], ADS 45.03 [44.76, 45.46].
- **ST2** (self-trained): pos 25.1 [24.4, 26.6], ADS 47.08 [46.87, 47.21].

| Variant (teacher) | ref | n | pos cm mean [min,max] | Δpos cm mean [min,max] (better) | Δdet pt mean [min,max] | ΔADS mean [min,max] (better) | source |
|---|---|---|---|---|---|---|---|
| KDir: feature+output KD (IR teacher) | RAMP | 5 | 30.9 [28.6, 33.0] | +0.95 [−2.5, +4.3] (2/5) | −0.10 [−2.0, +1.8] | −0.38 [−0.95, +0.19] (1/5) | kd/KD_SUMMARY.md, kd/json |
| MMHir: KD + hallucinated IR head (IR) | RAMP | 5 | 31.4 [30.5, 32.0] | +1.39 [−2.5, +4.7] (2/5) | −0.22 [−2.6, +2.0] | −0.30 [−0.65, +0.50] (1/5) | same |
| PIM: IR silhouette as auxiliary target (IR) | P0 | 5 | 28.9 [25.5, 34.9] | −1.41 [−6.8, +2.2] (2/5) | +0.00 [−2.8, +1.1] | +0.10 [−0.49, +0.98] (3/5) | geo1004/IRLOC_PIM_vs_P0.md, irloc_json |
| PIO: output distillation (IR teacher TIRs0) | P0 | 5 | 32.2 [28.7, 35.5] | +1.84 [−1.3, +7.0] (2/5) | +1.64 [−0.0, +3.9] | +0.20 [−0.67, +0.79] (3/5) | IRLOC_PIO_vs_P0.md |
| PIH: heatmap-only distillation (IR) | P0 | 5 | 31.2 [28.6, 36.2] | +0.87 [−1.0, +3.6] (2/5) | +0.91 [−0.5, +2.3] | −0.03 [−0.62, +0.40] (2/5) | IRLOC_PIH_vs_P0.md |
| DH: 2nd backbone imitates IR-teacher features | P0 | 5 | 32.4 [25.6, 36.7] | +2.09 [−0.2, +4.5] (1/5) | −0.07 [−2.2, +2.5] | −0.47 [−1.20, +0.19] (2/5) | DUAL_RESULTS.md, dual_json |
| DHs: same, task gradient stopped | P0 | 5 | 30.7 [28.8, 32.3] | +0.42 [−4.6, +4.5] (2/5) | −1.43 [−3.0, +2.0] | **−0.89 [−1.89, +0.26] (1/5)** | same |
| (D0: two-branch capacity control, no KD) | P0 | 5 | 32.2 [29.9, 35.3] | +1.94 [−1.1, +5.7] (2/5) | −0.10 | −0.72 [−1.87, +0.21] (1/5) | same |
| KDmm2: feat+output KD (IR+LiDAR-depth+tag teacher T6) | RAMP | 5 | 28.8 [27.5, 30.5] | −1.17 [−5.6, +4.0] (2/5) | +1.08 [−1.4, +3.7] | −0.41 [−1.27, +0.46] (2/5) | KD_SUMMARY.md |
| KDmm3: feature cosine only (T6) | RAMP | 3 | 30.5 [27.3, 35.4] | +3.06 [+0.1, +6.7] (0/3) | −0.26 [−0.4, −0.1] | −0.57 [−1.10, +0.22] (1/3) | same |
| MMH: KD + hallucinated IR/depth/tag heads (T6) | RAMP | 5 | 32.1 [26.3, 37.6] | +2.13 [−6.9, +9.5] (1/5) | +1.03 [−0.4, +2.3] | −0.58 [−1.81, +0.95] (1/5) | same |
| **MMHP**: MMH pre-training 15 ep → RAMP (T6) | RAMP | 5 | **36.2 [33.4, 38.8]** | **+6.19 [+2.2, +10.1] (0/5)** | **−11.80 [−15.7, −5.0]** | **−2.96 [−4.45, −1.75] (0/5)** | same |
| **KDF**: KDmm2 with fg-mask / centre-cosine bugs fixed (T6) | RAMP | 5 | 32.9 [29.4, 35.7] | +2.86 [−3.4, +7.9] (1/5) | +0.62 [−1.0, +1.9] | **−1.09 [−2.08, −0.43] (0/5)** | night1003/E2_SUMMARY.md, night1003/json |
| **ST2KD**: KD on top of self-training (T6) | ST2 | 5 | 24.5 [23.7, 25.5] | −0.55 [−2.5, +0.6] (3/5) | −0.19 [−0.6, −0.0] | **−1.04 [−1.17, −0.86] (0/5)** | night1003/E1_SUMMARY.md |
| KDmm v1 (with CrossKD; implementation flaw) | RAMP | 2 | 63.5 | +35.6 (0/2) | −23.6 | −9.12 (0/2) | KD_SUMMARY.md |
| KDrgb (RGB teacher; control for "distillation itself") | RAMP | 5 | 38.5 [33.6, 41.3] | +8.46 [−0.9, +14.9] (1/5) | −1.83 | −1.87 [−4.50, −0.50] (0/5) | KD_SUMMARY.md |

### (b) Feature alignment / feature-level losses
Setup: full chain, the deployed MAV6D chain of 11-02 with a 5-detector ensemble, one run each. The noise reference is the same code retrained without the change.

| Variant | split | AP | ATE m | UDS | Δ UDS vs deployed | Δ UDS vs retrain-noise control | source |
|---|---|---|---|---|---|---|---|
| deployed (no detector alignment) | val / test | 42.96 / 57.02 | .339 / .225 | 60.13 / 70.41 | – | – | geo1004/detda/compare_detda.json |
| retrain, no alignment (noise control) | val / test | 42.44 / 57.10 | .346 / .226 | 59.80 / 70.45 | −0.33 / +0.03 | – | same |
| detector whole-image adversarial alignment (GRL, w 0.1) | val / test | 42.02 / 57.06 | .345 / .222 | 59.58 / 70.48 | −0.55 / +0.06 | −0.22 / +0.03 | same |
| refiner without its built-in adversarial alignment (seeds 5–9) | val / test | 42.63 / 56.60 | .340 / .226 | 59.96 / 70.20 | −0.17 / −0.22 (AP −0.42 test) | – | geo1004/noadv/compare_noadv.json |
| detector prototype loss only (K 30, w 0.1) | val / test | 45.13 / 56.27 | .324 / .228 | 61.46 / 70.00 | +1.33 / −0.41 | +1.66 / −0.45 | geo1004/proto/compare.log |
| filter + prototype loss | val / test | 46.43 / 55.80 | .313 / .224 | 62.23 / 69.82 | +2.10 / −0.59 | +2.43 / −0.63 | same |
| consensus prototype bank (no sim term) | val / test | 41.09 / 56.78 | .360 / .224 | 58.95 / 70.29 | −1.18 / −0.12 | −0.85 / −0.16 | same |
| consensus prototype bank + sim supervision | val / test | 39.31 / 54.19 | .358 / .242 | 58.43 / 69.16 | −1.70 / **−1.25** | −1.37 / −1.29 | same |

### Verdict
- **"Never improved beyond seed noise" is supported.**
  - No transfer variant's mean position error, detection rate or ADS lies outside the no-transfer seed range in the good direction.
  - Every paired improvement is at most 4/5 seeds, with a range crossing 0.
  - The best are PIO detection +1.6 pt (4/5, t 2.5) and KDmm2 position −1.2 cm (2/5). Both stay inside the reference range.
  - Detector adversarial alignment is +0.03 test UDS vs the retrain control. The retrain itself moved by 0.03 on test and 0.33 on val.
- **"Changed accuracy by no more than seed noise" is NOT supported as written.** Several variants made results consistently worse, by more than the seed spread:
  - **MMHP:** position +6.2 cm, detection −11.8 pt, ADS −2.96 (0/5). Its mean is outside the RAMP range on all three metrics.
  - **KDF:** ADS −1.09 (0/5). Its 44.18 is below the RAMP minimum of 44.64.
  - **ST2KD:** ADS −1.04 (0/5). Its 46.04 is below the ST2 minimum of 46.87.
  - **DHs:** ADS 44.14, below the P0 minimum of 44.76. The two-branch control D0 is also below that range, so this is partly an architecture cost.
  - **Prototype bank with sim supervision:** test UDS −1.25.
  - KDmm v1 (CrossKD) and KDrgb were also harmful, but one is a bug and the other a control.
- **Prototype-loss variants are within noise on test only.** They raised val UDS well beyond noise (+1.3 to +2.1) while test fell (−0.4 to −0.6).
- **Suggested wording:** "…whereas transferring or aligning features inside the detector never improved accuracy beyond seed noise, and several variants reduced it (Supp. Tab. X)."
- **Which are the "six variants"? The paper should list them by name.**
  - The disk has **seven** IR-teacher whole-image variants: KDir, MMHir, PIM, PIO, PIH, DH, DHs.
  - It has **six** multimodal-teacher variants (excluding the CrossKD bug run): KDmm2, KDmm3, MMH, MMHP, KDF, ST2KD.

---
## R2. AntiUAV3D old (v1, sequence-only) vs new (v2, focal-stratified) splits

Frame lists:
- v1: `D:/S2MUAV3D/real_outdoor_antiuav3d/labels/{train,val,test}_frames.txt` (3,361 / 1,104 / 2,825 frames).
- v2: `D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/{train,val,test}_frames.txt` (3,293 / 1,148 / 2,624 frames).

Sequences per split:
- **v1:**
  - train: `train_seq_` 0002, 0009, 0011, 0015, 0016, 0018, 0019, 0020, 0022
  - val: `train_seq_0010`
  - test: `test_seq_0003`, `0005`, `0006`
- **v2:**
  - train: `test_seq_0006` plus `train_seq_` 0002, 0011, 0016, 0018, 0020
  - val: `test_seq_0003`, `0005`
  - test: `train_seq_0010`, `train_seq_0022`

| v1 split × v2 split | shared sequences | identical frames | v1 frames in shared seqs | v2 frames in shared seqs | v2 frames within ±3 frame ids of a v1 frame |
|---|---|---|---|---|---|
| test × test | none | 0 | 0 | 0 | 0 |
| test × val | test_seq_0003, 0005 | 575 | 1,809 / 2,825 | 1,148 / 1,148 (100 %) | 1,148 |
| test × train | test_seq_0006 | 0 | 1,016 / 2,825 | 131 / 3,293 | 131 |
| val × test | train_seq_0010 | 368 | 1,104 / 1,104 | 2,576 / 2,624 (98.2 %) | 2,557 |
| train × test | train_seq_0022 | 7 | 59 / 3,361 | 48 / 2,624 (1.8 %) | 48 |
| train × train | 0002, 0011, 0016, 0018, 0020 | 450 | 2,574 / 3,361 | 3,162 / 3,293 | 3,139 |
| train × val, val × train, val × val | none | 0 | 0 | 0 | 0 |

- **Old test → new splits:** 64 % of v1-test frames went to v2 val (2 sequences) and 36 % to v2 train (`test_seq_0006`). **0 % went to v2 test.**
- **New test ← old splits:** 98.2 % of v2-test frames come from **v1 val** (`train_seq_0010`). Of these, 368 frames are identical and 2,557 are within ±3 frame ids. The other 1.8 % come from v1 train (`train_seq_0022`).

Points to check before answering:
1. **The v2 test sequence was the v1 validation sequence.** `train_seq_0010` labels were read for val sanity scores in v1-era runs: `srcx/lib/score_x.score_val` reads `sx.T['val_labels']`, which is v1 `val.pkl` for target `anti`. So this sequence was "seen during v1 development", but not trained on: v1 training used only unlabeled train images. Whether any v1 val score drove a design choice cannot be told from the files.
2. **Other v2 splits were also v1 test data.** The v2 train images include `test_seq_0006`, and v2 val consists entirely of v1-test sequences. All of these were scored with labels on v1, but only as test data.
3. **`D:/cmpfull/outdoor_val` is stale.** It (1,104 frames, dated 11-10) is still the **v1** val list, i.e. the v2 test sequence. `outdoor_train` and `outdoor_test` were updated and are identical to the v2 lists; `_antiuav3d_v1/outdoor_test` is v1 test.
   - I found no reference to `outdoor_val` in the scripts or configs of `D:/UAV3D_pack_update_1112` or `D:/cmpfull_runs`.
   - Still, anyone validating a v2 baseline on it would be validating on the test sequence. Rename or remove it.

---
## R3. Deployment constants from unlabeled TRAIN images instead of VAL images (MAV6D, label-free chain nl075)

Which constants come from where (from `nolabel/PREREG.md` rule 7 and `results_nl075.json`):
- **Already from unlabeled train images:** colour centres, c_k and r_k (`FE_CONST=train`; consts.src = "train").
- **From val images, only two constants:**
  - the deployment offsets **b_k**;
  - the constant size **s̄**. This is the detector's self-calibration `--self-calib val`: [0.3497, 0.3485, 0.1996] from 890 val boxes with conf ≥ 0.7 (`nl075/logs/dumpA.log`).

### b_k: done on CPU
- **Reproduction:** `r3_bk_train.py` reproduces the paper's chain exactly: test 56.44 / 0.226 / 1.81 / 13.54 / 70.16, val 62.49.
- **Definition:** b_k from train images uses the same definition as deployment, the median of log(z_kp / z_fin) per colour prototype.
  - z_fin: deployed stage-3 refiner on train frames (`nl075/unc/unc_train_z2.pkl` + `recs/nl075_train.pkl`).
  - z_kp: the same 4-member SA-TM/PnP on train frames (`ztrain_nl075_train1_ouS6I2M.pkl`).

| b_k source | b_k (dark / light) | n frames | val AP / ATE / UDS | test AP | test ATE m | test ASE | test AOE | test mDR | **test UDS** | test pos. median m |
|---|---|---|---|---|---|---|---|---|---|---|
| val images (paper) | 0.4829 / 0.3756 | 1,299 / 1,037 | 48.33 / .312 / 62.49 | 56.44 | 0.226 | 1.81 | 13.54 | 99.8 | **70.16** | 0.138 |
| train images | 0.4492 / 0.2971 | 7,427 / 5,438 | 48.59 / .312 / 62.62 | 54.40 | 0.235 | 1.81 | 13.54 | 99.8 | **69.03** | 0.148 |

- **Effect:** test UDS −1.13, AP −2.04, ATE +9 mm; val +0.13.
- This agrees with the supplement's "+1.15 (version before slimming)".
- AOE, ASE and mDR are unchanged, as expected: b_k only moves distance.

### s̄: needs GPU
- **Why it is missing:** the detector records on disk already carry the val-calibrated constant size (SIZE_SOURCE known). The size head's raw per-frame lwh on train images was never saved.
- **Missing input:** raw size-head predictions of the nl075 3-model + flip ensemble on train images (interval 2, conf ≥ 0.7).
- **Estimated cost:**
  - The inference alone is about 5–10 min of GPU. For scale, the train dump of 12,866 frames plus val self-calibration took 18 min (`nl075/logs/driver.log`).
  - s̄ sets the detector's decoded depth, so a deployment-only swap also needs re-inference: val/test detector, orientation windows/recut and z_kp, refiner outputs, height readings. That is about 0.5–1 h of GPU, with no training.
  - A fully consistent swap also changes the train pseudo-labels, because jobA used `--self-calib val`. That means retraining the detector and refiners. The previous chain took about 2.6 h after detector training (17:51→20:25), plus 3 detector seeds.

---
## R4. Why our Sim-Indoor detector has mDR 4.9 % on AntiUAV3D v2 while baselines reach 76–92 %

The metric is the same for everyone: `D:/UAV3D_pack_update_1112/scripts/score.py`, the same metric code as the pack used by `srcx/lib/score_x.py`.
- Only the top-scored box per frame counts.
- mDR = fraction of GT frames whose top box's *projected 2D box* has IoU ≥ 0.1 with the GT projected box, at any depth.
- **Ours:** `srcx/in2antifs/preds/srconly_test.pkl.score_antifs_test.json`: mDR 4.88, ATE 54.6 %, AOE 25.2°.
- **Baselines:** GUPNet 78.4, MonoCD 75.9, MonoDGP 91.7 (`D:/cmpfull_runs/<m>_s0/pred_outdoor_v2_test.pkl.score.json`).

### Ruled out
- **Confidence gate:** there is none. `det_infer_x.py` keeps one box per frame with no threshold, and 100 % of frames have a box.
  - The 0.7 threshold is used only for s̄ self-calibration. That fell back to the top-57 val boxes (conf ≥ 0.44), giving s̄ = 0.25 × 0.28 × 0.08 m.
- **One box per frame:** the baselines output 10 boxes but are also scored on the top one only.
- **mDR definition:** whenever our top box is near the target (centre within half a GT width), IoU ≥ 0.1 holds 100 % of the time (128/128 on test, 16/16 on val).
- **Zoom-crop handling:** the detector runs on the plain resized full image (1920×1080 → 512×288, `build_cache_fs.py`). Zoom crops exist only in the orientation branch, after detection.

### Actual cause: the indoor-source detector fails to localise in 2D
Its top box lies on the background.

| detector (top box) | split | mDR | top-box centre within 0.5 GT width | median centre error (px at 1920) | conf median |
|---|---|---|---|---|---|
| ours, Sim-Indoor | test | 4.9 % | 4.9 % | 606 | 0.23 |
| ours, Sim-Indoor | val (unsealed labels) | 1.4 % | 1.4 % | 647 | 0.27 |
| ours, Sim-Outdoor (same code, same 512 input) | test | 78.5 % | 78.5 % | 6 | 0.47 |
| GUPNet / MonoCD / MonoDGP, Sim-Indoor | test | 78.4 / 75.9 / 91.7 % | 78.1 / 76.0 / 91.0 % | 4 / 4 / 2 | – |

- When it is wrong, the box is about 8× too wide (median predicted/GT 2D width 8.1) and about 12× too close (z ratio 0.08). The detector fires on large image structures that resemble a near indoor drone.
- No val box reaches conf 0.7.
- By GT 2D width (at 1920), the share localised is 0 % at 20–40 px, 1.5 % at 40–80 px and 13.5 % above 80 px.
- The baselines' test predictions were scored against the sealed test meta for this diagnosis only; nothing was selected with it.

### Contributing factors (read from files, not tested causally)
1. **Input scale.**
   - The baselines run at 960×540 (`cmpfull/SPEC.md`); we run at 512×288.
   - AntiUAV3D v2 targets have a median 2D width of 12.6 px at 512 (5–95 %: 6.0–19.6; 72 % are below 16 px), versus 23.6 px at 960.
   - Indoor-sim training targets have a median of 32 px at 512 (5–95 %: 10.6–114). Indoor sim depth is 2.5–7.5 m (5th–95th percentile).
   - So at our resolution the real targets fall in the bottom ~20 % of what the indoor detector saw (`r4_sizes.json`).
2. **Source content.** At the same 512 input, the outdoor-sim detector localises 78.5 %, so small size alone is not fatal. What fails is the indoor source: near range, indoor clutter, no sky.
3. **mDR says nothing about 3D accuracy.** It only asks for 2D overlap of the projected box.
   - The baselines find the drone in 2D, at twice our resolution, but miss its distance by 43–63 %: AP = 0, predicted z / true z = 0.35–0.56.
   - Our indoor-source PST row also has AP = 0, but its failure is upstream (wrong 2D location). Self-training therefore has almost nothing to bootstrap from: mDR is 17.6 % after PST.

### Suggested text
*"Trained on Sim-Indoor and applied to the full 512×288 image, our detector's top box lies on the background in 95 % of AntiUAV3D frames (median offset 606 px at 1920 width; with the same code the Sim-Outdoor detector localises 78.5 %). The full-image baselines run at 960×540 and localise the drone in 2D (mDR 76–92 %) but still miss its distance by 43–63 % (AP 0). mDR counts 2D overlap of the projected box only."*
