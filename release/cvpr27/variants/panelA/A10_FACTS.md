# A10. AntiUAV3D v2 test split: every scoring, and every read of the test labels

Compiled 2026-11-14 ~03:20 machine time. The machine clock runs 13 h ahead of the user's Michigan time. Read-only: no scorer was re-run for this list.

**Sources.**
- Chain logs: `srcx/<combo>/logs/run_x.log` ("ALL DONE" lines).
- `cmpfull/run1112/pred_antifs.log` and `cmpfull/run1110/monocd_out_sim_backup.log`.
- `review1113/seeds/logs/run_all_seeds.log`.
- File mtimes of every `*.score_antifs_test.json` and `pred_outdoor_v2_test.pkl.score.json`. Search roots: E:/Open3DUAVDet/output/camnorm/geo1004, D:/S2MUAV3D, D:/cmpfull_runs, D:/UAV3D_baselines_pack, D:/UAV3D_pack_update_1112.

**Test split.** 2,624 frames in 2 sequences:
- `train_seq_0010`: Matrice 4E, 2,576 frames;
- `train_seq_0022`: Matrice 400, 48 frames.

**Scorers.** Two code paths:
- `srcx/lib/score_x.py` → `score_meta()`, reading the GT file `labels/sealed/antifs_test_meta.npz`, then the pack's `score.py::metrics` with the `outdoor_test` rules;
- for baselines, `D:/UAV3D_pack_update_1112/scripts/score.py::score(…, 'outdoor_test')`, reading its own copy of the GT, `data/gt/outdoor_test_meta.npz`. That copy was written on 11-12 at 07:39 by `make_pack_update.py`, which asserts it is identical to the sealed meta.

## 1. Every variant scored on the v2 test split, in chronological order

| # | Machine time (11-xx hh:mm) | Variant | Prediction file | UDS | AP | Timestamp source |
|---|---|---|---|---|---|---|
| 1 | 12 05:04:04 | Sim-only detector, outdoor source (det0: 3 seeds + flip) | `srcx/out2antifs/preds/srconly_test.pkl` | 30.55 | 16.04 | run_x.log "ALL DONE 05:04:04"; json mtime |
| 2 | 12 05:04:06 | **PST, outdoor source (reported)** | `srcx/out2antifs/preds/pcmd_fused_test.pkl` | **41.74** | **27.89** | same |
| 3 | 12 05:33:10 | Sim-only detector, indoor source | `srcx/in2antifs/preds/srconly_test.pkl` | 10.46 | 0.00 | in2antifs run_x.log; mtime |
| 4 | 12 05:33:12 | PST, indoor source | `srcx/in2antifs/preds/pcmd_fused_test.pkl` | 15.82 | 0.00 | same |
| 5 | 12 10:55:54 | Teachers = IR-distilled only (reuses #1 det0, re-scored) | `srcx/out2antifs_ir/preds/{srconly,pcmd_fused}_test.pkl` | 30.55 / 41.84 | 16.04 / 28.08 | out2antifs_ir run_x.log 10:55:53 |
| 6 | 12 11:07:00 | Teachers = LiDAR only | `srcx/out2antifs_lid/preds/{srconly,pcmd_fused}_test.pkl` | 30.55 / 37.52 | 16.04 / 21.31 | out2antifs_lid run_x.log 11:07:00 |
| 7 | 12 13:31:48 | **RGB-only teachers (reported control)** | `srcx/out2antifs_none/preds/{srconly,pcmd_fused}_test.pkl` | 30.55 / **38.92** | 16.04 / 23.39 | out2antifs_none run_x.log 13:31:47 |
| 8 | 12 13:34:37 | Baselines trained on Sim-Indoor (seed 0, local): GUPNet / MonoCD / MonoDGP | `D:/cmpfull_runs/{gupnet,monocd,monodgp}_s0/pred_outdoor_v2_test.pkl` | 17.25 / 18.48 / 20.44 | 0 / 0 / 0 | score.json mtimes 13:34:36–39. These were scored by hand after the automatic step in `pred_antifs.sh` failed at 08:18–08:28 with a path error (pred_antifs.log) |
| 9 | 13 03:57:46 | Sim-only detector, mixed source | `srcx/mix2antifs/preds/srconly_test.pkl` | 23.24 | 5.17 | mix2antifs run_x.log 03:57:46 |
| 10 | 13 03:57:48 | PST, mixed source | `srcx/mix2antifs/preds/pcmd_fused_test.pkl` | 33.25 | 15.57 | same |
| 11 | 13 04:11:08 | M12: reproduction of #2 (bit-identical predictions) | `review1113/M12/anti/repro/preds/pcmd_fused_test.pkl` | 41.74 | 27.89 | json mtime |
| 12 | 13 04:36:50 | M12: PST single model, no TTA | `review1113/M12/anti/single/preds/single_fused_test.pkl` | 42.85 | 30.07 | json mtime; run_m1_anti.log |
| 13 | 13 07:43 | Height-source swap on #2: h := 0.4433·(l+w)/2 (a new variant, test only) | `review1113/height` (in memory) | see HEIGHT_REPORT §2 | = #2 | height_results.json mtime |
| 14 | 13 15:32:24 | Baseline MonoCD trained on outdoor sim (seed 0, local) | `D:/cmpfull_runs/monocd_out_sim_s0/pred_outdoor_v2_test.pkl` | 10.62 | 0.00 | score.json mtime. The scripted scoring at 15:17:08 failed with a path error (monocd_out_sim_backup.log) |
| 15 | 14 02:57 | cpu5 item 5: rows (b) det1, (c) + students, (c′) + r_k and (d) pcmd_nofuse of out2antifs, all built from saved files | `review1113/cpu5/anti_*_test.pkl`, `srcx/out2antifs/preds/pcmd_nofuse_test.pkl` | 37.58 / 38.66 / 38.66 / 41.69 | 23.49 / 23.49 / 23.49 / 27.83 | cpu5.json mtime. This is the first scoring of `pcmd_nofuse_test` on AntiUAV3D |
| 16 | 14 03:14:48 | Seed run `antifs_full_s10` (PST, seed shift 10): sim-only + PST | `review1113/seeds/antifs_full_s10/preds/{srconly,pcmd_fused}_test.pkl` | 30.55 / 41.38 | 16.04 / 27.25 | run_all_seeds.log 03:14:47 |
| – | not finished | `antifs_none_s10` (launched 14 03:14:47; only `srconly_*` written so far), `antifs_full_s20` and `antifs_none_s20` (not started) | – | – | – | run_all_seeds.log |

**Not scored locally.** The AntiUAV3D baseline numbers in `tables/tab_main.tex` / `4_experiments.tex` came from the advisor's server run of the pack update, which shipped the sealed GT copy. These are the outdoor-source baselines, +ST and +PL, e.g. MonoDGP 20.80 / 27.78 / 29.43 and GUPNet 14.61 / 30.88. No local score file for them exists.

**Re-scores of existing predictions only** (no new variant): C1 (`c1a_bias.py`, 13 02:26), A3 (13 08:52), breakdown (13 15:42), orient_dist (13 23:45) and cpu5 items 4–5 (14 02:57). These all assert equality with #1, #2, #7 and #12. The 11-13 STATUS smoke test also re-scored #2 unchanged (41.735).

**Not on this split.** The v1 runs `srcx/out2anti` and `srcx/in2anti` were scored on the v1 test (`score_outdoor_test.json`, 11-11 10:34 and 23:22). v1 test and v2 test share 0 frames (R123 R2).

## 2. What was read from the test labels, and when

| When (machine) | What | Test-label fields used | Used to choose anything? |
|---|---|---|---|
| 12 01:28–01:54 | Building the v2 split, `make_subset_fs.py` | Pool = frames with GT camera distance < 80 m. The sequence-assignment score includes the near/mid/far share from GT z (`dist_bin(annos box z)`) and requires M400 in both train and test. The script's own text says focal bands and frame counts are the primary criterion and "不看任何真值" (no GT is looked at). Line 47, however, admits the distance-band term and the M400 constraint at dataset-construction level | **Yes, the split design only.** GT distance and airframe identity shaped which sequences became test. This matches the paper's "AntiUAV3D labels only built its split" |
| 12 01:53 | `kverify.py` / `check_zoom_sync.py` | No box fields: images and intrinsics only (beyond the < 80 m pool) | Frame inclusion, from images and K only |
| 12 01:54:58 | `seal_and_check.py`: moved train/test GT to `labels/sealed/{train,test}_gt.{pkl,json}` and checked for duplicates | `box_official` + K (identical-label check); `dist_m` printed for the 5 most similar test/train image pairs; `_dupcheck_top.npy` (01:55:37) = indices of the top-20 similar pairs | No; leakage check only |
| 12 01:57:09 | `labels/sealed/antifs_test_meta.npz` written (the scorer meta: seq, frame, gt box, K, cls). No script on disk writes this file name, so it was presumably an inline command | full test GT, repackaged | – |
| 12 07:39 | `make_pack_update.py` copied the meta into the advisor pack (`data/gt/outdoor_test_meta.npz`) | asserted that seq/frame/K match | – |
| 12 05:04 → 14 03:14 | Scorer reads: #1–#16 above | full metrics | See the note below |
| 13 02:14–02:26 | C1 (`c1a_bias.py`, `c1a_decomp.py`) | per-frame signed distance bias, size bias, size_px / w2d decomposition, by aircraft and distance band | No (diagnostic) |
| 13 03:07 | R123 `r4_diag.py` | detection rate / localisation of our sim-only detectors and of the local GUPNet/MonoCD/MonoDGP test predictions (why indoor-source mDR = 4.9 %) | No |
| 13 06:54 | N2 `n2_chance.py` | chance / constant orientation baselines, including a **best constant fitted on the test split** ("oracle", reported as a reference row) | No; it is labelled oracle |
| 13 07:43 | height `height_eval.py` | per-axis size errors of #2, the constant-aspect swap, and the IR-only / RGB-only arms | No |
| 13 08:52 | A3 `a_block_boot.py` | per-frame errors; ACF of the errors, used to set the block length L = 176 | **Block length only.** L was chosen from the test-error autocorrelation (an inference setting, not a method choice) |
| 13 12:42 | clean3 `select_qual.py` | per-frame position and AOE percentiles, used to pick Fig. 3 frames (rule fixed in code) | Figure examples only |
| 13 15:42 | breakdown `per_aircraft.py` | per-aircraft subset metas `breakdown/meta/antifs_test_{M4E,M400}.npz` (written there) | No |
| 13 23:45 | orient_dist `orient_dist.py` | AOE-90/180, heading, GT-range bands (C2) | No |
| 14 02:57 | cpu5 `cpu5.py` | items 4–5 (single vs ensemble; distance-source rows) | No |
| 14 ~03:00 → now | panelA (this review: A6 band interaction / block lengths, A11 size prior, `panelA/a11/a11_size.py`) | GT size, range, per-frame errors | No (diagnostic) |

**Development history of the test sequences** (R123 R2).
- 98.2 % of the v2 test frames come from `train_seq_0010`, which was the **v1 validation sequence**. Its labels were read by v1-era val-sanity scoring (`score_x.score_val` on v1 `val.pkl`).
- The remaining 1.8 % (`train_seq_0022`) were v1 unlabeled train images.
- So the v2 test sequence was not trained on, but it was seen with labels during v1 development.

**Choices.**
- Each report states that its test results were not used to choose anything: the chain logs ("ONE test scoring at the end", run_x.sh:8), M12, C1, A3, height, clean3, cpu5 and orient_dist.
- The reported row (outdoor source, `out2antifs`) is the method's source by design (outdoor sim → outdoor target). The ir / lid / none / indoor / mix runs are ablations of it.
- Even so, 7 chain variants and 4 local baselines were scored on this test split, plus diagnostics. These should be disclosed as test-time ablations, not as one-shot results:
  - chain variants: out, ir, lid, none, in, mix, and M12 single;
  - local baselines: GUPNet, MonoCD, MonoDGP (Sim-Indoor) and MonoCD (outdoor sim).
- The one place a test-derived quantity set an analysis parameter is A3's block length (L = 176, from the test-error ACF).
