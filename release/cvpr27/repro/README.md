# repro_1117: every number of the CVPR'27 PST paper, recomputed from saved predictions

Built 11-17 (CPU only, no GPU, no process killed, nothing outside this folder modified). Paper: `D:/CVPR27_UAV` at HEAD `df1321c` (read-only).

## What is here

| file | content |
|---|---|
| `MANIFEST.md`, `manifest.json` | every cell of Tab. 1-3 and every main-text number -> prediction files (sha256, size, mtime) -> aggregation -> command |
| `rescore_all.py` | recomputes all of them from the saved prediction files with the pack scorer and compares at the printed precision |
| `rescore_report.md` | its output (mismatch list first) |
| `rescore_one.py` | score any set of prediction files (`python rescore_one.py mav|anti files...`) |
| `texparse.py` | reads the printed numbers straight from `tables/*.tex` |
| `baseline_runs.py`, `baseline_runs.json` | baseline row -> run folders (found by `work/match_baselines.py`: for every row the unique subset of saved runs whose mean reproduces all six printed cells) |
| `audit_code_times.py`, `code_time_audit.md/.json` | code files modified after the start of each reported run |
| `copy_release.py` | copies (never moves) all experiment code into `release/cvpr27/` |
| `release/cvpr27/` | the code release prepared for branch `cvpr27-paper` of `D:/Open3DUAVDet_clean_release` (see "Release branch" below) |
| `scan_preds.py`, `work/` | discovery: every candidate prediction file scored once (`work/scan_*.json`) |

Run (CPU, ~3-4 min): `D:/Miniconda3/envs/city/python.exe rescore_all.py && D:/Miniconda3/envs/city/python.exe make_manifest_md.py`

## Result of the rescore

- Table cells: **328 checked, 326 reproduced exactly** at the printed precision (Tab. 1: 276/276, Tab. 2: 34/36, Tab. 3: 16/16).
- Main-text numbers and claims: **67 checked, 66 reproduced**.
- Mismatches (paper not changed):
  1. Tab. 2, row "PST w/ detection rate of row one", AP printed 30.63, recomputed 30.36; UDS printed 54.14, recomputed 54.00.
     The caption defines the row as the most confident frames of the full pipeline, "as many as row one detects" = 3,349 frames (69.77 %, printed mDR 69.77, which matches).
     The printed AP/UDS belong to 3,350 kept frames (`table_audit_1116/raised.py`, floor(0.698 x 4800)), which gives mDR 69.79, AP 30.63, UDS 54.14. ATE/ASE/AOE are the same either way.
     So the row mixes two frame counts: either AP/UDS -> 30.36/54.00, or mDR -> 69.79.
  2. Sec. 4 text "on MAV6D it comes within 4.6 UDS of a single CenterNet trained with real labels": 65.3565 - 60.8111 = 4.545 -> **4.5** (4.6 comes from subtracting the rounded table values 65.36 - 60.81).
- Not rescored (no saved test prediction behind them, or not a test number): the dataset sizes of Sec. 4.1, the runtime 175 ms, the bootstrap intervals
  (the two "exclude zero" claims were checked against the saved bootstrap output `review1113/sensorfree/E1/pooled_{mav,antifs}_vsPST.json`:
  MAV6D [-6.67, -2.26], RealDrone3D [-3.72, -1.13] for w/o sensors - PST). Fig. 4 numbers (7.6 / 5.9 / 8.8 %, 30.1 / 18.8 deg) are recomputed from
  the CSV that `review1113/figdata/fig_pl_quality.py` wrote (training frames, post hoc), not from test predictions; all five match.

## Where the numbers come from (summary; details in MANIFEST.md)

- **Baselines (Tab. 1, five detectors)**: advisor server runs `D:/advisor_results_1117b/runs/<method>_<in|out>_<step>_<mav|anti>_s<seed>/pred_*_test.pkl`
  (byte-identical copies of most of them are also in the 11-16 delivery), plus four local runs in `D:/cmpfull_runs/`
  (`monocd_s0`, `gupnet_s0` = seed 0 of the MAV6D simulation-only rows of MonoCD and GUPNet; `monodgp_s0` = the only MAV6D simulation-only MonoDGP run;
  `gupnet_uda_dann_s0` = GUPNet +DANN on MAV6D). Steps: `sim` = simulation only, `st` = +ST, `kdc` = +PST on MAV6D (5,227 pseudo-labelled frames),
  `kd` = +PST on RealDrone3D (249 frames), `dann` = +DANN. Multi-run rows are means over up to three seeds (see `baselines/ADVISOR_RUNS.md` in the release).
- **CenterNet rows**: `review1113/ourdet_baseproto/preds/*_test.pkl` (single model, seed 0, predicted size). Real-label row on RealDrone3D = `out2antifs_ubzoom`.
- **CenterNet\***: `srcx/in2mav/preds/srconly_test.pkl` (MAV6D); `srcx/out2antifs_k249/preds/srconly_test.pkl` (RealDrone3D, byte-identical to `srcx/out2antifs/preds/srconly_test.pkl`).
- **Full pipeline**: `srcx/in2mav/preds/pcmd_fused_test.pkl` (MAV6D); `srcx/out2antifs_k249/preds/pcmd_fused_test.pkl` (RealDrone3D, frames restricted to the 249 of the baselines).
- **Tab. 2**: rows 2-3 `review1113/clean2/buildup/{b_det1,c_students}_test.pkl`, row 4 = full pipeline, row 5 `review1113/heads/H3/mav_s0/preds/h3_fused_test.pkl`, row 6 derived from row 4.
- **Tab. 3** (3 runs each, seed shift 0/10/20): PST = `srcx/in2mav`, `review1113/sup3/run`, `review1113/seeds/mav_full_s20` (MAV6D) and `srcx/out2antifs`,
  `seeds/antifs_full_s10`, `seeds/antifs_full_s20` (RealDrone3D); w/o sensors = `sensorfree/E1/*`; w/o sensors and composites = `sensorfree/E4/*`;
  RGB-only teachers = `srcx/{in2mav,out2antifs}_none` + `seeds/{mav,antifs}_none_s{10,20}`. Mean and sample sd (ddof 1) of per-run UDS.

## Code changed after the run that produced a reported number (from `code_time_audit.md`, honest list)

Per-run lib copies (`review1113/seeds/*/lib`, `sup3/lib`, `sensorfree/E*/*/lib`, `heads/H3/mav_s0/lib`, `k249/lib`) are written by the launcher at the start
of each run and never edited afterwards, so for those runs the exact code is preserved. The following are not:

1. **`srcx/out2antifs` (Tab. 3 PST run 1 on RealDrone3D, 41.73; also the source of the CenterNet\* RealDrone3D predictions and of every copied stage-1 output of the RealDrone3D seed runs and of k249)**,
   run 11-12 01:58 -> 05:04, and **`srcx/out2antifs_none` (Tab. 3 RGB-only run 1, 38.92)**, run 11-12 08:28 -> 13:31, used `srcx/lib` + `srcx/run_x.sh`, which were edited afterwards:
   - `srcx/lib/det_infer_x.py` modified 11-12 02:05:16, i.e. **during** the out2antifs run (after its `srconly` predictions 02:04:46, before the self-trained detector inference). No backup of the earlier version.
   - `srcx/lib/chain_x.py` modified 11-12 08:19:12, after out2antifs (before out2antifs_none). No backup.
   - `srcx/lib/{make_pseudo_x,ori_x,refiner_x,score_x,sx}.py` and `srcx/run_x.sh` modified 11-13 04:07-04:55 (MAV6D target added). Pre-edit copies: `*.bak_1113` (and `refiner_x.py.bak_1113w`, `sx.py.bak_1113w`, `run_x.sh.bak_1113`).
     `seeds/SEEDS_README.md` records a diff check that these edits only touch MAV6D branches (`imread_raw`, `wscale`, `real_test` rules), so the RealDrone3D path is unchanged; this is a code-reading argument, not a rerun.
2. **`review1113/ourdet_baseproto`** (Tab. 1 CenterNet rows, run 11-16 ~14:30 -> 18:29, real-label RealDrone3D row 11-17 01:36): `lib/make_labels.py` (writes the +ST/+PST label indexes)
   was modified 11-17 09:52 (backup `make_labels.py.bak_1117` = state before the 11-17 12:17 edit) and again 11-17 12:17, after all reported rows; `common.sh` 11-16 17:14
   (after the sim / +PST / +DANN rows, before +ST and the MAV6D real-label row); cfg `in2mav_st.yaml` 11-16 17:14 and `in2mav_ub.yaml` 11-16 17:14 (before those two runs finished; their training start time is not logged).
   Other files with later mtimes (`ub_zoom_*`, `build_ub_jpeg.py`, `*_pstg*.yaml`, `out2antifs_pstu.yaml`) are new files of later runs not used by the reported rows.
3. **`D:/cmpfull_runs/gupnet_s0`** (seed 0 of GUPNet simulation-only on MAV6D): `pred_real_test.pkl` written 10-30 06:57; `cmpfull/gupnet/predict.py` modified 11-10 11:27. No backup.
4. Generator scripts edited after their runs but whose per-run outputs are frozen: `review1113/sensorfree/make_sf_lib.py` (11-17 04:33, backups `.bak_1117`, `.bak_e1b`, `.bak_silw`; E1/E4 runs use their own `lib/` copies),
   `review1113/heads/make_heads_lib.py`, `heads/run_arm.sh` (11-17 04:38, backups `.bak_1117`; H3 uses `heads/H3/mav_s0/lib`).
5. Advisor server runs: the code that ran there is the pack + patches in the order listed in `release/cvpr27/baselines/ADVISOR_RUNS.md`; the server copies were not
   timestamped back to us, so a per-run code check is not possible. The patch scripts themselves record what they changed.
6. Generated configs (`tools/cfgs/.../srcx/real_<combo>.yaml`, `st_<combo>.yaml`) are rewritten by `mkcfg.py` at the start of each run (seconds after start); not a change.

## Retraining: what is exact and what is not

**Exactly reproducible (bit-identical):** every number above, from the saved prediction files, with `rescore_all.py` (deterministic CPU code; the scorer uses a stable sort for tied confidences).
Deterministic CPU steps of the chain also reproduce exactly from their saved inputs: `to_pred.py`, `make_pseudo_x.py` (pseudo-label table from saved detector/refiner outputs),
`avg_inf.py`, the fusion and deployment in `chain_x.py`, the label tables for the baselines (`cmpfull/make_pseudo_pcmd_clean.py`, `make_pseudo_pcmd_out.py`, and the table builders in the advisor patches), Tab. 2 row 6.

**Not bit-exact on retraining** (expect run-to-run spread of the size shown by Tab. 3, sd 0.3-4.4 UDS):
- GPU training uses `--cudnn_benchmark` and `--use_amp` (self-trained detectors in `run_x.sh`, stage-1 detectors), so kernels and fp16 rounding are not deterministic; MonoCD's official code sets `cudnn.benchmark = True`.
- Windows data loading: stage-1 detectors use 2 spawn workers; self-trained detectors and the review runs use `--workers 0` or the thread loader of `ourdet_baseproto/lib/thread_loader.py` (6 threads), whose random stream is not identical to the DataLoader's.
- The GPU (one RTX 5090) was shared with other jobs through `gproto/gpu_slot.sh`; some runs were resumed from checkpoints after host-memory errors (`ourdet_baseproto/README.md`), and the advisor runs ran on a different server (multi-GPU queue, one GPU per task).
- Seeds used: stage-1 simulation-only detectors 0, 1, 2 (indoor = release `ramp_s{0,1,2}.pth`; outdoor = `stage1_det.sh` seeds 0-2); teachers and RGB-only orientation nets seeds 0, 1 per group;
  chain at shift k (k = 0, 10, 20): depth refiner seeds 5+k, 6+k, self-trained detectors seeds k, k+1, k+2, orientation students seeds k, k+1 (stage-1 networks and their outputs are copied from shift 0, `COPY_FIXED=1`);
  CenterNet rows seed 0; baselines: the seed in the run folder name (`_s0/_s1/_s2`), `--fix_random_seed` where the official code supports it.

## Release branch (requested scope extension)

The code release was prepared in `release/cvpr27/` here. Creating the branch `cvpr27-paper` in `D:/Open3DUAVDet_clean_release` (git checkout -b) was **denied by the
permission system** in this session, so nothing was written to that repository. To apply it (no push, no remote involved):

```bash
bash E:/Open3DUAVDet/output/camnorm/geo1004/gproto/repro_1117/apply_to_release_branch.sh
```
