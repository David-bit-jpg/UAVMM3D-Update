# MANIFEST: every number of Tab. 1-3 (and the main-text numbers) -> prediction files -> aggregation -> command

Generated 2026-11-17 23:16:23 from `manifest.json` (written by `rescore_all.py`). Paper: D:/CVPR27_UAV (read-only).

Prefixes: `A/` = D:/advisor_results_1117b/runs/ (advisor server, 11-17 delivery), `L/` = D:/cmpfull_runs/ (local baseline runs),
`X/` = E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/ (clean chain), `R/` = E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ (variants).

Scorer `D:/UAV3D_baselines_pack/scripts/score.py` (sha256 904b8189d4dce213), most confident box per frame, `S.metrics(..., np.arange(N), S.CFG[split])`.
GT: mav = `D:/UAV3D_baselines_pack/data/gt/real_test_meta.npz` (sha256 0bc3cdd174cf5b70); anti = `D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz` (sha256 dd05225e116f41ad)

Commands (CPU only, `D:/Miniconda3/envs/city/python.exe`):
- everything: `python rescore_all.py` -> `rescore_report.md`, `manifest.json`
- one cell group: `python rescore_one.py <mav|anti> <pred files of the row>` (mean over the listed runs; RealDrone3D ATE in %)

Aggregation: "single run" = the metric of that run; "mean over n runs" = arithmetic mean of the per-run metrics (so UDS is averaged per run);
Tab. 3 = mean and sample standard deviation (ddof = 1) of the per-run UDS of three runs (seed shift 0, 10, 20).

## Table cells (grouped by row)

| table | row | target | printed AP / ATE / ASE / AOE / mDR / UDS | ok | aggregation | prediction files |
|---|---|---|---|---|---|---|
| Tab. 1 | MonoDETR | MAV6D | 1.98 / 1.091 / 6.99 / 20.23 / 78.71 / 22.67 | all | single run | `A/monodetr_in_sim_s0/pred_real_test.pkl` |
| Tab. 1 | MonoDETR | RealDrone3D | 0.00 / 48.5 / 16.8 / 23.3 / 64.10 / 15.44 | all | single run | `A/monodetr_out_sim_s0/pred_outdoor_test.pkl` |
| Tab. 1 | MonoDETR +ST | MAV6D | 1.89 / 0.991 / 3.90 / 18.54 / 68.81 / 25.74 | all | single run | `A/monodetr_in_st_mav_s0/pred_real_test.pkl` |
| Tab. 1 | MonoDETR +ST | RealDrone3D | 0.02 / 41.6 / 14.0 / 26.1 / 67.42 / 15.24 | all | single run | `A/monodetr_out_st_anti_s0/pred_outdoor_test.pkl` |
| Tab. 1 | MonoDETR +PST | MAV6D | 40.97 / 0.304 / 2.51 / 16.52 / 96.14 / 59.45 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monodetr_in_kdc_mav_s0/pred_real_test.pkl`<br>`A/monodetr_in_kdc_mav_s1/pred_real_test.pkl`<br>`A/monodetr_in_kdc_mav_s2/pred_real_test.pkl` |
| Tab. 1 | MonoDETR +PST | RealDrone3D | 0.78 / 45.7 / 16.8 / 20.6 / 81.08 / 18.54 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monodetr_out_kd_anti_s0/pred_outdoor_test.pkl`<br>`A/monodetr_out_kd_anti_s1/pred_outdoor_test.pkl`<br>`A/monodetr_out_kd_anti_s2/pred_outdoor_test.pkl` |
| Tab. 1 | MonoCD | MAV6D | 3.68 / 1.013 / 8.58 / 23.83 / 41.64 / 18.55 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monocd_in_sim_s1/pred_real_test.pkl`<br>`A/monocd_in_sim_s2/pred_real_test.pkl`<br>`L/monocd_s0/pred_real_test.pkl` |
| Tab. 1 | MonoCD | RealDrone3D | 1.02 / 35.8 / 12.6 / 22.4 / 37.89 / 12.86 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monocd_out_sim_s0/pred_outdoor_test.pkl`<br>`A/monocd_out_sim_s1/pred_outdoor_test.pkl`<br>`A/monocd_out_sim_s2/pred_outdoor_test.pkl` |
| Tab. 1 | MonoCD +ST | MAV6D | 0.82 / 0.787 / 5.54 / 20.29 / 18.12 / 18.99 | all | single run | `A/monocd_in_st_mav_s1/pred_real_test.pkl` |
| Tab. 1 | MonoCD +ST | RealDrone3D | 0.29 / 44.9 / 17.8 / 18.7 / 22.80 / 11.41 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monocd_out_st_anti_s0/pred_outdoor_test.pkl`<br>`A/monocd_out_st_anti_s1/pred_outdoor_test.pkl`<br>`A/monocd_out_st_anti_s2/pred_outdoor_test.pkl` |
| Tab. 1 | MonoCD +PST | MAV6D | 39.04 / 0.309 / 2.45 / 18.12 / 95.47 / 58.08 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monocd_in_kdc_mav_s0/pred_real_test.pkl`<br>`A/monocd_in_kdc_mav_s1/pred_real_test.pkl`<br>`A/monocd_in_kdc_mav_s2/pred_real_test.pkl` |
| Tab. 1 | MonoCD +PST | RealDrone3D | 13.66 / 13.5 / 14.1 / 21.3 / 81.33 / 30.61 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monocd_out_kd_anti_s0/pred_outdoor_test.pkl`<br>`A/monocd_out_kd_anti_s1/pred_outdoor_test.pkl`<br>`A/monocd_out_kd_anti_s2/pred_outdoor_test.pkl` |
| Tab. 1 | MonoDGP | MAV6D | 25.15 / 0.544 / 5.15 / 16.55 / 81.60 / 43.43 | all | single run | `L/monodgp_s0/pred_real_test.pkl` |
| Tab. 1 | MonoDGP | RealDrone3D | 2.76 / 26.6 / 15.9 / 27.9 / 88.36 / 19.22 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monodgp_out_sim_s0/pred_outdoor_test.pkl`<br>`A/monodgp_out_sim_s1/pred_outdoor_test.pkl`<br>`A/monodgp_out_sim_s2/pred_outdoor_test.pkl` |
| Tab. 1 | MonoDGP +ST | MAV6D | 25.01 / 0.463 / 3.79 / 15.25 / 83.06 / 46.53 | all | single run | `A/monodgp_in_st_mav_s0/pred_real_test.pkl` |
| Tab. 1 | MonoDGP +ST | RealDrone3D | 6.91 / 19.8 / 12.5 / 25.4 / 88.36 / 24.08 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monodgp_out_st_anti_s0/pred_outdoor_test.pkl`<br>`A/monodgp_out_st_anti_s1/pred_outdoor_test.pkl`<br>`A/monodgp_out_st_anti_s2/pred_outdoor_test.pkl` |
| Tab. 1 | MonoDGP +PST | MAV6D | 42.68 / 0.318 / 2.50 / 16.67 / 99.47 / 60.55 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monodgp_in_kdc_mav_s0/pred_real_test.pkl`<br>`A/monodgp_in_kdc_mav_s1/pred_real_test.pkl`<br>`A/monodgp_in_kdc_mav_s2/pred_real_test.pkl` |
| Tab. 1 | MonoDGP +PST | RealDrone3D | 11.85 / 16.7 / 15.8 / 20.6 / 93.41 / 29.75 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monodgp_out_kd_anti_s0/pred_outdoor_test.pkl`<br>`A/monodgp_out_kd_anti_s1/pred_outdoor_test.pkl`<br>`A/monodgp_out_kd_anti_s2/pred_outdoor_test.pkl` |
| Tab. 1 | MonoCoP | MAV6D | 27.18 / 0.514 / 4.15 / 15.46 / 84.56 / 46.68 | all | single run | `A/monocop_in_sim_s0/pred_real_test.pkl` |
| Tab. 1 | MonoCoP | RealDrone3D | 1.49 / 25.4 / 12.5 / 25.3 / 91.39 / 19.14 | all | single run | `A/monocop_out_sim_s0/pred_outdoor_test.pkl` |
| Tab. 1 | MonoCoP +ST | MAV6D | 26.79 / 0.438 / 3.47 / 14.51 / 86.40 / 48.71 | all | single run | `A/monocop_in_st_mav_s0/pred_real_test.pkl` |
| Tab. 1 | MonoCoP +ST | RealDrone3D | 3.39 / 24.9 / 11.7 / 25.2 / 95.16 / 20.63 | all | single run | `A/monocop_out_st_anti_s0/pred_outdoor_test.pkl` |
| Tab. 1 | MonoCoP +PST | MAV6D | 44.60 / 0.276 / 2.60 / 16.79 / 99.71 / 61.90 | all | single run | `A/monocop_in_kdc_mav_s0/pred_real_test.pkl` |
| Tab. 1 | MonoCoP +PST | RealDrone3D | 10.80 / 16.2 / 16.0 / 19.8 / 90.14 / 29.25 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/monocop_out_kd_anti_s0/pred_outdoor_test.pkl`<br>`A/monocop_out_kd_anti_s1/pred_outdoor_test.pkl`<br>`A/monocop_out_kd_anti_s2/pred_outdoor_test.pkl` |
| Tab. 1 | GUPNet | MAV6D | 6.05 / 1.418 / 14.64 / 23.61 / 65.34 / 18.54 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/gupnet_in_sim_s1/pred_real_test.pkl`<br>`A/gupnet_in_sim_s2/pred_real_test.pkl`<br>`L/gupnet_s0/pred_real_test.pkl` |
| Tab. 1 | GUPNet | RealDrone3D | 0.67 / 30.7 / 11.4 / 30.3 / 48.22 / 12.26 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/gupnet_out_sim_s0/pred_outdoor_test.pkl`<br>`A/gupnet_out_sim_s1/pred_outdoor_test.pkl`<br>`A/gupnet_out_sim_s2/pred_outdoor_test.pkl` |
| Tab. 1 | GUPNet +DANN | MAV6D | 10.47 / 1.316 / 13.46 / 21.88 / 72.71 / 22.05 | all | single run | `L/gupnet_uda_dann_s0/pred_real_test.pkl` |
| Tab. 1 | GUPNet +DANN | RealDrone3D | 0.28 / 32.0 / 11.5 / 27.0 / 60.67 / 14.34 | all | single run | `A/gupnet_out_dann_anti_s0/pred_outdoor_test.pkl` |
| Tab. 1 | GUPNet +ST | MAV6D | 7.79 / 0.861 / 8.84 / 19.44 / 46.81 / 21.20 | all | single run | `A/gupnet_in_st_mav_s0/pred_real_test.pkl` |
| Tab. 1 | GUPNet +ST | RealDrone3D | 0.02 / 25.3 / 11.3 / 29.0 / 27.90 / 10.10 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/gupnet_out_st_anti_s0/pred_outdoor_test.pkl`<br>`A/gupnet_out_st_anti_s1/pred_outdoor_test.pkl`<br>`A/gupnet_out_st_anti_s2/pred_outdoor_test.pkl` |
| Tab. 1 | GUPNet +PST | MAV6D | 38.91 / 0.321 / 2.80 / 16.92 / 94.96 / 57.61 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/gupnet_in_kdc_mav_s0/pred_real_test.pkl`<br>`A/gupnet_in_kdc_mav_s1/pred_real_test.pkl`<br>`A/gupnet_in_kdc_mav_s2/pred_real_test.pkl` |
| Tab. 1 | GUPNet +PST | RealDrone3D | 15.97 / 14.3 / 15.7 / 19.5 / 93.03 / 33.21 | all | mean over 3 runs (per-run metrics, UDS averaged per run) | `A/gupnet_out_kd_anti_s0/pred_outdoor_test.pkl`<br>`A/gupnet_out_kd_anti_s1/pred_outdoor_test.pkl`<br>`A/gupnet_out_kd_anti_s2/pred_outdoor_test.pkl` |
| Tab. 1 | CenterNet | MAV6D | 13.09 / 0.666 / 5.92 / 20.42 / 69.79 / 32.59 | all | single run | `R/ourdet_baseproto/preds/in2mav_sim_s0_test.pkl` |
| Tab. 1 | CenterNet | RealDrone3D | 2.93 / 47.0 / 37.0 / 26.3 / 71.95 / 17.22 | all | single run | `R/ourdet_baseproto/preds/out2antifs_sim_s0_test.pkl` |
| Tab. 1 | CenterNet +DANN | MAV6D | 7.33 / 0.894 / 6.79 / 23.75 / 80.60 / 26.39 | all | single run | `R/ourdet_baseproto/preds/in2mav_dann_test.pkl` |
| Tab. 1 | CenterNet +DANN | RealDrone3D | 0.01 / 56.8 / 20.4 / 23.5 / 52.74 / 13.97 | all | single run | `R/ourdet_baseproto/preds/out2antifs_dann_test.pkl` |
| Tab. 1 | CenterNet +ST | MAV6D | 14.88 / 0.564 / 4.17 / 20.16 / 73.96 / 37.52 | all | single run | `R/ourdet_baseproto/preds/in2mav_st_test.pkl` |
| Tab. 1 | CenterNet +ST | RealDrone3D | 5.47 / 25.3 / 26.2 / 23.1 / 59.68 / 17.67 | all | single run | `R/ourdet_baseproto/preds/out2antifs_st_test.pkl` |
| Tab. 1 | CenterNet +PST | MAV6D | 23.84 / 0.485 / 2.94 / 20.89 / 95.88 / 47.10 | all | single run | `R/ourdet_baseproto/preds/in2mav_pst_test.pkl` |
| Tab. 1 | CenterNet +PST | RealDrone3D | 14.77 / 12.8 / 15.3 / 23.2 / 72.83 / 30.01 | all | single run | `R/ourdet_baseproto/preds/out2antifs_pst_test.pkl` |
| Tab. 1 | CenterNet$^*$ | MAV6D | 26.87 / 0.283 / 1.74 / 20.32 / 69.77 / 49.51 | all | single run | `X/in2mav/preds/srconly_test.pkl` |
| Tab. 1 | CenterNet$^*$ | RealDrone3D | 16.04 / 13.2 / 15.4 / 26.1 / 78.54 / 30.55 | all | single run | `X/out2antifs_k249/preds/srconly_test.pkl` |
| Tab. 1 | CenterNet$^*$, full pipeline of PST | MAV6D | 40.93 / 0.306 / 1.75 / 14.48 / 96.17 / 60.81 | all | single run | `X/in2mav/preds/pcmd_fused_test.pkl` |
| Tab. 1 | CenterNet$^*$, full pipeline of PST | RealDrone3D | 23.28 / 11.9 / 16.2 / 17.3 / 94.32 / 38.69 | all | single run | `X/out2antifs_k249/preds/pcmd_fused_test.pkl` |
| Tab. 1 | CenterNet, real labels | MAV6D | 45.46 / 0.276 / 0.05 / 17.62 / 99.94 / 65.36 | all | single run | `R/ourdet_baseproto/preds/in2mav_ub_test.pkl` |
| Tab. 1 | CenterNet, real labels | RealDrone3D | 5.73 / 19.5 / 1.2 / 24.7 / 84.91 / 34.39 | all | single run | `R/ourdet_baseproto/preds/out2antifs_ubzoom_test.pkl` |
| Tab. 2 | CenterNet$^*$ | MAV6D | 26.87 / 0.283 / 1.74 / 20.32 / 69.77 / 49.51 | all | single run | `X/in2mav/preds/srconly_test.pkl` |
| Tab. 2 | + Step 1, pseudo box | MAV6D | 34.04 / 0.356 / 2.54 / 25.44 / 96.23 / 53.38 | all | single run | `R/clean2/buildup/b_det1_test.pkl` |
| Tab. 2 | + Step 2, pseudo orientation | MAV6D | 34.04 / 0.355 / 2.54 / 14.48 / 96.21 / 55.78 | all | single run | `R/clean2/buildup/c_students_test.pkl` |
| Tab. 2 | + Step 3, pseudo depth, height ratio and fusion | MAV6D | 40.93 / 0.306 / 1.75 / 14.48 / 96.17 / 60.81 | all | single run | `X/in2mav/preds/pcmd_fused_test.pkl` |
| Tab. 2 | PST w/o composites in the depth refiner | MAV6D | 31.97 / 0.375 / 1.66 / 15.19 / 95.31 / 55.32 | all | single run | `R/heads/H3/mav_s0/preds/h3_fused_test.pkl` |
| Tab. 2 | PST w/ detection rate of row one | MAV6D | 30.63 / 0.247 / 1.36 / 11.96 / 69.77 / 54.14 | MISMATCH: AP 30.63->30.36, UDS 54.14->54.00 | derived: full pipeline, keep the 3349 most confident test frames (= frames detected by row one), others dropped | `X/in2mav/preds/pcmd_fused_test.pkl` |
| Tab. 3 | PST | MAV6D | 60.93 / 1.72 | all | mean of per-run UDS over 3 runs (shift 0/10/20) | `X/in2mav/preds/pcmd_fused_test.pkl`<br>`R/sup3/run/preds/pcmd_fused_test.pkl`<br>`R/seeds/mav_full_s20/preds/pcmd_fused_test.pkl` |
| Tab. 3 | PST | RealDrone3D | 41.71 / 0.31 | all | mean of per-run UDS over 3 runs (shift 0/10/20) | `X/out2antifs/preds/pcmd_fused_test.pkl`<br>`R/seeds/antifs_full_s10/preds/pcmd_fused_test.pkl`<br>`R/seeds/antifs_full_s20/preds/pcmd_fused_test.pkl` |
| Tab. 3 | PST w/o sensors | MAV6D | 56.71 / 4.39 | all | mean of per-run UDS over 3 runs (shift 0/10/20) | `R/sensorfree/E1/mav_s0/preds/e1_fused_test.pkl`<br>`R/sensorfree/E1/mav_s10/preds/e1_fused_test.pkl`<br>`R/sensorfree/E1/mav_s20/preds/e1_fused_test.pkl` |
| Tab. 3 | PST w/o sensors | RealDrone3D | 39.23 / 1.49 | all | mean of per-run UDS over 3 runs (shift 0/10/20) | `R/sensorfree/E1/antifs_s0/preds/e1_fused_test.pkl`<br>`R/sensorfree/E1/antifs_s10/preds/e1_fused_test.pkl`<br>`R/sensorfree/E1/antifs_s20/preds/e1_fused_test.pkl` |
| Tab. 3 | PST w/o sensors and composites | MAV6D | 58.00 / 2.32 | all | mean of per-run UDS over 3 runs (shift 0/10/20) | `R/sensorfree/E4/mav_s0/preds/e4_fused_test.pkl`<br>`R/sensorfree/E4/mav_s10/preds/e4_fused_test.pkl`<br>`R/sensorfree/E4/mav_s20/preds/e4_fused_test.pkl` |
| Tab. 3 | PST w/o sensors and composites | RealDrone3D | 35.11 / 3.29 | all | mean of per-run UDS over 3 runs (shift 0/10/20) | `R/sensorfree/E4/antifs_s0/preds/e4_fused_test.pkl`<br>`R/sensorfree/E4/antifs_s10/preds/e4_fused_test.pkl`<br>`R/sensorfree/E4/antifs_s20/preds/e4_fused_test.pkl` |
| Tab. 3 | PST w/ RGB-only teachers | MAV6D | 58.08 / 0.74 | all | mean of per-run UDS over 3 runs (shift 0/10/20) | `X/in2mav_none/preds/pcmd_fused_test.pkl`<br>`R/seeds/mav_none_s10/preds/pcmd_fused_test.pkl`<br>`R/seeds/mav_none_s20/preds/pcmd_fused_test.pkl` |
| Tab. 3 | PST w/ RGB-only teachers | RealDrone3D | 34.83 / 3.61 | all | mean of per-run UDS over 3 runs (shift 0/10/20) | `X/out2antifs_none/preds/pcmd_fused_test.pkl`<br>`R/seeds/antifs_none_s10/preds/pcmd_fused_test.pkl`<br>`R/seeds/antifs_none_s20/preds/pcmd_fused_test.pkl` |

## Main-text numbers and claims

| id | where | printed | recomputed | ok | how |
|---|---|---|---|---|---|
| gain PST-sim MAV min | sec/4_experiments.tex | 14.5 | 14.5 | yes | min over 6 detectors of UDS(+PST) - UDS(sim), unrounded means |
| gain PST-sim MAV max | sec/4_experiments.tex | 39.5 | 39.5 | yes | max, same |
| gain PST-sim RD min | sec/4_experiments.tex | 3.1 | 3.1 | yes | min over 6 detectors |
| gain PST-sim RD max | sec/4_experiments.tex | 21.0 | 21.0 | yes | max over 6 detectors |
| ATE sim 5 baselines min | sec/4_experiments.tex | 0.51 | 0.51 | yes | min ATE of the 5 baselines, sim only |
| ATE sim 5 baselines max | sec/4_experiments.tex | 1.42 | 1.42 | yes | max |
| ATE PST 5 baselines min | sec/4_experiments.tex | 0.28 | 0.28 | yes | min ATE of the 5 baselines, +PST |
| ATE PST 5 baselines max | sec/4_experiments.tex | 0.32 | 0.32 | yes | max |
| ASE of every detector below 3 cm (MAV, +PST) | sec/4_experiments.tex | ASE < 3 cm for all six +PST rows | max 2.94 | yes | qualitative claim, checked numerically |
| min mDR +PST MAV | sec/4_experiments.tex | 94.96 | 94.96 | yes | min mDR over 6 detectors, +PST |
| RD: +PST lowers ATE and AOE for all six | sec/4_experiments.tex | ATE and AOE lower than sim-only for all six on RD | checked 12 pairs | yes | qualitative claim, checked numerically |
| ST ATE MAV min | sec/4_experiments.tex | 0.44 | 0.44 | yes | min ATE of the six +ST rows |
| ST ATE MAV max | sec/4_experiments.tex | 0.99 | 0.99 | yes | max |
| ST changes UDS by at most 5 | sec/4_experiments.tex | |UDS(+ST) - UDS(sim)| <= 5 for all 12 | max |change| 4.93 | yes | qualitative claim, checked numerically |
| ST falls in N of 12 | sec/4_experiments.tex | 3 | 3 | yes | count of UDS(+ST) < UDS(sim) over 6 detectors x 2 targets |
| DANN lowers UDS for CenterNet | sec/4_experiments.tex | UDS(+DANN) < UDS(sim) on both targets |  | yes | qualitative claim, checked numerically |
| GUPNet DANN MAV | sec/4_experiments.tex | 22.05 | 22.05 | yes | table cell |
| GUPNet DANN RD | sec/4_experiments.tex | 14.34 | 14.34 | yes | table cell |
| PST-ST MAV min | sec/4_experiments.tex | 9.6 | 9.6 | yes | min over 6 detectors of UDS(+PST) - UDS(+ST) |
| PST-ST MAV max | sec/4_experiments.tex | 39.1 | 39.1 | yes | max |
| PST-ST RD min | sec/4_experiments.tex | 3.3 | 3.3 | yes | min |
| PST-ST RD max | sec/4_experiments.tex | 23.1 | 23.1 | yes | max |
| CenterNet* ASE | sec/4_experiments.tex | 1.74 | 1.74 | yes | table cell |
| CenterNet ASE | sec/4_experiments.tex | 5.92 | 5.92 | yes | table cell |
| CenterNet* ATE | sec/4_experiments.tex | 0.283 | 0.283 | yes | table cell |
| CenterNet ATE | sec/4_experiments.tex | 0.666 | 0.666 | yes | table cell |
| full pipeline gain mav | sec/4_experiments.tex | 11.3 | 11.3 | yes | UDS(full) - UDS(CenterNet*) |
| conclusion gain mav | sec/5_conclusion.tex | 11.3 | 11.3 | yes | same |
| full pipeline gain anti | sec/4_experiments.tex | 8.1 | 8.1 | yes | UDS(full) - UDS(CenterNet*) |
| conclusion gain anti | sec/5_conclusion.tex | 8.1 | 8.1 | yes | same |
| full pipeline UDS MAV | sec/4_experiments.tex | 60.81 | 60.81 | yes | table cell |
| full pipeline UDS RD | sec/4_experiments.tex | 38.69 | 38.69 | yes | table cell |
| AP CenterNet* MAV | sec/4_experiments.tex | 26.87 | 26.87 | yes | table cell |
| AP full MAV | sec/4_experiments.tex | 40.93 | 40.93 | yes | table cell |
| AP CenterNet* RD | sec/4_experiments.tex | 16.04 | 16.04 | yes | table cell |
| AP full RD | sec/4_experiments.tex | 23.28 | 23.28 | yes | table cell |
| full pipeline detects more and lower AOE (both targets) | sec/4_experiments.tex | mDR up and AOE down vs CenterNet* |  | yes | qualitative claim, checked numerically |
| full pipeline best UDS of Tab. 1 on RD | sec/4_experiments.tex | max UDS on RD |  | yes | qualitative claim, checked numerically |
| within 4.6 of real labels | sec/4_experiments.tex | 4.6 | 4.5 | **NO** | UDS(real labels) - UDS(full), MAV6D |
| MonoCoP +PST MAV UDS | sec/4_experiments.tex | 61.90 | 61.90 | yes | table cell |
| MonoDGP +PST MAV UDS | sec/4_experiments.tex | 60.55 | 60.55 | yes | table cell |
| PST - w/o sensors mav | sec/4_experiments.tex | 4.22 | 4.22 | yes | mean UDS(PST) - mean UDS(w/o sensors), 3 runs each |
| conclusion PST - w/o sensors mav | sec/5_conclusion.tex | 4.2 | 4.2 | yes | same, 1 decimal |
| PST - w/o sensors anti | sec/4_experiments.tex | 2.47 | 2.47 | yes | mean UDS(PST) - mean UDS(w/o sensors), 3 runs each |
| conclusion PST - w/o sensors anti | sec/5_conclusion.tex | 2.5 | 2.5 | yes | same, 1 decimal |
| 95% interval excludes zero mav (read from saved bootstrap json) | sec/4_experiments.tex | interval of PST - w/o sensors excludes zero | saved: [-6.671124491006582, -2.2611529783667925] | yes | qualitative claim, checked numerically |
| 95% interval excludes zero anti (read from saved bootstrap json) | sec/4_experiments.tex | interval of PST - w/o sensors excludes zero | saved: [-3.7216377891465338, -1.1285085457151804] | yes | qualitative claim, checked numerically |
| PST above all controls on both targets | sec/4_experiments.tex | mean PST > every control mean |  | yes | qualitative claim, checked numerically |
| furthest behind on RD (up to) | sec/4_experiments.tex | 6.87 | 6.87 | yes | max over the two no-composite controls of mean UDS(PST) - mean UDS(control), RD |
| PST RD all frames | sec/4_experiments.tex | 41.71 | 41.71 | yes | Tab. 3 cell |
| PST RD restricted | sec/4_experiments.tex | 38.69 | 38.69 | yes | Tab. 1 cell |
| tab2 mDR row1 | sec/4_experiments.tex | 69.77 | 69.77 | yes | table cell |
| tab2 mDR step1 | sec/4_experiments.tex | 96.23 | 96.23 | yes | table cell |
| tab2 AOE step2 | sec/4_experiments.tex | 14.48 | 14.48 | yes | table cell |
| tab2 AOE row1 | sec/4_experiments.tex | 20.32 | 20.32 | yes | table cell |
| tab2 AP step2 | sec/4_experiments.tex | 34.04 | 34.04 | yes | table cell |
| tab2 AP step3 | sec/4_experiments.tex | 40.93 | 40.93 | yes | table cell |
| full ATE lower on as many frames as CenterNet* detects | sec/4_experiments.tex | ATE(row 6) < ATE(row 1) | 0.247 vs 0.283 | yes | qualitative claim, checked numerically |
| composites: UDS from | sec/4_experiments.tex | 60.81 | 60.81 | yes | table cell |
| composites: UDS to | sec/4_experiments.tex | 55.32 | 55.32 | yes | table cell |
| abstract CenterNet PST-sim mav | sec/0_abstract.tex | 14.5 | 14.5 | yes | UDS(CenterNet +PST) - UDS(CenterNet) |
| abstract CenterNet PST-sim anti | sec/0_abstract.tex | 12.8 | 12.8 | yes | UDS(CenterNet +PST) - UDS(CenterNet) |
| fig4 det depth err | sec/4_experiments.tex | 7.6 | 7.6 | yes | median |rel. depth error| %, 5,227 pseudo-labelled training frames |
| fig4 PST depth err | sec/4_experiments.tex | 5.9 | 5.9 | yes | same, PST pseudo box |
| fig4 RGB-only depth err | sec/4_experiments.tex | 8.8 | 8.8 | yes | same, RGB-only teachers |
| fig4 det orient err | sec/4_experiments.tex | 30.1 | 30.1 | yes | mean AOE, frames whose detected centre lies on the UAV (6,047) |
| fig4 PST orient err | sec/4_experiments.tex | 18.8 | 18.8 | yes | same, PST orientation pseudo label |

## Prediction files (sha256, size, mtime)

| file | sha256 | size | mtime |
|---|---|---|---|
| `D:/CVPR27_UAV/figures/data/pl_quality_mav6d.csv` | 143a59adf916a2300619efc8ff625ff8ee89056d3380ddf6821762504db37ab0 | 2508782 | 2026-11-14 11:53:18 |
| `A/gupnet_in_kdc_mav_s0/pred_real_test.pkl` | c3e9432164183fa53759a5a9b33b1dff931eb7c6d89088587b729727443ba534 | 4366852 | 2026-11-14 13:24:54 |
| `A/gupnet_in_kdc_mav_s1/pred_real_test.pkl` | 1ca7080c8c3bd8f434814bdf623676ae4e83f16ad63257c01dac8e63a9f5e0d4 | 4366852 | 2026-11-14 13:43:28 |
| `A/gupnet_in_kdc_mav_s2/pred_real_test.pkl` | 21eea0f63dec55ca0f50f9bc87add9974174a188b9245eb1d0f73bf0323e06b4 | 4366852 | 2026-11-14 13:45:40 |
| `A/gupnet_in_sim_s1/pred_real_test.pkl` | 9c2aad408c796acfa5df365828660b7e81223575c1902b6c81a485132239ae6c | 4366852 | 2026-11-12 15:56:38 |
| `A/gupnet_in_sim_s2/pred_real_test.pkl` | 2668d15f278145c02be1095becdddbdae95ab700e68835db1c1a3349b9dc7910 | 4366852 | 2026-11-12 13:43:46 |
| `A/gupnet_in_st_mav_s0/pred_real_test.pkl` | 8d3a6adcfb6127fb21615f0834f3f03d67f0a5a32bd0402661ac7822a5c7d12d | 4366852 | 2026-11-12 15:55:12 |
| `A/gupnet_out_dann_anti_s0/pred_outdoor_test.pkl` | 60db89e467db1723c603335b1b6be09de976b005a04021534d8c051014f779d2 | 2343005 | 2026-11-16 20:03:14 |
| `A/gupnet_out_kd_anti_s0/pred_outdoor_test.pkl` | a051b5b799a30cf7f6ad151fe8d208b10721e1121e99e610ccf0aa0c02e13eaf | 2343005 | 2026-11-12 17:27:26 |
| `A/gupnet_out_kd_anti_s1/pred_outdoor_test.pkl` | 36a9f62b385a5e77f05c9a2e41434301c5b4f7b7af4ab8512a505c94425dcead | 2343005 | 2026-11-16 00:37:44 |
| `A/gupnet_out_kd_anti_s2/pred_outdoor_test.pkl` | d3f25b5063d2756cab0640d554b26eb90d07be6ea9ba161b70417cc82d7e9689 | 2343005 | 2026-11-16 00:38:52 |
| `A/gupnet_out_sim_s0/pred_outdoor_test.pkl` | ce2150b2260dbe57a17a5fb7ba649adb2624e8e3bf9c89c425d9381482d194a8 | 2343005 | 2026-11-12 17:12:34 |
| `A/gupnet_out_sim_s1/pred_outdoor_test.pkl` | db1fec64a9afe7ba35582948308c0e716c180797ad1ff2f9b2018fe5f0f98b78 | 2343005 | 2026-11-16 15:30:34 |
| `A/gupnet_out_sim_s2/pred_outdoor_test.pkl` | 5ef8ac9da681ecdd372e76d03f9617ec3674e50e1cfcd3206af0763bb7ee34bd | 2343005 | 2026-11-16 15:33:30 |
| `A/gupnet_out_st_anti_s0/pred_outdoor_test.pkl` | 5b2d7612aab82906f308411163499f197db779c0b0484c12b383fbc38fa54b27 | 2343005 | 2026-11-12 23:00:32 |
| `A/gupnet_out_st_anti_s1/pred_outdoor_test.pkl` | 31e896e13695ba4e0cf010a5f78e52c851259177037e0434e9231240611d01eb | 2343005 | 2026-11-16 20:05:22 |
| `A/gupnet_out_st_anti_s2/pred_outdoor_test.pkl` | 07d612998493e9ffd94421d60972ee430fa9ee7565b57ddf4ba0dd808a45b029 | 2343005 | 2026-11-16 20:07:46 |
| `A/monocd_in_kdc_mav_s0/pred_real_test.pkl` | d9017588ddb7962848ce5bfe0d7a30d1735ef8af563773d771e48f93f06dbd4d | 4366852 | 2026-11-16 05:28:14 |
| `A/monocd_in_kdc_mav_s1/pred_real_test.pkl` | 9e83a0b4cc4a0c7e2abed6ae68dd8cf5016842acf7cda83ea9581a8a0803692b | 4366852 | 2026-11-14 21:00:58 |
| `A/monocd_in_kdc_mav_s2/pred_real_test.pkl` | 25504c9af698f9a6a9cb78b0f19c03abd89ed7974501affd6b5ef65e9b1a9e74 | 4366852 | 2026-11-16 03:10:54 |
| `A/monocd_in_sim_s1/pred_real_test.pkl` | f92ee1ba69a54878d12e4275f3e6e8820a50568648cdd3b8a5084bddc0f38cbf | 4366852 | 2026-11-14 03:05:50 |
| `A/monocd_in_sim_s2/pred_real_test.pkl` | f15111946f97f803138df889185bd169087324ce9c04b959ce5afc92225703a2 | 4366852 | 2026-11-14 04:03:10 |
| `A/monocd_in_st_mav_s1/pred_real_test.pkl` | 97704ac9187d555274b64455df30462df5c6f4b2fa8bc21d913ef8af8542454d | 4366852 | 2026-11-15 02:35:52 |
| `A/monocd_out_kd_anti_s0/pred_outdoor_test.pkl` | 643623ab9ecc6000b22adc502658c50af6b4733135aaa8465c3edac3cbd110ee | 2343005 | 2026-11-13 20:25:30 |
| `A/monocd_out_kd_anti_s1/pred_outdoor_test.pkl` | 42db605308fbc68c59b3814d90f7b54ef421db17d12eaba11976c46eea4e6b55 | 2343005 | 2026-11-16 05:34:46 |
| `A/monocd_out_kd_anti_s2/pred_outdoor_test.pkl` | 4ccab4a5b8cffa236a304fc8b19d2666e73b5895dcb866ffea4d3aad36b41b30 | 2343005 | 2026-11-16 07:05:38 |
| `A/monocd_out_sim_s0/pred_outdoor_test.pkl` | e39ded9cb5fefbf750a9ae09c63db2b323eba4a1f09c6ff54fae968bdab15440 | 2343005 | 2026-11-13 17:24:08 |
| `A/monocd_out_sim_s1/pred_outdoor_test.pkl` | b992c34d6e1999f4e3639896baf943bfd8ad28770b95d8f4635d2f2a837d4e31 | 2343005 | 2026-11-16 13:43:10 |
| `A/monocd_out_sim_s2/pred_outdoor_test.pkl` | d6b1122f6c9964ee6b09fac8c3c9f8494aa8737ec7c077c88a2bb429f21700eb | 2343005 | 2026-11-16 15:33:06 |
| `A/monocd_out_st_anti_s0/pred_outdoor_test.pkl` | a2b5febce7f2afced614fcab9f6af92b7b3dca0962725122cb1f91a67b317ae8 | 2343005 | 2026-11-14 01:06:36 |
| `A/monocd_out_st_anti_s1/pred_outdoor_test.pkl` | a03ba96dc3c109449d31bce62a602ec1efe1b8bd3a8c115dd27236f650087c12 | 2343005 | 2026-11-16 20:38:32 |
| `A/monocd_out_st_anti_s2/pred_outdoor_test.pkl` | bed48738386c8a29c84af15299573a27cc79e1391006f5b896de08bc94934553 | 2343005 | 2026-11-16 22:02:24 |
| `A/monocop_in_kdc_mav_s0/pred_real_test.pkl` | 689491a9e0e1346a48ba3ccb22f4201daa93cebc580c3338c06c148a9a7f974e | 2446591 | 2026-11-15 06:01:58 |
| `A/monocop_in_sim_s0/pred_real_test.pkl` | d5553885b301932f09ec8cd2d8ab0a3915dba0945266f2ae160a8e25ab6c1776 | 2446591 | 2026-11-12 10:11:46 |
| `A/monocop_in_st_mav_s0/pred_real_test.pkl` | 81b7945d1d2e482e8475dbb63c3b9b4dec39ba80981d562992949b600f360d2a | 2446591 | 2026-11-13 14:25:32 |
| `A/monocop_out_kd_anti_s0/pred_outdoor_test.pkl` | c7ab172b872c060fbb07aa27c7536ab7ae28f0af2d88e76e4cc73ea0805ca11f | 1293261 | 2026-11-13 02:07:16 |
| `A/monocop_out_kd_anti_s1/pred_outdoor_test.pkl` | d0c185ec2bcafb4bc6e774483209ab231cfd7d91632be96f6386366cad6a5657 | 1293261 | 2026-11-16 13:48:44 |
| `A/monocop_out_kd_anti_s2/pred_outdoor_test.pkl` | 2bba169616999a9bfc7fa9643783d10774f980531ec6fad6d797a55f46e17e81 | 1293261 | 2026-11-16 13:48:00 |
| `A/monocop_out_sim_s0/pred_outdoor_test.pkl` | 786e6bf22c3789cb062c50c4e5d93629b4b6d3a39968764c406f14651c359a06 | 1293261 | 2026-11-12 16:18:38 |
| `A/monocop_out_st_anti_s0/pred_outdoor_test.pkl` | 2beaee7efb2d48ae0bda39280a319959e4343ef8c816877ef99d9a3999384792 | 1293261 | 2026-11-13 03:10:40 |
| `A/monodetr_in_kdc_mav_s0/pred_real_test.pkl` | 68a468c8162f64a7eb1c44eb0120ceb664fb506bcaa5319c93e39a2b15a97c4c | 2446591 | 2026-11-14 23:41:18 |
| `A/monodetr_in_kdc_mav_s1/pred_real_test.pkl` | 3d5c023b9e61cf1a5fe5412bfc5d3f22ce6110c4f82f0b45205063a85357735e | 2446591 | 2026-11-17 05:27:46 |
| `A/monodetr_in_kdc_mav_s2/pred_real_test.pkl` | 37d255ed0ea7b399ef81fc7bd44af9a42cbe5cb400d131661873c87b69f74ce1 | 2446591 | 2026-11-17 05:30:52 |
| `A/monodetr_in_sim_s0/pred_real_test.pkl` | 7263978901d636ef7b7de7bf993d188105207ce5d723f87d2fbc6007f6ef0d4c | 2446591 | 2026-11-12 10:13:14 |
| `A/monodetr_in_st_mav_s0/pred_real_test.pkl` | d3b550a004299f6ff6c91cc17fd7239ab7f5fbbe2673d538fac346408f1cf9ea | 2446591 | 2026-11-13 03:17:08 |
| `A/monodetr_out_kd_anti_s0/pred_outdoor_test.pkl` | 7123f59f1ea29b41d2e4a895bbe4b1162c4d35c1c6ee321fb15f24d96f76a840 | 1293261 | 2026-11-12 19:38:54 |
| `A/monodetr_out_kd_anti_s1/pred_outdoor_test.pkl` | fc86dfbdbd42c8f065943702302c6b9a5135e5aea019555ed757a1fb6a5d1c7b | 1293261 | 2026-11-16 08:51:42 |
| `A/monodetr_out_kd_anti_s2/pred_outdoor_test.pkl` | 169fa4f5d71321c0e8a93dfe715fed46fa8b9e7a3c503bac0b05066e9e2f1d82 | 1293261 | 2026-11-16 10:21:24 |
| `A/monodetr_out_sim_s0/pred_outdoor_test.pkl` | c8fe382285d86d9d261ead1971f9275e8a33c7390c0621525749e3a29ecc65f9 | 1293261 | 2026-11-12 16:08:02 |
| `A/monodetr_out_st_anti_s0/pred_outdoor_test.pkl` | cf21f4040114a0cb50d4c39d0ba74bdebd4966043bcc8f767c05bd0907d5d404 | 1293261 | 2026-11-13 00:14:26 |
| `A/monodgp_in_kdc_mav_s0/pred_real_test.pkl` | b2fb1848ecf73b456992d5a7f5dd6bc21ab90d889d7973e9f191d0e2e2f7e861 | 2446591 | 2026-11-15 07:12:18 |
| `A/monodgp_in_kdc_mav_s1/pred_real_test.pkl` | 5d9ee0c1e4754a50d423e3b692d34eea5b1f05a2e64e422f0dae95549a72946b | 2446591 | 2026-11-16 11:24:04 |
| `A/monodgp_in_kdc_mav_s2/pred_real_test.pkl` | 66cafb292cd05ff128f0167211ad7250e9c7cd478bf2f48d50a9d22d3b4da062 | 2446591 | 2026-11-16 11:27:18 |
| `A/monodgp_in_st_mav_s0/pred_real_test.pkl` | 4f3c6582895b41ab81b7b9d4c92a637bc32116463ea38e1117d709bcc0764490 | 2446591 | 2026-11-13 12:48:22 |
| `A/monodgp_out_kd_anti_s0/pred_outdoor_test.pkl` | 77576d22e77462644353ece7a7f3ab4b489a4fcc3f1aa7a0ce5a7be083fcad5e | 1293261 | 2026-11-12 20:30:04 |
| `A/monodgp_out_kd_anti_s1/pred_outdoor_test.pkl` | 26298506c0fbb9ee36ada6891ee388aa63cd37dcc34882d409055b1740312b16 | 1293261 | 2026-11-16 08:50:40 |
| `A/monodgp_out_kd_anti_s2/pred_outdoor_test.pkl` | ba1ebf49780d443b902cb995e9c1ba544515b52205626219cc1f1d7dd26adc49 | 1293261 | 2026-11-16 10:46:54 |
| `A/monodgp_out_sim_s0/pred_outdoor_test.pkl` | 9609e16bbac82078ecd2bc1eaab3b82bb8c114da545e87b42474f0dc018c7dd4 | 1293261 | 2026-11-12 13:52:56 |
| `A/monodgp_out_sim_s1/pred_outdoor_test.pkl` | bf1364134a2d6e2f1f4b93c2a6a151f7b29b0cb712062ebf0f5eb87cd2b377b7 | 1293261 | 2026-11-16 19:50:08 |
| `A/monodgp_out_sim_s2/pred_outdoor_test.pkl` | 3072ac38a4a438f07a81d768722f4698bd2bd72a0c828a412475f2f04caa51b4 | 1293261 | 2026-11-16 19:53:46 |
| `A/monodgp_out_st_anti_s0/pred_outdoor_test.pkl` | 6349b4b7504a536f7af33847efc5d570962aaecfc3e9c33acad17193236fcce5 | 1293261 | 2026-11-13 03:33:18 |
| `A/monodgp_out_st_anti_s1/pred_outdoor_test.pkl` | 22831423437f4074273052e50e2b77d53e20e436a84f33811783a3dabc5fa6a4 | 1293261 | 2026-11-17 03:04:18 |
| `A/monodgp_out_st_anti_s2/pred_outdoor_test.pkl` | b2fe77762308d17b81cff3d5334e2e14aae02b1dafc7485383d0a155adf4ab38 | 1293261 | 2026-11-17 03:07:14 |
| `L/gupnet_s0/pred_real_test.pkl` | 041c865da2f83fa80635a4a4bdc10d32c0e4796afe82302817f34201e57e3a3e | 4366852 | 2026-10-30 06:57:25 |
| `L/gupnet_uda_dann_s0/pred_real_test.pkl` | fc737b8115414e93b16da69f88dfbe5120ce9b8265ea0fdc452397bea9443251 | 4366852 | 2026-11-14 13:59:34 |
| `L/monocd_s0/pred_real_test.pkl` | 94a487d7f0e3fb2f82218125198a947fad679e094c8782ae1829c818455d05c6 | 4366852 | 2026-11-10 11:23:58 |
| `L/monodgp_s0/pred_real_test.pkl` | 4ba2c3a6c8c8664ec3e0a97ab117b1b19d19da7becad186d925547ba0a7dba0c | 2446591 | 2026-11-10 21:29:33 |
| `R/clean2/buildup/b_det1_test.pkl` | 3edf89e8801a1c699a64fb454f3864442fc6c958244e392fb00bfd1eeb018c9a | 895975 | 2026-11-13 10:54:32 |
| `R/clean2/buildup/c_students_test.pkl` | 6a16969e0fbbbc5cb8c8066723225cac75adb66baa532f647f94db1db7a0adb0 | 895975 | 2026-11-13 10:54:32 |
| `R/heads/H3/mav_s0/preds/h3_fused_test.pkl` | 01940d75f3168aa986cb306fa9221afb6fce350a1a1641783c0ceac8b2f1700b | 895975 | 2026-11-16 02:21:18 |
| `R/ourdet_baseproto/preds/in2mav_dann_test.pkl` | ec1cf030268367e7db8b4f75484db8c2c8e80acd8bb91839ff1d898cde4da25d | 895975 | 2026-11-16 16:39:20 |
| `R/ourdet_baseproto/preds/in2mav_pst_test.pkl` | 711cd0f9135b366eebfd22736ab09b3201b3fa6a361809a3246e25cca4f85fed | 895975 | 2026-11-16 16:36:56 |
| `R/ourdet_baseproto/preds/in2mav_sim_s0_test.pkl` | 0512587208c89cd0346325d88c7e4e5dfb3caa9ca214f64a278e2fb7afd5ddab | 895975 | 2026-11-16 16:42:06 |
| `R/ourdet_baseproto/preds/in2mav_st_test.pkl` | ccfa2d286797f3e7408ae726b4218198963f89250df6e850e52e85493307c6cb | 895975 | 2026-11-16 18:29:40 |
| `R/ourdet_baseproto/preds/in2mav_ub_test.pkl` | 1dda03c58ded5990475578c173c0918f0ab156a61dd6af6ae51f95d9ff0c7910 | 895975 | 2026-11-16 17:58:08 |
| `R/ourdet_baseproto/preds/out2antifs_dann_test.pkl` | 2c84a324148390d914aa9f2851223263018bb30acf29f94aa14c90fa1d675390 | 445592 | 2026-11-16 17:04:50 |
| `R/ourdet_baseproto/preds/out2antifs_pst_test.pkl` | 011183c0181adf119a62ac46c7beba16beb6b1204f57c90e84634414d0aca0ec | 445592 | 2026-11-16 17:02:22 |
| `R/ourdet_baseproto/preds/out2antifs_sim_s0_test.pkl` | c6afa6e39d137cca808d291d45cabf6f9913fc43cbe5bfd3cc28fbd28b138a12 | 445592 | 2026-11-16 17:08:54 |
| `R/ourdet_baseproto/preds/out2antifs_st_test.pkl` | 9da464167e27c149fb66a748952a91d9d704656d420e80d03451144f4999b821 | 445592 | 2026-11-16 17:43:42 |
| `R/ourdet_baseproto/preds/out2antifs_ubzoom_test.pkl` | 0a2cb069a54bcd7d512d10b153317545c1408e297fbdf639c408e617cc850cb9 | 445592 | 2026-11-17 01:36:02 |
| `R/seeds/antifs_full_s10/preds/pcmd_fused_test.pkl` | b667a7eafe6f860e0068528f11e57b07449420d37dd672a57526e1b2729c0418 | 445592 | 2026-11-14 03:14:44 |
| `R/seeds/antifs_full_s20/preds/pcmd_fused_test.pkl` | 27dbbb9c58a3bfcca2006283e1dacb1ca67a53f05708d329675e22aa381cc914 | 445592 | 2026-11-14 13:05:24 |
| `R/seeds/antifs_none_s10/preds/pcmd_fused_test.pkl` | 929a1aa938c71da3a12a0daf9f5a47d643d692dca0cdba1518f53ec7e90f3863 | 445592 | 2026-11-14 05:57:06 |
| `R/seeds/antifs_none_s20/preds/pcmd_fused_test.pkl` | 35db69276dc57bcafb934a253b3fd6ce8b954e0403be1e530dc50c30e5fa00e3 | 445592 | 2026-11-14 15:41:38 |
| `R/seeds/mav_full_s20/preds/pcmd_fused_test.pkl` | cfb676d0a7c64f976ebb94818fcbc8fd60d0f960a08fb650015b764c6e67d71a | 895975 | 2026-11-13 22:51:14 |
| `R/seeds/mav_none_s10/preds/pcmd_fused_test.pkl` | 70769055e965c1e3c05b08f02bd5b765a1ab71293d5ff6d3b0ce8e92f9dd8f4b | 895975 | 2026-11-13 20:04:24 |
| `R/seeds/mav_none_s20/preds/pcmd_fused_test.pkl` | bfcdf30029cb6af3dc5c9e2d05f80cdc1368bd6d384f3972f74d138e9ca65fc3 | 895975 | 2026-11-14 01:26:12 |
| `R/sensorfree/E1/antifs_s0/preds/e1_fused_test.pkl` | 4b7ce82461625ea21df9b23700ca79bd908573fbef85569d2680dc61e9e08606 | 445592 | 2026-11-15 04:48:40 |
| `R/sensorfree/E1/antifs_s10/preds/e1_fused_test.pkl` | 09334e360b6e37885eb55478f64d50d80d1ffc883bcb40c48a941c42fc980cd6 | 445592 | 2026-11-15 06:54:16 |
| `R/sensorfree/E1/antifs_s20/preds/e1_fused_test.pkl` | 0f1fce8e5d44b1cb3d84ba849d2c9b651a1e4c13e12eea6135698e4b03b750d8 | 445592 | 2026-11-15 10:41:50 |
| `R/sensorfree/E1/mav_s0/preds/e1_fused_test.pkl` | 248fb62257be9673976e7dd46c818a141b33b3d4773c43c59256ba74b3a43689 | 895975 | 2026-11-15 06:55:52 |
| `R/sensorfree/E1/mav_s10/preds/e1_fused_test.pkl` | 189edfafc858c46ad40120bdeac7d0ee558d9724915eb1c5b65d0b4f238b9b59 | 895975 | 2026-11-15 12:24:36 |
| `R/sensorfree/E1/mav_s20/preds/e1_fused_test.pkl` | f69336ffc0dfd25e3fe9a631c99e40d11d4dcdda140ba7c4678dc3fdc9af800f | 895975 | 2026-11-15 18:44:00 |
| `R/sensorfree/E1/pooled_antifs_vsPST.json` | e77660b979146c0e78ac14404b33fa7e88023a8ae3b88ce9f3381522dd93d45c | 5534 | 2026-11-15 10:42:32 |
| `R/sensorfree/E1/pooled_mav_vsPST.json` | 58fbc20d203fb4f5091af76fbdbd7d3a739cc13454cc79b3e84ef370d55bc66b | 5378 | 2026-11-15 18:45:26 |
| `R/sensorfree/E4/antifs_s0/preds/e4_fused_test.pkl` | 8457bfbba2d69d4454f4c7b475c1cb937ce1785f093dcb1be332ed2476f13761 | 445592 | 2026-11-16 01:42:52 |
| `R/sensorfree/E4/antifs_s10/preds/e4_fused_test.pkl` | 50955ed3a44fcf2a12ca9b1e1cd978c3885e28303562f965cdcf918480676b7b | 445592 | 2026-11-17 11:38:04 |
| `R/sensorfree/E4/antifs_s20/preds/e4_fused_test.pkl` | 5cf19bea199892c7bf77b21fe3343ef2de6dab82de68f3e9a3f85298d65f3352 | 445592 | 2026-11-17 11:43:14 |
| `R/sensorfree/E4/mav_s0/preds/e4_fused_test.pkl` | 2ff09a4234aad596bfe61a192f18d1c148abbd8f7099fb76b9f9fd6589e1f24f | 895975 | 2026-11-15 23:58:02 |
| `R/sensorfree/E4/mav_s10/preds/e4_fused_test.pkl` | ff904ffab15a462cfa8ce49a17bb8a14e274f95d59a62e7e9154fdd545c0bde0 | 895975 | 2026-11-17 08:30:44 |
| `R/sensorfree/E4/mav_s20/preds/e4_fused_test.pkl` | be9d3a2bcde7e20e35044c0bbce103c99ce92e8682b5c6fe8bc548e9bfe2b4a6 | 895975 | 2026-11-17 12:12:14 |
| `R/sup3/run/preds/pcmd_fused_test.pkl` | e07cd8e32ea42a8267d789d8ca73df0a13ba3e536074954801457b8e125f9106 | 895975 | 2026-11-13 17:37:38 |
| `X/in2mav/preds/pcmd_fused_test.pkl` | 4574b709a1694e9e77449c145eb356473bf80a5bd495ff0c0160b882f0f7f07b | 895975 | 2026-11-13 08:24:18 |
| `X/in2mav/preds/srconly_test.pkl` | 33061861003e27eee31e431e04eeb8d0483c65cbb5fd89fec2f8cf17096c02a3 | 895975 | 2026-11-13 04:34:38 |
| `X/in2mav_none/preds/pcmd_fused_test.pkl` | ab0b8cddc76b873e8119250dbc282f18437094376f6dc701a186bacc107946e4 | 895975 | 2026-11-13 12:06:10 |
| `X/out2antifs/preds/pcmd_fused_test.pkl` | 70cac1563d1b03233fdaf42027ce2af4e628c3653375e73ecec9307172a15a79 | 445592 | 2026-11-12 05:04:00 |
| `X/out2antifs_k249/preds/pcmd_fused_test.pkl` | 2177a800d7dcba2e1a6c2b718960ba9d692ebbe52b2e73844a98421833a3ab17 | 445592 | 2026-11-17 14:36:22 |
| `X/out2antifs_k249/preds/srconly_test.pkl` | d42238f40c54d89b5f03798889ed53674b6aea8ad4f77bf71d69271faa9a9028 | 445592 | 2026-11-17 12:10:44 |
| `X/out2antifs_none/preds/pcmd_fused_test.pkl` | db95f6d74fd46ce21977a3b805976d0a3f41e3b506c80611752e5e3dbe76ad14 | 445592 | 2026-11-12 13:31:44 |
