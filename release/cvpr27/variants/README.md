# variants: every experiment of gproto/review1113 (what feeds what)

Each subfolder is a copy of `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/<name>/` without data, crops, checkpoints and prediction pickles
(those are listed in `../repro/ARTIFACTS.md`). Most variants build an isolated copy of `pipeline/srcx/lib` with a small patch (`make_*_lib.py`) and run it with
their own run script; the per-run `lib/` copies are the exact code that ran. `_top/` holds the review1113 top-level notes and `sup_patch.py`.

| folder | feeds |
|---|---|
| `k249` | **Tab. 1**, RealDrone3D row "CenterNet*, full pipeline of PST" (frame restriction to the 249 frames of the baselines); supplement "The full pipeline on the same frames" |
| `ourdet_baseproto` | **Tab. 1** CenterNet rows (simulation only, +DANN, +ST, +PST, real labels; RealDrone3D real labels = `ubzoom`); supplement "CenterNet rows" |
| `clean2` (`buildup/`) | **Tab. 2** rows 2-3 (Step 1, Step 2); `CLEAN2_REPORT.md` |
| `heads` (H1-H7) | **Tab. 2** row 5 = `H3/mav_s0` (no composites in the depth refiner); supplement ablations "Silhouette of the depth refiner", "Sim-on-real composites", "Mirror-consistency loss", "LiDAR restriction of the paste mask" |
| `seeds`, `sup3` | **Tab. 3** runs 2-3 of PST and RGB-only teachers (`sup3` = PST MAV6D shift 10; `seeds/*` = the others); `seeds/pooled_anti_test.py` = pooled bootstrap |
| `sensorfree` | **Tab. 3** "PST w/o sensors" = `E1/*`, "PST w/o sensors and composites" = `E4/*`; E1b, E2, E2a, E2b = supplement controls; pooled bootstrap json behind the Sec. 4 intervals |
| `boxhull` | box-hull rendered-truth control of the depth refiner (supplement; pooled with E2) |
| `cpu5` | RealDrone3D build-up of the steps (supplement) |
| `single_sbar` | single CenterNet + mirror + constant size (supporting the Sec. 4 sentence that the gain of CenterNet* comes from the constant size and the mirror image) |
| `stage1seeds` | stage-1 networks retrained with new seeds (supplement "Number of runs") |
| `hparam_1117` | hyperparameter sensitivity (supplement) |
| `strata_1117` | per-stratum analysis of IR / LiDAR on real test frames (supplement) |
| `lowres` | full chain with crops cut from the 512x288 frames the baselines see (supplement, image size) |
| `corrupt_1117` | test-time corruptions (supplement) |
| `xdomain_1117` | cross-target transfer without re-adaptation (supplement) |
| `figdata` | Fig. 4 pseudo-label quality (`fig_pl_quality.py`) and student-follows-label analysis |
| `panelA`, `A3` | fusion seeds, size sweep, block bootstrap (supplement) |
| `breakdown` | results per UAV model (supplement) |
| `orient_dist`, `senses`, `height` | front/back and orientation analyses, sensor agreement, LiDAR box height (supplement) |
| `initabl`, `lidraw`, `rgbst_equal`, `B3`, `B4`, `B8`, `cleanfix`, `sup1`, `sup2`, `M12`, `N1`, `N1clean`, `N2`, `R123`, `clean`, `clean3`, `r6`, `plq_1116` | review / diagnostic experiments (student init, raw LiDAR, equal-condition RGB self-training, routing, naive ST, w0 seeds, clean-chain fixes, AdaBN, no gate, orientation/depth diagnostics, qualitative figure); each has its own report |
| `table_audit_1116` | 11-16 audit of every table cell (predecessor of `../repro`) |
