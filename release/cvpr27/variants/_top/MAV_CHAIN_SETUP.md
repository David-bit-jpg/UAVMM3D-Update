# MAV6D runs of the source-swap chain (11-13)

Chain: `gproto/srcx/run_x.sh <src> mav` (method unchanged; target constants in `srcx/lib/sx.py` TARGETS['mav']).
Full record of choices and glue edits: `srcx/STATUS.md` § "11-13 MAV6D target".

## Runs
| run | what | outputs | log |
|---|---|---|---|
| M4 `in2mav` | indoor sim source → MAV6D, clean protocol: starts from the sim-only detectors release_20260920/ramp_s{0,1,2}; no development weights, no bootstrap networks (refiner stage 1 retrained, self-trained detectors from the sim-only ones) | `srcx/in2mav/` (+ crops `D:/S2MUAV3D/srcx_work/in2mav/`) | `srcx/in2mav/logs/run_x.log` (ends `ALL DONE` or `!!`), stdout `srcx/in2mav/logs/run_x.stdout` |
| M3 `in2mav_ir` | teachers = 2 IR-distilled nets; reuses M4 detector / refiner / pseudo-labels | `srcx/in2mav_ir/` | `srcx/in2mav_ir/logs/run_x.log` |
| M3 `in2mav_lid` | teachers = 2 LiDAR nets; refiner without IR silhouettes (full rerun) | `srcx/in2mav_lid/` | `srcx/in2mav_lid/logs/run_x.log` |
| M3 `in2mav_none` | teachers = indoor plain-RGB KZs0/KZs1 (ep 30); refiner without IR (full rerun) | `srcx/in2mav_none/` | `srcx/in2mav_none/logs/run_x.log` |

M3 is started by `srcx/launch_mav_teach_1113.sh` (log `srcx/logs/launch_mav_teach_1113.log`): waits for `ALL DONE` in the M4 log,
then runs ir → lid → none in sequence, default seeds. If M4 ends with `!!` the launcher exits without starting M3.

## PIDs (Windows PIDs; launched 11-13 04:09 / 04:11 machine time)
- M4 `bash run_x.sh in mav`: **41948** (msys 46001)
- M3 launcher `bash launch_mav_teach_1113.sh`: **48508** (msys 46125); each variant runs as a child `bash run_x.sh in mav` with SX_TEACH set.
Stop only these, by PID, after checking the command line.

## Target-side choices (all mirror the indoor final chain, gproto/nolabel + wf24 README §6-7)
| item | value | source |
|---|---|---|
| real cache | E:/mmcache/mav6d_cn_obj | nolabel/make_nolabel_pseudo.py; camnorm_mav6d.yaml DATA_PATH |
| image size | raw 1920×1080 undistorted to K_pin (K_raw), input 512×288 | tools/build_mav6d_cache.py |
| crop zoom cap | 4 (zoom = 95/w2d in [1, 4]) | tools/kp_zoom.py z_range (indoor KZ, --kp2d-zoom 95) |
| RGB norm | [0.4360, 0.4418, 0.4374] / [0.2239, 0.2244, 0.2200] | mav6d_cn_obj/norm.txt = camnorm_mav6d.yaml NORM |
| det_base | mav6d_r34_geo_full_tr.yaml | nolabel/run_chain_nl.sh REAL |
| self-training recipe | rot 0, LR 2e-4, 3 epochs, hflip-only aug | st2nl075wa → st2pluswa → st_mav6d_r34_geo_full_tr |
| agree_rel | 0.5 / 3.5 | indoor 0.5 m at the 3.5 m median distance (wf23 build_targets) |
| val (unlabeled images for constants / s̄; labels only for the sanity score) | cache val split, 16 sequences (val_sequences.txt), 2,336 frames; sanity labels cmpeval/val_meta.npz | build_mav6d_cache split; nolabel val row |
| test split + scorer | 4,800 frames, cmpeval/test_meta.npz (= pack real_test_meta.npz, byte-identical), pack `real_test` rules (0.1/0.2/0.4/0.8 m, a = 1 m) | cmpeval/paper_table.metrics |
| raw frames | E:/MAV6D JPEGs undistorted on the fly (sx.imread_raw) | tools/kp_zoom.RawSource; check: resized result == cache rgb (mean abs diff 0.0) |

Labels: read only by `score_x.py` (val sanity + one test scoring at the end), as in the AntiUAV3D runs.

## Smoke test
`score_x.py` (SX_TGT=mav) on the paper's final nl075 chain predictions (re-created read-only from nolabel, r3_bk_train "paper" row):
test AP 56.44 / ATE 0.226 m / ASE 1.81 cm / AOE 13.54° / mDR 99.8 / **UDS 70.16**; val UDS 62.49 — identical to the paper numbers.
AntiUAV3D scoring unchanged after the glue edit (out2antifs test re-scored 41.735).

## Expected durations
AntiUAV3D chains took ~3–5 h each. MAV6D has ~20k frames (train 12,866 / val 2,336 / test 4,800) vs ~7k for antifs, so detector
inference and crop extraction (raw JPEG decode + undistort) are ~2–3× longer; refiner/orientation training steps are fixed-step.
Estimate (shared GPU, gated by gpu_slot.sh): M4 ≈ 5–8 h; M3 ir ≈ 2–3 h (reuses M4's detector/refiner), lid ≈ 5–8 h, none ≈ 5–8 h.
M3 total ≈ 12–19 h after M4 → everything ≈ 17–27 h from launch.

## Concerns
- `none` for the in source previously died in run_x.sh (plain-RGB nets existed only for out); now uses the indoor KZs0/KZs1 (epoch 30,
  same kp_cfg as the source). This is the only choice of networks made here — check it is the intended "plain RGB" control.
- Inherited deviations 1–3, 6 (STATUS.md) also apply here, notably refiner sim samples / w_sim from ramp s0 instead of the indoor irloc P0_s0.
- **Main concern — refiner width filter units.** refiner_x keeps boxes of projected width 6–170 px in RAW 1920-px units; the indoor
  original applies [6,170] to the 512×288 input (K_in). On MAV6D this keeps only far frames (bank 878 / 12,866 train frames), so the
  depth refiner sees ~7% of frames. Not changed (method-side constant, inherited from the outdoor reimplementation); M4 is running with it.
  Mirroring the original would mean [6,170]×3.75 raw px for mav — decide before trusting M4 / M3 numbers (fix = stop PID 41948 + 48508,
  delete in2mav/ref/{bank,real_d0_*} and later outputs, relaunch).
