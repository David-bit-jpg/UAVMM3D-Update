# srcx — source-domain ablation of PCMD (status)

Pipeline: `run_x.sh <in|out|mix> <anti|mav>` → `srcx/<src>2<tgt>/` (big crops: `D:/S2MUAV3D/srcx_work/<combo>/`). Libs in `srcx/lib/` are copies of
`gproto/outdoor/*_out.py` with all source/target constants moved to `lib/sx.py` (no method change). Per-combo detector configs written by
`lib/mkcfg.py` into `tools/cfgs/models/uavdet_3d/camnorm/srcx/`. Log per combo: `srcx/<combo>/logs/run_x.log` (ends with ALL DONE or `!!`).

Order (user, 11-11): in→anti, out→anti, mix→anti; MAV6D combos only after all three anti results.

## Running
- 06:5x launched in→anti and out→anti (detached).
- out source detector seeds 1,2: `stage1_det.sh` via `launch_det_out12.sh` (heavy slot, sequential).
- out source irlid/kdir/lid seed 1: trained by the earlier run_orig chain (detached shells); run_x waits for the files.

## Deviations from the original indoor method (with reason)
1. Refiner stage 1 is retrained for every combo (incl. in→anti) instead of reusing indoor V2P_s{5,6}_st: stage 1 composites sim crops onto a
   REAL background bank, which is target-specific (the indoor V2P used MAV6D backgrounds). Same network/loss/recipe (refiner_out = DCNetU).
2. Refiner sim samples and w_sim use the sim-only ramp detector s0 of the source (indoor original used the irloc detector P0_s0 for these);
   the outdoor/mix sources have no irloc detector, so ramp s0 is used for all sources uniformly.
3. Metre thresholds converted by distance ratio (as in the compact outdoor chain): sim matching 25% of distance (indoor 1 m), pseudo-label
   agreement 14.3% of distance (indoor 0.5 m at 3.5 m). Orientation-crop zoom cap on AntiUAV = 18.75 (target camera/distance range);
   on sim validation crops = source's own cap (indoor 4, outdoor 12.5).
4. Real detector config per combo = target data + SOURCE decoder statistics (DEPTH/SIZE mean/std), required for correct decoding.
6. s̄ self-calibration fallback: if the self-trained detector ensemble has NO conf ≥ 0.7 box on unlabeled val images (happened for in→anti:
   150 pseudo-labels only), s̄ is taken from the stage-1 detector ensemble's self-calibration (det0, also unlabeled val images). The indoor
   final also keeps s̄ unchanged across the chain (wf24 §7 "s̄ 不变"). Applied by `det_infer_x.py --size-fallback`, only when the set is empty.
5. gpu_slot.sh fix (shared): count() used `(...).Count`, which is empty for exactly one matching process (treated as 9 = busy);
   now `@(...).Count`; regex also counts *_x.py jobs.

## 11-12 additions (AntiUAV3D v2 = target `antifs`)
- Target `antifs` (sx.py): D:/mmcache/outdoor_real_fs, split by D:/S2MUAV3D/real_outdoor_antiuav3d_fs (intrinsics verified against images, sequence-disjoint focal/distance-balanced). Frames train/val/test = 3,293 / 1,148 / 2,624. Scoring: score_x.py score_meta() on labels/sealed/antifs_test_meta.npz (= pack score.score code).
6b. s̄ self-calibration has TWO fallbacks in lib/det_infer_x.py:
   - stage-1 detector ensemble (det0, no --size-fallback): if no conf>=0.7 box on unlabeled val images -> median over the top-10% most confident val boxes (>= 20). Used by in→antifs (top-57, min conf 0.440).
   - self-trained ensemble (det1, --size-fallback recs/det0.pkl): if no conf>=0.7 box -> reuse det0's s̄ (item 6). Used by in→anti (v1).
7. run_x.sh: self-trained detectors run with --workers 0 through the light GPU slot (data loading only; recipe unchanged).

## 11-13 MAV6D target
Target `mav` added to lib/sx.py (combos `in2mav` = M4 clean protocol, `in2mav_{ir,lid,none}` = M3 teacher swaps). Every target-side choice
mirrors the original indoor final chain (gproto/nolabel, wf24 README §6-7); nothing on the method side changed.
| item | value | source |
|---|---|---|
| real cache | E:/mmcache/mav6d_cn_obj (official object box, undistorted to K_pin) | nolabel make_nolabel_pseudo.py / camnorm_mav6d.yaml DATA_PATH |
| image size | raw 1920x1080 (undistorted K_raw = K_pin), input 512x288 | tools/build_mav6d_cache.py |
| crop zoom cap | 4.0 (zoom = 95 / w2d clipped to [1, 4]) | tools/kp_zoom.py z_range=(1,4), --kp2d-zoom 95 (indoor KZ) |
| RGB norm (orientation crops) | mean [0.4360,0.4418,0.4374], std [0.2239,0.2244,0.2200] | mav6d_cn_obj/norm.txt = camnorm_mav6d.yaml NORM (600 unlabeled train frames) |
| det_base (real detector cfg) | mav6d_r34_geo_full_tr.yaml (+ indoor-source decoder stats, identical to its own) | nolabel run_chain_nl.sh REAL |
| self-training cfg | mkcfg recipe = st_mav6d_r34_geo_full_tr (rot 0, LR 2e-4, 3 ep) + st2pluswa weak aug (hflip only) = st2nl075wa | st2nl075wa -> st2pluswa -> st2plus -> st_mav6d |
| pseudo-label agreement | 0.5 / 3.5 (0.5 m at the 3.5 m median distance) | wf23 build_targets (indoor 0.5 m) |
| val sanity labels | geo1004/cmpeval/val_meta.npz (2,336 frames, 16 sequences) | nolabel fuse_eval_nl / chain_eval_m_nl (val row 62.49) |
| test scoring | cmpeval/test_meta.npz (4,800 frames; byte-identical to pack data/gt/real_test_meta.npz), pack rules `real_test` (0.1/0.2/0.4/0.8 m, a = 1 m) | cmpeval/paper_table.metrics == pack score.py real_test |
| raw frames | E:/MAV6D/<cls>/JPEGImages/... undistorted to K_raw on the fly (sx.imread_raw) | tools/kp_zoom.RawSource (bit-identical: resized remap == cache rgb, mean abs diff 0.0) |
Glue edits (all backed up as *.bak_1113): sx.py ('mav' + imread_raw); ori_x.py / refiner_x.py (raw read via sx.imread_raw, identical for
AntiUAV3D metas with raw_path); make_pseudo_x.py (copy raw_path only if present); score_x.py (rules from T['rules']; real_test bins by z
as in the pack; val npz meta); run_x.sh (SX_TEACH=none with the in source uses the indoor plain-RGB KZ nets KZs0/KZs1 epoch 30 = indoor twin of
OKZrgb_s0/1; previously died for src != out).
Smoke tests: score_x (SX_TGT=mav) on the paper's final nl075 predictions -> test 56.44 / 0.226 / 1.81 / 13.54 / 99.8 / UDS 70.16, val 62.49
(= R123 reproduction); out2antifs test re-scored unchanged (41.735).
Deviations 1-3, 6 above apply unchanged (refiner stage 1 retrained on MAV6D backgrounds; refiner sim samples / w_sim from ramp s0, not irloc P0_s0).
Running: see gproto/review1113/MAV_CHAIN_SETUP.md.
**11-13 concern (not changed, needs a decision):** refiner_x det_crop / bank keep boxes with projected width 6–170 px measured in RAW
pixels (K_raw, 1920 wide). The indoor original (cropw_extract / extract_nbm_nl / xpaste_bank_sf2_nl) applies the same [6, 170] to the
512x288 INPUT image (K_in) and crops from the 512x288 cache. On MAV6D (drone 0.3 m at 1.6–5.6 m ≈ 100–700 raw px) the raw-px rule keeps
only far frames: in2mav bank = 878 of 12,866 train frames (indoor original: almost all conf≥0.5 frames). Depth refiner adaptation,
pseudo-label correction and deployment refinement therefore only see far frames. The equivalent rule would be [6,170]·(1920/512) raw px.

### 11-13 修正：MAV6D 目标的宽度门限单位
- **问题**：`refiner_x` 里的 6–170 px 宽度门限，原来是按原始 1920 px 图像的内参算的。最初的室内链用的是 512 px 输入图像的内参：wf21/strict 的检测记录 K 中 fx=492，见 nolabel/extract_nbm_nl.py:42。
- **后果**：in2mav 的背景库只剩 878 个窗口，全部是远距离的。
- **修法**：`sx.T['wscale']`，mav 取 3.75，宽度门限改为在输入像素上判断。只有真实图像、原始内参的两处调用用它，即 bank 和 det_crop 处理真实图像时；仿真样本用的是 K_in，系数不变，仍是 1。anti / antifs 也不变，系数是 1.0。
- **AntiUAV3D 不受影响**：门限按原始像素算，在 out2antifs 上只去掉了少数窗口（909 个置信 ≥0.5 的帧里留下 862 个），已报告的结果不重跑。
- **处置**：in2mav 中止后重启。det0、srconly 预测和仿真样本保留，因为它们与这个门限无关；错误的 bank 改名为 bank_rawgate_bad_1113.npy。
