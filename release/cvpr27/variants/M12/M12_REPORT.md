# M12 — MAV6D sim-only row (M2) + single-model / no-TTA PST on both targets, with timing (M1)

2026-11-13. Inference only (no training). All outputs are under `gproto/review1113/M12/`. No existing file was modified, and no process was killed. Every GPU job ran with `torch.cuda.set_per_process_memory_fraction(0.3)` (`memcap.py`) and batch size ≤ 8; the copied scripts had their internal batches lowered from 16/32/128/256 to 8. The MonoCD training (PID 29384) ran on the same GPU the whole time.

Metric order: AP, ATE, ASE (cm), AOE (°, quadrotor-symmetric), mDR (%), UDS. On MAV6D, ATE is in metres. On AntiUAV3D, ATE is **relative to the ground-truth distance** (pack `score.py`, `outdoor_test` rules, a = 25 %).

## 0. Reproduction check (before any new run): both deployed numbers reproduced exactly
| Target | Reported | Re-scored here | How |
|---|---|---|---|
| MAV6D test (nolabel nl075) | AP 56.44 / ATE 0.226 m / ASE 1.81 / AOE 13.54 / mDR 99.8 / UDS 70.16 | **identical**, and the val rows (62.49 / 61.51) and the w=0 test row (68.26) are also identical | `fuse_eval_m12.py`, a copy of `nolabel/fuse_eval_nl.py` (only the output path changed) + `chain_eval_m12.py`, a copy of `chain_eval_m_nl.py` (input dirs can be overridden by env; defaults = the deployed chain) → `M1/repro_full/results.json` |
| AntiUAV3D v2 test (out2antifs) | UDS 41.74 | AP 27.89 / ATE 0.099 / ASE 16.30 / AOE 17.11 / mDR 91.9 / **UDS 41.735**; predictions are bit-identical to the deployed `pcmd_fused_test.pkl` (max abs diff 0) | `antilib/chain_x.py deploy` (a copy of `srcx/lib`, with `sx.O`/`sx.W` redirected to M12) on copies of the deployed inputs → `anti/repro/` |

## M2 — MAV6D "Sim. only (our detector)"
- **Detector.** The starting point of the label-free chain is `release_20260920/ramp_s{0,1,2}.pth`. `run_proto.train` initialises every nl075 detector from `INIT.get(name, ramp_s%d)`, and `nl075` has no INIT override.
- **Configuration, as deployed.** 3-seed ensemble + flip TTA, label-free self-calibrated s̄ on val images (conf ≥ 0.7), and no temporal background (strict single frame, like the nl chain and srcx `srconly`). The detector's own depth/size/rotation heads are used. There is no refiner and no orientation student.
- **Not reused.** The existing `geo1004/sim3_ens` (11-06) used `--temporal-bg causal100` (cross-frame), so I did not reuse it and produced a new dump.
- **Scorer.** The same metric code as `fuse_eval_nl.py`: `cmpeval/paper_table.metrics` on `cmpeval/test_meta.npz`, top-confidence box per frame, no re-ranking. Test labels were read once.

| MAV6D test (4,800 frames) | AP | ATE (m) | ASE (cm) | AOE (°) | mDR (%) | UDS |
|---|---|---|---|---|---|---|
| Sim. only (our detector, 3 seeds + flip) | **26.87** | **0.283** | **1.74** | **20.32** | **69.8** | **49.51** |
| PST full (reference, reproduced) | 56.44 | 0.226 | 1.81 | 13.54 | 99.8 | 70.16 |

(For reference only, not rerun: the AntiUAV3D v2 counterpart `srcx/out2antifs/preds/srconly_test` scores 16.04 / 0.132 / 15.36 / 26.06 / 78.5 / 30.55.)

## M1 — single model, no TTA
**Setup.**
- The first seed of every network: detector s0 (no flip), depth refiner/student seed 5, orientation student s0 (no mirror view), with single-model "fusion".
- The height network is unchanged; it is one network in the deployed chain too.
- Frozen hyperparameters are kept from the deployed chain: w = 0.1579 (MAV6D) / 0.0964 (AntiUAV3D), α = 0.5, LAM, thresholds, and the BN-adapted student weights.
- Image-statistic constants are recomputed by the deployed recipe from the single model's own outputs on unlabeled images: s̄ (val images), colour centres, c_k (train), b_k (val images).
- One shortcut: the height-ratio readings on the **train** split are reused from the deployed chain (MAV6D `shape_train_nl075_k0.pkl`; AntiUAV3D `tr1` crops for the height-net BN). These come from the ensemble detector's train windows. Val/test height readings were recomputed on the single detector's windows.
- Test labels were read once per configuration.

| Target | Config | AP | ATE | ASE (cm) | AOE (°) | mDR (%) | UDS |
|---|---|---|---|---|---|---|---|
| MAV6D test | PST full (3 det + flip, 2 ref, 2 stu + mirror) | 56.44 | 0.226 m | 1.81 | 13.54 | 99.8 | 70.16 |
| MAV6D test | **PST single, no TTA** | **54.44** | **0.236 m** | **1.79** | **14.39** | **99.75** | **68.86** |
| AntiUAV3D v2 test | PST full | 27.89 | 9.9 % | 16.30 | 17.11 | 91.9 | 41.74 |
| AntiUAV3D v2 test | **PST single, no TTA** | **30.07** | **10.3 %** | **16.21** | **17.20** | **93.9** | **42.85** |

**Notes.**
- On AntiUAV3D, the single model scores higher than the full ensemble (AP +2.2, mDR +1.9). This is one seed and one evaluation, so seed variance is unknown; it should not be read as "the ensemble hurts" without more seeds.
- Val sanity checks (not used for any choice):
  - MAV6D val: single 60.60 vs full 62.49.
  - AntiUAV3D val: single 33.44 vs full 34.98 (`out2antifs` run log, "val sanity").
- MAV6D b_k gained a third class (−1, 6 val frames whose crop has no silhouette brightness), produced automatically by the frozen rule. It affects at most 1 test frame.

## Timing and memory (MAV6D test, RTX 5090 Laptop GPU)
- Batch size 1, 30 warm-up frames, then **300 timed frames**. Times are in ms per frame, measured with `torch.cuda.synchronize`.
- Each stage uses the deployed code path: the eval_camnorm ensemble/flip hooks, the final_unc crop + refiner forward, the adabn_kz zoom recut + student forward (+ mirror), the KPS head, and SQPnP.
- **Upper bound:** during every timing run the GPU was 99 % utilised by other jobs (MonoCD training + other sessions, about 16 GB in use), so all times are inflated, the small-network stages most of all.
- Peak memory is `torch.cuda.max_memory_allocated` with all models loaded.

| Stage (mean / median ms) | PST full (7 nets + mirror + height net) | PST single (3 nets + height net) | Detector only (s0, no flip) |
|---|---|---|---|
| Detector | 148.8 / 127.6 (3 seeds × flip = 6 passes) | 42.3 / 32.5 | 39.5 / 29.6 |
| Depth refiner(s) | 61.8 / 42.2 (2 nets) | 37.8 / 22.8 (1) | – |
| Orientation students | 81.6 / 64.8 (2 nets × orig + mirror) | 29.8 / 24.3 (1 × orig) | – |
| Height network | 23.9 / 15.1 | 22.4 / 17.6 | – |
| PnP (CPU) | 0.3 (5 solves) | 0.2 (2 solves) | – |
| **Model chain total** | **316.3 / 291.9** (p90 450) | **132.5 / 107.6** (p90 282) | **39.5 / 29.6** |
| + raw-frame read/undistort for zoom crop (I/O) | 15.1 | 13.9 | – |
| (excluded) cache read + host→GPU of detector input | 59.5 / 20.9 | 21.2 / 17.4 | 17.6 / 14.6 |
| Weights resident | 738 MB | 371 MB | 108 MB |
| **Peak GPU memory (max_memory_allocated)** | **810 MB** | **431 MB** | **150 MB** |

The single-model chain is about 2.4–2.7× faster than the full chain (mean / median), at −1.30 UDS on MAV6D and +1.11 UDS on AntiUAV3D. The AntiUAV3D chain uses the same network architectures and 512×288 inputs, so its per-frame cost is the same; it was not timed separately.

## Exact commands
These are bash commands. Common env: `PYTHONUTF8=1 PYTHONPATH=E:/Open3DUAVDet TORCH_HOME=E:/torch_home`. `PY=D:/Miniconda3/envs/city/python.exe`; `M=.../review1113/M12`; `NL=.../gproto/nolabel`; `REAL=cfgs/models/uavdet_3d/camnorm/mav6d_r34_geo_full_tr.yaml`; `SETS="--set MODEL.POST_PROCESSING.SIZE_SOURCE pred MODEL.POST_PROCESSING.SIZE2D_MODE width"`.

**M2**
```
cd E:/Open3DUAVDet/tools
DUMP_PKL=$M/M2/sim3nb_test.pkl $PY $M/memcap.py $NL/eval_dump_nl.py --cfg $REAL --ckpt release_20260920/ramp_s0.pth --ensemble ramp_s1.pth ramp_s2.pth \
   --flip-tta --tag m12_sim3nb_test --split test --interval 1 --workers 0 --match greedy --self-calib val $SETS
$PY $M/score_dump_m12.py $M/M2/sim3nb_test.pkl test $M/M2/score_sim3nb_test.json
```

**M1 MAV6D**
```
# single detector dumps (test / val / train)
DUMP_PKL=$M/M1/single_<sp>.pkl $PY $M/memcap.py $NL/eval_dump_nl.py --cfg $REAL --ckpt .../st2nl075wa_mav6d_r34_geo_full_tr/NL075_s0/ckpt/checkpoint_epoch_3.pth \
   --tag m12_single_<sp> --split <sp> --interval 1 --workers 0 --match greedy --self-calib val $SETS
# orientation windows (wrapper = proto/dump_orient_p.py recipe, pos = NL075_s0 only, --no-flip --bg none, out root M12/M1/repl)
$PY $M/memcap.py $M/dump_orient_single.py test|val
# remaining steps
bash $M/run_m1_mav.sh
#   = recut_mirror_m12.py (S6I_s0) -> final_unc_m12.py nl075 z2 XFZnl075 5 val,test,train
#     -> kps_shape_m12.py nl075 shape val test
#     -> fuse_eval_m12.py with FE_UNC=z2 FE_ORI=ouS1 FE_MEMS=0 M12_WS=w
# reproduction
NL_ARM=nl075 FE_UNC=z2 FE_ORI=ouS6I2M FE_NMEM=4 M12_OUT=$M/M1/repro_full $PY $M/fuse_eval_m12.py
```

**M1 AntiUAV3D**
```
# reproduction: chain_x.py deploy 0.0964 pcmd on copies of the deployed inputs in anti/repro
#   -> score_x.py anti/repro/preds/pcmd_fused_test.pkl antifs_test
bash $M/run_m1_anti.sh
#   = det_infer_x.py (M12_NOFLIP=1) --ckpt srcx/st_out2antifs/DET1_s0 --splits train val test --size-fallback out2antifs/recs/det0.pkl
#     -> ori_x.py crops val1/te1 -> ORI_MIRROR=0 ori_x.py read ... out2antifs/ou/S_ab0.pth
#     -> ori_x.py kps OKZkps_s0 tr1,val1,te1 -> refiner_x.py infer out2antifs/ref/ref_st3_s5.pt (train/val/test)
#     -> chain_x.py deploy 0.0964 single -> score_x.py preds/single_fused_test.pkl antifs_test
```

**Timing**
```
cd E:/Open3DUAVDet/tools; $PY $M/timing_m12.py full|single|det 300 30
```
Output: `M1/timing/timing_*.json`.

## Files
- **Scripts (all in M12).** Wrappers: `memcap.py`, `dump_orient_single.py`, `score_dump_m12.py`, `timing_m12.py`, `run_m1_mav.sh`, `run_m1_anti.sh`. Copies (each header lists its changes): `chain_eval_m12.py`, `fuse_eval_m12.py`, `final_unc_m12.py`, `recut_mirror_m12.py`, `kps_shape_m12.py`, `antilib/*`.
- **Results.**
  - `M2/score_sim3nb_test.json`
  - `M1/repro_full/results.json`
  - `M1/single_result/results.json`
  - `anti/repro/preds/pcmd_fused_test.pkl.score_antifs_test.json`
  - `anti/single/preds/single_fused_test.pkl.score_antifs_test.json` and `anti/single/deploy_single.json`
  - `M1/timing/*.json`
  - Logs in `M1/logs/`, `anti/single/logs/`.
- **Test-label use.** One scoring per configuration: M2 once, MAV6D single once, AntiUAV3D single once, plus the reproduction re-scoring of the two frozen deployed chains.
- **Ground-truth fields in the orientation dump.** The detection-rate fields are computed inside `dump_real_preds` (same as the deployed chain's dumps). They are sealed into `sealed_gt_*.pkl`, and their console print was masked.
