# Seed-spread runs for PST and RGB-only (written 11-13, NOT run)

## Launch
```
bash E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/seeds/run_all_seeds.sh
```
To make sure it does not overlap the sup3 seed run that is already going:
```
WAIT_FOR=E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run/logs/run_seed2.log bash .../run_all_seeds.sh
```
`DRY=1 bash run_all_seeds.sh` prints the config of all 7 runs. It only builds the lib copies and uses no GPU.

## Runs
The runs go strictly one after another. Each run's output dir is `review1113/seeds/<RUN>/` and its log is `<RUN>/logs/run.log`.

| # | RUN | Target | Teachers | Shift | Effective seeds (refiner / detectors / students) | Fixed stage-1 outputs copied from |
|---|---|---|---|---|---|---|
| 1 | `mav_none_s10` | MAV6D | RGB-only | 10 | 15,16 / 10–12 / 10,11 | `srcx/in2mav_none` |
| 2 | `mav_full_s20` | MAV6D | PST | 20 | 25,26 / 20–22 / 20,21 | `srcx/in2mav` |
| 3 | `mav_none_s20` | MAV6D | RGB-only | 20 | 25,26 / 20–22 / 20,21 | `srcx/in2mav_none` |
| 4 | `antifs_full_s10` | AntiUAV3D | PST | 10 | 15,16 / 10–12 / 10,11 | `srcx/out2antifs` |
| 5 | `antifs_none_s10` | AntiUAV3D | RGB-only | 10 | 15,16 / 10–12 / 10,11 | `srcx/out2antifs_none` |
| 6 | `antifs_full_s20` | AntiUAV3D | PST | 20 | 25,26 / 20–22 / 20,21 | `srcx/out2antifs` |
| 7 | `antifs_none_s20` | AntiUAV3D | RGB-only | 20 | 25,26 / 20–22 / 20,21 | `srcx/out2antifs_none` |

MAV6D PST with shift 10 is `review1113/sup3/run_seed2.sh`, which is already running. It is not repeated here. Once everything finishes, each of the 4 arms has 3 runs: shift 0 (the reported run), shift 10 and shift 20.

## Files
- **`make_seeds_lib.py <teach> <src> <tgt> <shift>`** generalizes `sup3/make_seed_lib.py` and reuses `review1113/sup_patch.py`, so the patches are identical to sup3. It writes `<RUN>/lib/`. The diff against `sup3/lib` is only the paths and tags, plus a `SX_NOMKDIR` guard used by the dry run.
- **`run_seed_chain.sh <teach> <src> <tgt> <shift>`** is `sup3/run_seed2.sh` generalized:
  - It adds the `teach=none` branch of `srcx/run_x.sh`: the 2 plain-RGB orientation nets (`KZs0,1` for the in source, `OKZrgb_s0,1` for the out source), a gate equal to their agreement, a refiner without IR (`sx.NOIR`), and a constant height aspect. The constant-height step reads this run's own crops.
  - It adds `SERIAL=1`, which runs the paired trainings sequentially instead of with the `run_x.sh` overlap.
- **`run_all_seeds.sh`** is the sequential wrapper:
  - It skips any run whose `run.log` already has `ALL DONE:`.
  - A failed run is logged and the wrapper moves on (`STOP_ON_FAIL=1` stops instead). Re-launching retries the failed runs and resumes them.
  - Wrapper log: `seeds/logs/run_all_seeds.log`. Per-run stdout: `seeds/logs/<RUN>.stdout`.

## Isolation
Every new name carries `C_sd<shift>`, where `C` = `in2mav`, `in2mav_none`, `out2antifs` or `out2antifs_none`. The dry run confirmed that none of these exists yet:
- configs: `tools/cfgs/models/uavdet_3d/camnorm/srcx/{real,st}_<C>_sd<shift>.yaml`;
- pseudo-label index: `<cache>/train/index_pseudo_srcx_<C>_sd<shift>.pkl`. The cache is `E:/mmcache/mav6d_cn_obj` or `D:/mmcache/outdoor_real_fs`;
- self-trained detectors: `output/models/uavdet_3d/camnorm/srcx/st_<C>_sd<shift>/DET1_s{0,1,2}/`;
- crops: `D:/S2MUAV3D/srcx_work/<C>_sd<shift>/`, about 9 GB per MAV6D run and less for AntiUAV3D.

Nothing under `srcx/` is written.

## Kept fixed (same decision as sup3)
- **Stage-1 networks of the source.** These are the sim-only detectors, the IR+LiDAR / IR-distilled / LiDAR teachers, the height net and the plain-RGB nets. They are pretrained and `run_x.sh` does not train them.
- **Their outputs are copied (`COPY_FIXED=1`).** The copy comes from the reported run with the same teacher setting and covers `det0.pkl`, `srconly_{val,test}`, the refiner sim samples, bank and real d0 crops, and `wsim_cache.pkl`.
- **AntiUAV3D runs** keep the outdoor (CARLA) source networks, `sx.SOURCES['out']`.
- **Seed spread covers adaptation only.** It covers the seeds of what the chain adapts, not the source pretraining.
- **Code check.** The current `srcx/lib` differs from the version that produced `out2antifs` / `out2antifs_none` only in MAV6D-only branches: `imread_raw`, `wscale` and `real_test` rules. I checked the `*.bak_1113` diffs, so the AntiUAV3D code path is unchanged.

## Robustness (as sup3)
- Every training step resumes from its last checkpoint. Refiner: per epoch / every 250 steps. Students: every 250 steps. Detectors: `train.py` epoch auto-resume.
- Every step is skipped once its output exists.
- CUDA OOM halves the batch and retries, up to 4 times. Retries go to `logs/retry.log`, and a halved training batch is flagged as a RECIPE DEVIATION.
- DataLoader workers = 0 everywhere.
- Inference runs through `gpu_slot.sh light` plus `infer_wrap.py` (memory fraction 0.2).
- No process is ever killed.
- The last line of each `run.log` is `ALL DONE: test UDS .. AP .. ATE .. ASE .. AOE .. mDR .. | val UDS .. AP .. ATE .. ASE .. AOE .. mDR .. | <RUN> w .. | reported (shift 0): ..`.

## Estimated runtime
Taken from the `run_x.log` timestamps and output mtimes. The copied fixed steps are excluded.

| Reported run | Window | Duration |
|---|---|---|
| in2mav | 04:55 → 08:24 | 3 h 29 min |
| in2mav_none | 08:51 → 12:06 | 3 h 15 min |
| out2antifs | 02:08 → 05:04 | 2 h 56 min |
| out2antifs_none | 08:42 → 13:31 | 4 h 49 min, of which about 2 h 03 min was waiting for the OKZrgb nets to finish training → about 2 h 46 min |

**Total for the 7 runs:** 3.25 + 3.5 + 3.25 + 2.9 + 2.8 + 2.9 + 2.8 ≈ **21.5 h** with the GPU to itself. Allow **about 25–28 h** with `gpu_slot` queueing. If the wrapper waits for sup3, add the rest of that run (about 3.5–5 h).

## Checks done
- `bash -n` passes on all three scripts.
- `DRY=1` passes for all 7 runs: paths, tags, `NOIR`, score split, checkpoints present, the 10 fixed files present, and no target name exists yet.
- `py_compile` passes on all 7 lib copies.
