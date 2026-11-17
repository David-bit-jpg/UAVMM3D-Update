# CVPR'27 PST paper: all experiment code in one place

Copies (not moves) of the code that produced the numbers of the paper, collected 11-17. The model code of this repository (`uavdet3d/`, `tools/`) is the
detector/teacher framework; this folder adds the paper's experiment chain, every variant, the baselines and the advisor's patches.
Original locations are kept in the paths below; all scripts use absolute paths of the author's machine (`E:/Open3DUAVDet`, `D:/...`) and are documented, not rewritten.

| folder | content | original location |
|---|---|---|
| `pipeline/srcx/` | the paper's clean chain: `run_x.sh` + `lib/` (`.bak_*` = earlier versions, see `repro/code_time_audit.md`), stage-1 detector scripts | `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/` |
| `pipeline/tools/cfgs/` | every cfg yaml the chain uses: `camnorm/srcx/*`, `camnorm/ourdet_baseproto/*`, the stage-1 configs named in `srcx/lib/sx.py`, and all `_BASE_CONFIG_` parents | `E:/Open3DUAVDet/tools/cfgs/` |
| `pipeline/runs/` | run logs and small json of the reported chain runs (in2mav, in2mav_none, out2antifs, out2antifs_none, out2antifs_k249) | `gproto/srcx/<run>/` |
| `variants/` | every experiment under `gproto/review1113/` (run scripts, per-run `lib/` copies, notes, logs); `variants/README.md` says what feeds which table/figure/supplement section | `gproto/review1113/` |
| `baselines/cmpfull/` | the five ported baselines (GUPNet, MonoCD, MonoDGP, MonoDETR, MonoCoP), pseudo-label builders, `angles.py`, `SPEC.md` | `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/` |
| `baselines/advisor_pack/NN_*` | the pack sent to the advisor and every patch, numbered in the order applied (code, scripts, notes, small pseudo-label tables) | `D:/UAV3D_baselines_pack`, `D:/UAV3D_pack_*` |
| `baselines/ADVISOR_RUNS.md` | which advisor run (method / seed / step / target) produced which Tab. 1 number; patch order | |
| `repro/` | `MANIFEST.md`, `manifest.json`, `rescore_all.py` (+ helpers) and `rescore_report.md`, `code_time_audit.md`, `FILES_SHA256.tsv` (every file of this folder), `ARTIFACTS.md` (data, checkpoints, predictions not copied: path, size, sha256) | `gproto/repro_1117/` |

Reproduce every number of Tab. 1-3 from the saved predictions (CPU): `python repro/rescore_all.py` (paths inside point to the prediction files listed in `repro/ARTIFACTS.md`).
Retraining is not bit-exact (cudnn benchmark, AMP, Windows data loading, shared GPU); see `repro/README.md`.
