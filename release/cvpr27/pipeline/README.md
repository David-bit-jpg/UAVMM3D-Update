# pipeline: the clean PST chain used by the paper (gproto/srcx)

`bash srcx/run_x.sh <in|out> <mav|antifs>` runs the chain for one source/target pair: simulation-only detectors x3 (ensemble + flip) -> Step 1 pseudo box
(depth refiner seeds 5, 6) -> self-trained detectors x3 -> Step 2 teachers (IR+LiDAR, IR-distilled, LiDAR; 2 each) and orientation students (seeds 0, 1)
-> Step 3 pseudo depth and height ratio -> fusion -> one scoring of the test split (`lib/score_x.py`). `SX_TEACH=none` gives the RGB-only-teacher control.
Feeds: Tab. 1 CenterNet* and full pipeline (MAV6D: `in2mav`), Tab. 2 rows 1 and 4, Tab. 3 run 1 of PST (`in2mav`, `out2antifs`) and of RGB-only (`*_none`).
The RealDrone3D full-pipeline row of Tab. 1 is `variants/k249` (same chain, frames restricted to the 249 given to the baselines).
`stage1_det.sh` trains the outdoor simulation-only detectors (seeds 0-2); the indoor ones are the release checkpoints named in `lib/sx.py`.
`runs/<run>/logs/run_x.log` is the log of the reported run. Code edited after `out2antifs` / `out2antifs_none`: see `../repro/code_time_audit.md` and the `.bak_*` files.
`tools/cfgs/` mirrors `E:/Open3DUAVDet/tools/cfgs/` (copy it over the repository's `tools/cfgs/` to use it).
