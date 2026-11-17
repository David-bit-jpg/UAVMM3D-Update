# Code files modified after the start of a reported run

Files written within 3 min after the start are the per-run lib copies / generated configs made by the launcher itself and are not counted as changes.
mtime of each code file vs the run window (start = last start line in the run log before the prediction file; end = prediction file mtime).
A file listed here is the CURRENT version; the version that ran is the .bak_* copy only when a matching backup exists (see notes in README).

## srcx/in2mav (Tab.1/2 full pipeline MAV6D, Tab.3 PST run 1)
window 11-13 04:55:34 -> 11-13 08:24:18 (last "start" log line before the prediction file)
- no code file modified after the start
- 2 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change
- backups present: `make_pseudo_x.py.bak_1113` (11-13 04:07:46), `ori_x.py.bak_1113` (11-13 04:07:46), `refiner_x.py.bak_1113` (11-13 04:07:46), `refiner_x.py.bak_1113w` (11-13 04:55:00), `score_x.py.bak_1113` (11-13 04:07:44), `sx.py.bak_1113` (11-13 04:07:44), `sx.py.bak_1113w` (11-13 04:55:00)

## srcx/in2mav_none (Tab.3 RGB-only run 1)
window 11-13 08:28:35 -> 11-13 12:06:10 (last "start" log line before the prediction file)
- no code file modified after the start
- 2 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change
- backups present: `make_pseudo_x.py.bak_1113` (11-13 04:07:46), `ori_x.py.bak_1113` (11-13 04:07:46), `refiner_x.py.bak_1113` (11-13 04:07:46), `refiner_x.py.bak_1113w` (11-13 04:55:00), `score_x.py.bak_1113` (11-13 04:07:44), `sx.py.bak_1113` (11-13 04:07:44), `sx.py.bak_1113w` (11-13 04:55:00)

## srcx/out2antifs (Tab.3 PST run 1 RD; CenterNet* RD det0 reused by k249)
window 11-12 01:58:06 -> 11-12 05:04:00 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/chain_x.py` (mtime 11-12 08:19:12)
- CHANGED during run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/det_infer_x.py` (mtime 11-12 02:05:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/make_pseudo_x.py` (mtime 11-13 04:08:06)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/ori_x.py` (mtime 11-13 04:08:06)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/refiner_x.py` (mtime 11-13 04:55:24)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/score_x.py` (mtime 11-13 04:08:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/sx.py` (mtime 11-13 04:55:00)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/run_x.sh` (mtime 11-13 04:08:26)
- 2 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change
- backups present: `make_pseudo_x.py.bak_1113` (11-13 04:07:46), `ori_x.py.bak_1113` (11-13 04:07:46), `refiner_x.py.bak_1113` (11-13 04:07:46), `refiner_x.py.bak_1113w` (11-13 04:55:00), `score_x.py.bak_1113` (11-13 04:07:44), `sx.py.bak_1113` (11-13 04:07:44), `sx.py.bak_1113w` (11-13 04:55:00)

## srcx/out2antifs_none (Tab.3 RGB-only run 1 RD)
window 11-12 08:28:27 -> 11-12 13:31:44 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/make_pseudo_x.py` (mtime 11-13 04:08:06)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/ori_x.py` (mtime 11-13 04:08:06)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/refiner_x.py` (mtime 11-13 04:55:24)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/score_x.py` (mtime 11-13 04:08:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib/sx.py` (mtime 11-13 04:55:00)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/run_x.sh` (mtime 11-13 04:08:26)
- 2 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change
- backups present: `make_pseudo_x.py.bak_1113` (11-13 04:07:46), `ori_x.py.bak_1113` (11-13 04:07:46), `refiner_x.py.bak_1113` (11-13 04:07:46), `refiner_x.py.bak_1113w` (11-13 04:55:00), `score_x.py.bak_1113` (11-13 04:07:44), `sx.py.bak_1113` (11-13 04:07:44), `sx.py.bak_1113w` (11-13 04:55:00)

## srcx/out2antifs_k249 (Tab.1 full pipeline RD)
window 11-17 12:12:24 -> 11-17 14:36:22 (last "start" log line before the prediction file)
- no code file modified after the start
- backups present: `make_pseudo_x.py.bak_1113` (11-17 12:10:10), `ori_x.py.bak_1113` (11-17 12:10:10), `refiner_x.py.bak_1113` (11-17 12:10:10), `refiner_x.py.bak_1113w` (11-17 12:10:10), `score_x.py.bak_1113` (11-17 12:10:08), `sx.py.bak_1113` (11-17 12:10:10), `sx.py.bak_1113w` (11-17 12:10:10)

## review1113/sup3 (Tab.3 PST run 2 MAV6D)
window 11-13 14:31:03 -> 11-13 17:37:38 (last "start" log line before the prediction file)
- no code file modified after the start
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/seeds/mav_full_s20 (Tab.3)
window 11-13 20:04:27 -> 11-13 22:51:14 (last "start" log line before the prediction file)
- no code file modified after the start
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/seeds/mav_none_s10 (Tab.3)
window 11-13 17:38:21 -> 11-13 20:04:24 (last "start" log line before the prediction file)
- no code file modified after the start
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/seeds/mav_none_s20 (Tab.3)
window 11-13 22:51:18 -> 11-14 01:26:12 (last "start" log line before the prediction file)
- no code file modified after the start
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/seeds/antifs_full_s10 (Tab.3)
window 11-14 01:26:17 -> 11-14 03:14:44 (last "start" log line before the prediction file)
- no code file modified after the start
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/seeds/antifs_full_s20 (Tab.3)
window 11-14 05:57:11 -> 11-14 13:05:24 (last "start" log line before the prediction file)
- no code file modified after the start
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/seeds/antifs_none_s10 (Tab.3)
window 11-14 03:14:48 -> 11-14 05:57:06 (last "start" log line before the prediction file)
- no code file modified after the start
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/seeds/antifs_none_s20 (Tab.3)
window 11-14 13:05:28 -> 11-14 15:41:38 (last "start" log line before the prediction file)
- no code file modified after the start
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/sensorfree/E1/mav_s0 (Tab.3)
window 11-15 06:54:44 -> 11-15 06:55:52 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/make_sf_lib.py` (mtime 11-17 04:33:04)

## review1113/sensorfree/E1/mav_s10 (Tab.3)
window 11-15 06:56:30 -> 11-15 12:24:36 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/make_sf_lib.py` (mtime 11-17 04:33:04)
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/sensorfree/E1/mav_s20 (Tab.3)
window 11-15 13:26:03 -> 11-15 18:44:00 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/make_sf_lib.py` (mtime 11-17 04:33:04)

## review1113/sensorfree/E1/antifs_s0 (Tab.3)
window 11-15 02:37:35 -> 11-15 04:48:40 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/make_sf_lib.py` (mtime 11-17 04:33:04)
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/sensorfree/E1/antifs_s10 (Tab.3)
window 11-15 06:46:38 -> 11-15 06:54:16 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/make_sf_lib.py` (mtime 11-17 04:33:04)

## review1113/sensorfree/E1/antifs_s20 (Tab.3)
window 11-15 06:56:30 -> 11-15 10:41:50 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/make_sf_lib.py` (mtime 11-17 04:33:04)

## review1113/sensorfree/E4/mav_s0 (Tab.3)
window 11-15 23:56:59 -> 11-15 23:58:02 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/make_sf_lib.py` (mtime 11-17 04:33:04)

## review1113/sensorfree/E4/mav_s10 (Tab.3)
window 11-17 04:33:18 -> 11-17 08:30:44 (last "start" log line before the prediction file)
- no code file modified after the start
- 10 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/sensorfree/E4/mav_s20 (Tab.3)
window 11-17 11:38:33 -> 11-17 12:12:14 (last "start" log line before the prediction file)
- no code file modified after the start

## review1113/sensorfree/E4/antifs_s0 (Tab.3)
window 11-15 23:58:35 -> 11-16 01:42:52 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sensorfree/make_sf_lib.py` (mtime 11-17 04:33:04)
- 10 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/sensorfree/E4/antifs_s10 (Tab.3)
window 11-17 10:14:33 -> 11-17 11:38:04 (last "start" log line before the prediction file)
- no code file modified after the start

## review1113/sensorfree/E4/antifs_s20 (Tab.3)
window 11-17 11:38:33 -> 11-17 11:43:14 (last "start" log line before the prediction file)
- no code file modified after the start
- 10 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/heads/H3/mav_s0 (Tab.2 row 5)
window 11-15 18:49:09 -> 11-16 02:21:18 (last "start" log line before the prediction file)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/make_heads_lib.py` (mtime 11-17 04:38:54)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/run_arm.sh` (mtime 11-17 04:39:10)
- 13 file(s) (per-run lib copy / generated yaml) written by the run itself within 3 min of its start: not a change

## review1113/clean2/buildup (Tab.2 rows 2-3)
window 11-13 10:54:56 -> 11-13 10:54:32 (earliest log/run_info mtime in the run folder (upper bound of the start))
- no code file modified after the start

## review1113/ourdet_baseproto in2mav_sim_s0 (Tab.1 CenterNet rows)
window 11-16 16:42:06 -> 11-16 16:42:06 (prediction mtime (no start time found))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/build_ub_jpeg.py` (mtime 11-17 01:21:54)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/check_ub_zoom.py` (mtime 11-17 01:23:28)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/make_labels.py` (mtime 11-17 12:17:34)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/score_traindiag.py` (mtime 11-17 01:24:26)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/ub_zoom_dataset.py` (mtime 11-17 01:26:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ub_zoom_train.py` (mtime 11-17 01:23:04)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ubzoom_eval.sh` (mtime 11-17 01:23:40)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstg.yaml` (mtime 11-17 12:03:28)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstgs.yaml` (mtime 11-17 13:15:10)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_st.yaml` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_ub.yaml` (mtime 11-16 17:14:14)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_pstu.yaml` (mtime 11-17 09:52:36)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_ubzoom.yaml` (mtime 11-17 01:23:04)
- backups present: `make_labels.py.bak_1117` (11-17 09:52:18)

## review1113/ourdet_baseproto in2mav_st (Tab.1 CenterNet rows)
window 11-16 18:29:40 -> 11-16 18:29:40 (prediction mtime (no start time found))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/build_ub_jpeg.py` (mtime 11-17 01:21:54)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/check_ub_zoom.py` (mtime 11-17 01:23:28)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/make_labels.py` (mtime 11-17 12:17:34)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/score_traindiag.py` (mtime 11-17 01:24:26)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/ub_zoom_dataset.py` (mtime 11-17 01:26:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ub_zoom_train.py` (mtime 11-17 01:23:04)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ubzoom_eval.sh` (mtime 11-17 01:23:40)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstg.yaml` (mtime 11-17 12:03:28)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstgs.yaml` (mtime 11-17 13:15:10)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_pstu.yaml` (mtime 11-17 09:52:36)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_ubzoom.yaml` (mtime 11-17 01:23:04)
- backups present: `make_labels.py.bak_1117` (11-17 09:52:18)

## review1113/ourdet_baseproto in2mav_pst (Tab.1 CenterNet rows)
window 11-16 16:36:56 -> 11-16 16:36:56 (prediction mtime (no start time found))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/build_ub_jpeg.py` (mtime 11-17 01:21:54)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/check_ub_zoom.py` (mtime 11-17 01:23:28)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/make_labels.py` (mtime 11-17 12:17:34)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/score_traindiag.py` (mtime 11-17 01:24:26)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/ub_zoom_dataset.py` (mtime 11-17 01:26:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ub_zoom_train.py` (mtime 11-17 01:23:04)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ubzoom_eval.sh` (mtime 11-17 01:23:40)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstg.yaml` (mtime 11-17 12:03:28)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstgs.yaml` (mtime 11-17 13:15:10)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_st.yaml` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_ub.yaml` (mtime 11-16 17:14:14)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_pstu.yaml` (mtime 11-17 09:52:36)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_ubzoom.yaml` (mtime 11-17 01:23:04)
- backups present: `make_labels.py.bak_1117` (11-17 09:52:18)

## review1113/ourdet_baseproto in2mav_dann (Tab.1 CenterNet rows)
window 11-16 16:39:20 -> 11-16 16:39:20 (prediction mtime (no start time found))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/build_ub_jpeg.py` (mtime 11-17 01:21:54)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/check_ub_zoom.py` (mtime 11-17 01:23:28)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/make_labels.py` (mtime 11-17 12:17:34)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/score_traindiag.py` (mtime 11-17 01:24:26)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/ub_zoom_dataset.py` (mtime 11-17 01:26:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ub_zoom_train.py` (mtime 11-17 01:23:04)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ubzoom_eval.sh` (mtime 11-17 01:23:40)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstg.yaml` (mtime 11-17 12:03:28)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstgs.yaml` (mtime 11-17 13:15:10)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_st.yaml` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_ub.yaml` (mtime 11-16 17:14:14)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_pstu.yaml` (mtime 11-17 09:52:36)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_ubzoom.yaml` (mtime 11-17 01:23:04)
- backups present: `make_labels.py.bak_1117` (11-17 09:52:18)

## review1113/ourdet_baseproto in2mav_ub (Tab.1 CenterNet rows)
window 11-16 17:58:08 -> 11-16 17:58:08 (prediction mtime (no start time found))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/build_ub_jpeg.py` (mtime 11-17 01:21:54)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/check_ub_zoom.py` (mtime 11-17 01:23:28)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/make_labels.py` (mtime 11-17 12:17:34)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/score_traindiag.py` (mtime 11-17 01:24:26)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/ub_zoom_dataset.py` (mtime 11-17 01:26:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ub_zoom_train.py` (mtime 11-17 01:23:04)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ubzoom_eval.sh` (mtime 11-17 01:23:40)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstg.yaml` (mtime 11-17 12:03:28)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstgs.yaml` (mtime 11-17 13:15:10)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_pstu.yaml` (mtime 11-17 09:52:36)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_ubzoom.yaml` (mtime 11-17 01:23:04)
- backups present: `make_labels.py.bak_1117` (11-17 09:52:18)

## review1113/ourdet_baseproto out2antifs_sim_s0 (Tab.1 CenterNet rows)
window 11-16 17:08:54 -> 11-16 17:08:54 (prediction mtime (no start time found))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/build_ub_jpeg.py` (mtime 11-17 01:21:54)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/check_ub_zoom.py` (mtime 11-17 01:23:28)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/make_labels.py` (mtime 11-17 12:17:34)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/score_traindiag.py` (mtime 11-17 01:24:26)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/ub_zoom_dataset.py` (mtime 11-17 01:26:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ub_zoom_train.py` (mtime 11-17 01:23:04)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ubzoom_eval.sh` (mtime 11-17 01:23:40)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstg.yaml` (mtime 11-17 12:03:28)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstgs.yaml` (mtime 11-17 13:15:10)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_st.yaml` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_ub.yaml` (mtime 11-16 17:14:14)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_pstu.yaml` (mtime 11-17 09:52:36)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_ubzoom.yaml` (mtime 11-17 01:23:04)
- backups present: `make_labels.py.bak_1117` (11-17 09:52:18)

## review1113/ourdet_baseproto out2antifs_st (Tab.1 CenterNet rows)
window 11-16 17:43:42 -> 11-16 17:43:42 (prediction mtime (no start time found))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/build_ub_jpeg.py` (mtime 11-17 01:21:54)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/check_ub_zoom.py` (mtime 11-17 01:23:28)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/make_labels.py` (mtime 11-17 12:17:34)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/score_traindiag.py` (mtime 11-17 01:24:26)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/ub_zoom_dataset.py` (mtime 11-17 01:26:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ub_zoom_train.py` (mtime 11-17 01:23:04)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ubzoom_eval.sh` (mtime 11-17 01:23:40)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstg.yaml` (mtime 11-17 12:03:28)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstgs.yaml` (mtime 11-17 13:15:10)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_pstu.yaml` (mtime 11-17 09:52:36)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_ubzoom.yaml` (mtime 11-17 01:23:04)
- backups present: `make_labels.py.bak_1117` (11-17 09:52:18)

## review1113/ourdet_baseproto out2antifs_pst (Tab.1 CenterNet rows)
window 11-16 17:02:22 -> 11-16 17:02:22 (prediction mtime (no start time found))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/build_ub_jpeg.py` (mtime 11-17 01:21:54)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/check_ub_zoom.py` (mtime 11-17 01:23:28)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/make_labels.py` (mtime 11-17 12:17:34)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/score_traindiag.py` (mtime 11-17 01:24:26)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/ub_zoom_dataset.py` (mtime 11-17 01:26:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ub_zoom_train.py` (mtime 11-17 01:23:04)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ubzoom_eval.sh` (mtime 11-17 01:23:40)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstg.yaml` (mtime 11-17 12:03:28)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstgs.yaml` (mtime 11-17 13:15:10)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_st.yaml` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_ub.yaml` (mtime 11-16 17:14:14)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_pstu.yaml` (mtime 11-17 09:52:36)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_ubzoom.yaml` (mtime 11-17 01:23:04)
- backups present: `make_labels.py.bak_1117` (11-17 09:52:18)

## review1113/ourdet_baseproto out2antifs_dann (Tab.1 CenterNet rows)
window 11-16 17:04:50 -> 11-16 17:04:50 (prediction mtime (no start time found))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/common.sh` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/build_ub_jpeg.py` (mtime 11-17 01:21:54)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/check_ub_zoom.py` (mtime 11-17 01:23:28)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/make_labels.py` (mtime 11-17 12:17:34)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/score_traindiag.py` (mtime 11-17 01:24:26)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/ub_zoom_dataset.py` (mtime 11-17 01:26:16)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ub_zoom_train.py` (mtime 11-17 01:23:04)
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/ubzoom_eval.sh` (mtime 11-17 01:23:40)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstg.yaml` (mtime 11-17 12:03:28)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstgs.yaml` (mtime 11-17 13:15:10)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_st.yaml` (mtime 11-16 17:14:22)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_ub.yaml` (mtime 11-16 17:14:14)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_pstu.yaml` (mtime 11-17 09:52:36)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_ubzoom.yaml` (mtime 11-17 01:23:04)
- backups present: `make_labels.py.bak_1117` (11-17 09:52:18)

## review1113/ourdet_baseproto out2antifs_ubzoom (Tab.1 CenterNet rows)
window 11-17 01:36:02 -> 11-17 01:36:02 (prediction mtime (no start time found))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/ourdet_baseproto/lib/make_labels.py` (mtime 11-17 12:17:34)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstg.yaml` (mtime 11-17 12:03:28)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/in2mav_pstgs.yaml` (mtime 11-17 13:15:10)
- CHANGED after run: `E:/Open3DUAVDet/tools/cfgs/models/uavdet_3d/camnorm/ourdet_baseproto/out2antifs_pstu.yaml` (mtime 11-17 09:52:36)
- backups present: `make_labels.py.bak_1117` (11-17 09:52:18)

## D:/cmpfull_runs/monocd_s0 (Tab.1 baseline, local run)
window 10-31 04:44:01 -> 11-10 11:23:58 (earliest log/run_info mtime in the run folder (upper bound of the start))
- CHANGED during run: `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/monocd/predict.py` (mtime 10-31 11:50:10)

## D:/cmpfull_runs/monodgp_s0 (Tab.1 baseline, local run)
window 11-10 21:29:33 -> 11-10 21:29:33 (prediction mtime (no start time found))
- no code file modified after the start

## D:/cmpfull_runs/gupnet_s0 (Tab.1 baseline, local run)
window 10-30 06:20:49 -> 10-30 06:57:25 (earliest log/run_info mtime in the run folder (upper bound of the start))
- CHANGED after run: `E:/Open3DUAVDet/output/camnorm/geo1004/cmpfull/gupnet/predict.py` (mtime 11-10 11:27:46)

## D:/cmpfull_runs/gupnet_uda_dann_s0 (Tab.1 baseline, local run)
window 11-14 09:41:02 -> 11-14 13:59:34 (earliest log/run_info mtime in the run folder (upper bound of the start))
- no code file modified after the start

