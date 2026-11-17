# Advisor server runs behind Tab. 1, and the order of the pack patches

The five baselines (GUPNet, MonoDETR, MonoCD, MonoDGP, MonoCoP) were trained by the advisor on his server from `UAV3D_baselines_pack`
(`advisor_pack/00_UAV3D_baselines_pack`, shipped 11-10 as `D:/UAV3D_baselines_pack.zip`), with the patches below applied in this order.
The predictions came back as `results_to_send_*.zip`; the copy used for the paper is `D:/advisor_results_1117b/runs/` (11-17 b delivery);
most files are byte-identical to the 11-16 delivery. The run folder name is `<method>_<source>_<step>_<target>_s<seed>`:
source `in` = indoor SimDrone3D, `out` = outdoor SimDrone3D; step `sim` = simulation only, `st` = +ST (RGB self-training),
`kdc` = +PST on MAV6D (table `pcmd_teacher_pseudo_real_train_clean.pkl`, 5,227 frames, patch 1114), `kd` = +PST on RealDrone3D
(table `pcmd_teacher_pseudo_outdoor_train.pkl`, 249 frames, patch 1112), `dann` = +DANN (patch 1116). Scores: `D:/advisor_results_1117b/runs/SCORES.md`
(advisor-side aggregation) and `repro/rescore_report.md` (our rescore with the same scorer; all 192 baseline cells reproduce).

## Patch order (shipped zip, date, what it changed)

| # | folder | shipped | what it does |
|---|---|---|---|
| 0 | `UAV3D_baselines_pack` | 11-10 11:59 (4.7 GB zip) | base pack: data (cmpfull exports), five ported baselines under `Open3DUAVDet/output/camnorm/geo1004/cmpfull/`, `scripts/{run.sh,run_all.sh,score.py,tasks.txt}` |
| 1 | `UAV3D_pack_update_1112` | 11-12 07:44 (1.36 GB) | RealDrone3D v2 (intrinsics verified against images, new sequence-disjoint split), new outdoor GT meta, optional focal-unified runs (`simfu/stfu/kdfu`, supplement only), `scripts/score.py`, PST outdoor table (249 frames) |
| 2 | `UAV3D_pack_hotfix_1112b` (shipped as `hotfix_1112c.zip`) | 11-12 09:16 / 10:04 | path placeholders in the code, two missing MonoCD configs (`monocd/runs/monocd*.yaml`), `scripts/run_all.sh` |
| 3 | `UAV3D_pack_prio_1113` | 11-13 10:29 | reorders `scripts/tasks.txt` only |
| 4 | `UAV3D_pack_fix_1113` (+ `fix_1113b.zip`) | 11-13 11:04 / 11:13 | MonoCD single-process data loading (pin-memory crash), focal-unified predictor dataset names, restart |
| 5 | `UAV3D_pack_fix_1114` (+ `fix_1114b`) | 11-14 08:33 / 10:01 | MAV6D +PST table replaced by the clean-chain table (5,227 frames) -> new step `kdc`; 1114b queues MonoCD/GUPNet seeds on idle GPUs |
| 6 | `UAV3D_pack_fix_1115` | 11-15 10:17 | queues extra seeds (`kdc`, `kd`, outdoor `sim`/`st`) |
| 7 | `UAV3D_pack_fix_1115b` | 11-15 10:43 | sensor-free label tables (`kdn` RGB-only teachers, `kde` sensor-free E1), supplement only |
| 8 | `UAV3D_pack_fix_1116` | 11-16 15:41 | GUPNet +DANN on RealDrone3D (`uda_train.py`) -> `gupnet_out_dann_anti_s0` |
| 9 | `UAV3D_pack_fix_1117` | 11-17 10:05 | ungated RealDrone3D tables (`kdu`, 909 frames; `kdnu`, `kdeu`), 6 data workers for these new tasks; **not used in Tab. 1** |
| 10 | `UAV3D_pack_fix_1117b` | 11-17 12:03 | gated MAV6D table (`kdg`, 2,126 frames); **not used in Tab. 1** |
| 11 | `UAV3D_pack_fix_1117c` | 11-17 12:17 | `kdg` with the 90-degree representative of the teacher orientation; **not used in Tab. 1** |

Tab. 1 uses `kdc` on MAV6D and `kd` on RealDrone3D, as Sec. 4.1 states (all 5,227 frames on MAV6D, only the 249 agreement-checked frames on RealDrone3D).
The `kdu` / `kdg` runs of patches 9-11 were not part of the 11-17 b delivery and feed no number of the paper.

## Which run produced which Tab. 1 row

Means over the listed runs (per-run metrics averaged, UDS averaged per run). `[server]` = advisor run in `D:/advisor_results_1117b/runs/`;
`[local]` = run on our machine in `D:/cmpfull_runs/` with the same ported code (`baselines/cmpfull/<method>`).

| row | target | runs | run folders |
|---|---|---|---|
| MonoDETR | MAV6D | 1 | `monodetr_in_sim_s0` [server, also byte-identical in 11-16 delivery] |
| MonoDETR | RealDrone3D | 1 | `monodetr_out_sim_s0` [server, also byte-identical in 11-16 delivery] |
| MonoDETR +ST | MAV6D | 1 | `monodetr_in_st_mav_s0` [server, also byte-identical in 11-16 delivery] |
| MonoDETR +ST | RealDrone3D | 1 | `monodetr_out_st_anti_s0` [server, also byte-identical in 11-16 delivery] |
| MonoDETR +PST | MAV6D | 3 | `monodetr_in_kdc_mav_s0` [server, also byte-identical in 11-16 delivery]<br>`monodetr_in_kdc_mav_s1` [server]<br>`monodetr_in_kdc_mav_s2` [server] |
| MonoDETR +PST | RealDrone3D | 3 | `monodetr_out_kd_anti_s0` [server, also byte-identical in 11-16 delivery]<br>`monodetr_out_kd_anti_s1` [server, also byte-identical in 11-16 delivery]<br>`monodetr_out_kd_anti_s2` [server, also byte-identical in 11-16 delivery] |
| MonoCD | MAV6D | 3 | `monocd_in_sim_s1` [server, also byte-identical in 11-16 delivery]<br>`monocd_in_sim_s2` [server, also byte-identical in 11-16 delivery]<br>`monocd_s0` [local] |
| MonoCD | RealDrone3D | 3 | `monocd_out_sim_s0` [server, also byte-identical in 11-16 delivery]<br>`monocd_out_sim_s1` [server, also byte-identical in 11-16 delivery]<br>`monocd_out_sim_s2` [server] |
| MonoCD +ST | MAV6D | 1 | `monocd_in_st_mav_s1` [server, also byte-identical in 11-16 delivery] |
| MonoCD +ST | RealDrone3D | 3 | `monocd_out_st_anti_s0` [server, also byte-identical in 11-16 delivery]<br>`monocd_out_st_anti_s1` [server]<br>`monocd_out_st_anti_s2` [server] |
| MonoCD +PST | MAV6D | 3 | `monocd_in_kdc_mav_s0` [server, also byte-identical in 11-16 delivery]<br>`monocd_in_kdc_mav_s1` [server, also byte-identical in 11-16 delivery]<br>`monocd_in_kdc_mav_s2` [server, also byte-identical in 11-16 delivery] |
| MonoCD +PST | RealDrone3D | 3 | `monocd_out_kd_anti_s0` [server, also byte-identical in 11-16 delivery]<br>`monocd_out_kd_anti_s1` [server, also byte-identical in 11-16 delivery]<br>`monocd_out_kd_anti_s2` [server, also byte-identical in 11-16 delivery] |
| MonoDGP | MAV6D | 1 | `monodgp_s0` [local] |
| MonoDGP | RealDrone3D | 3 | `monodgp_out_sim_s0` [server, also byte-identical in 11-16 delivery]<br>`monodgp_out_sim_s1` [server]<br>`monodgp_out_sim_s2` [server] |
| MonoDGP +ST | MAV6D | 1 | `monodgp_in_st_mav_s0` [server, also byte-identical in 11-16 delivery] |
| MonoDGP +ST | RealDrone3D | 3 | `monodgp_out_st_anti_s0` [server, also byte-identical in 11-16 delivery]<br>`monodgp_out_st_anti_s1` [server]<br>`monodgp_out_st_anti_s2` [server] |
| MonoDGP +PST | MAV6D | 3 | `monodgp_in_kdc_mav_s0` [server, also byte-identical in 11-16 delivery]<br>`monodgp_in_kdc_mav_s1` [server, also byte-identical in 11-16 delivery]<br>`monodgp_in_kdc_mav_s2` [server, also byte-identical in 11-16 delivery] |
| MonoDGP +PST | RealDrone3D | 3 | `monodgp_out_kd_anti_s0` [server, also byte-identical in 11-16 delivery]<br>`monodgp_out_kd_anti_s1` [server, also byte-identical in 11-16 delivery]<br>`monodgp_out_kd_anti_s2` [server, also byte-identical in 11-16 delivery] |
| MonoCoP | MAV6D | 1 | `monocop_in_sim_s0` [server, also byte-identical in 11-16 delivery] |
| MonoCoP | RealDrone3D | 1 | `monocop_out_sim_s0` [server, also byte-identical in 11-16 delivery] |
| MonoCoP +ST | MAV6D | 1 | `monocop_in_st_mav_s0` [server, also byte-identical in 11-16 delivery] |
| MonoCoP +ST | RealDrone3D | 1 | `monocop_out_st_anti_s0` [server, also byte-identical in 11-16 delivery] |
| MonoCoP +PST | MAV6D | 1 | `monocop_in_kdc_mav_s0` [server, also byte-identical in 11-16 delivery] |
| MonoCoP +PST | RealDrone3D | 3 | `monocop_out_kd_anti_s0` [server, also byte-identical in 11-16 delivery]<br>`monocop_out_kd_anti_s1` [server, also byte-identical in 11-16 delivery]<br>`monocop_out_kd_anti_s2` [server, also byte-identical in 11-16 delivery] |
| GUPNet | MAV6D | 3 | `gupnet_in_sim_s1` [server, also byte-identical in 11-16 delivery]<br>`gupnet_in_sim_s2` [server, also byte-identical in 11-16 delivery]<br>`gupnet_s0` [local] |
| GUPNet | RealDrone3D | 3 | `gupnet_out_sim_s0` [server, also byte-identical in 11-16 delivery]<br>`gupnet_out_sim_s1` [server]<br>`gupnet_out_sim_s2` [server] |
| GUPNet +DANN | MAV6D | 1 | `gupnet_uda_dann_s0` [local] |
| GUPNet +DANN | RealDrone3D | 1 | `gupnet_out_dann_anti_s0` [server] |
| GUPNet +ST | MAV6D | 1 | `gupnet_in_st_mav_s0` [server, also byte-identical in 11-16 delivery] |
| GUPNet +ST | RealDrone3D | 3 | `gupnet_out_st_anti_s0` [server, also byte-identical in 11-16 delivery]<br>`gupnet_out_st_anti_s1` [server]<br>`gupnet_out_st_anti_s2` [server] |
| GUPNet +PST | MAV6D | 3 | `gupnet_in_kdc_mav_s0` [server, also byte-identical in 11-16 delivery]<br>`gupnet_in_kdc_mav_s1` [server, also byte-identical in 11-16 delivery]<br>`gupnet_in_kdc_mav_s2` [server, also byte-identical in 11-16 delivery] |
| GUPNet +PST | RealDrone3D | 3 | `gupnet_out_kd_anti_s0` [server, also byte-identical in 11-16 delivery]<br>`gupnet_out_kd_anti_s1` [server, also byte-identical in 11-16 delivery]<br>`gupnet_out_kd_anti_s2` [server, also byte-identical in 11-16 delivery] |
