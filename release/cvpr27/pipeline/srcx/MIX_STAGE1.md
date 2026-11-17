# Mixed curated sim source — stage-1 checkpoints (11-13 00:49)

Source: indoor UE5 sim train (19,490) + CARLA outdoor sim train (5,367) + CARLA outdoor sim test/Town10 quality-filtered (1,564 of 1,818 kept) = 26,421 frames (25,032 after the inherited indoor MIN_RGB_VIS 5 filter).
Cache: D:/S2MUAV3D/sim_mix_cache/{train,test}; test = junction to E:/mmcache/full_v2/test = **mixed-source sim validation = indoor sim val split** (use for the fusion weight rule).
Kept list: srcx/mix/outdoor_test_kept.json. Details/decisions: srcx/mix/STATUS.md.

Stats (tools/calc_cache_norm.py, tools/calc_target_stats.py on the mixed train split): NORM mean [0.3912,0.4095,0.4113] std [0.2708,0.2067,0.2361];
detector DEPTH 14.2488±15.2051; KP log(Zv/L) 1.5278±0.2695; SIZE mean [0.6957,0.6691,0.2874] std [0.5128,0.4726,0.2116].
VIEW_AUG: det t_range [1.0,4.6]; KP t_range [1.15,1.75] max_upscale 5; DEPTH_MAX 80.

Configs (tools/cfgs/models/uavdet_3d/camnorm): sim_mix_r34_geo_ramp.yaml, sim_mix_r34_kp_w12_zoom{,_irT,_irlid,_kdir,_lid,_irlid_kps}.yaml;
dataset configs camnorm_sim_mix_geo_tr_mblur_sym2_mix.yaml, camnorm_sim_mix_kp_t_zoom{,_ir,_mm}_mix.yaml.
LiDAR templates: srcx/mix/templates_mix.npy (14 = indoor/outdoor × 7 airframes). KP crop cache: D:/S2MUAV3D/pcmd_mix_work/crops_12931.

| network | checkpoint | exists |
|---|---|---|
| det s0 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_geo_ramp/DET_s0/ckpt/checkpoint_epoch_5.pth | yes |
| det s1 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_geo_ramp/DET_s1/ckpt/checkpoint_epoch_5.pth | yes |
| det s2 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_geo_ramp/DET_s2/ckpt/checkpoint_epoch_5.pth | yes |
| irT s0 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_kp_w12_zoom_irT/MKZirT_s0/ckpt/checkpoint_epoch_30.pth | yes |
| irlid s0 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_kp_w12_zoom_irlid/MKZirlid_s0/ckpt/checkpoint_epoch_30.pth | yes |
| irlid s1 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_kp_w12_zoom_irlid/MKZirlid_s1/ckpt/checkpoint_epoch_30.pth | yes |
| kdir s0 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_kp_w12_zoom_kdir/MKZkdir_s0/ckpt/checkpoint_epoch_30.pth | yes |
| kdir s1 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_kp_w12_zoom_kdir/MKZkdir_s1/ckpt/checkpoint_epoch_30.pth | yes |
| lid s0 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_kp_w12_zoom_lid/MKZlid_s0/ckpt/checkpoint_epoch_30.pth | yes |
| lid s1 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_kp_w12_zoom_lid/MKZlid_s1/ckpt/checkpoint_epoch_30.pth | yes |
| kps s0 | E:/Open3DUAVDet/output/models/uavdet_3d/camnorm/sim_mix_r34_kp_w12_zoom_irlid_kps/MKZkps_s0/ckpt/checkpoint_epoch_8.pth | yes |
