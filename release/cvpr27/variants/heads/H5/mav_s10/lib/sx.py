# [review1113 copy of srcx/lib/sx.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s10/lib] [heads H5 mav s10, mode ORI] O -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s10, W -> D:/S2MUAV3D/srcx_work/heads_H5_in2mav_s10
# -*- coding: utf-8 -*-
"""[srcx 11-11] source-domain ablation: one place for every source- / target-specific constant.
Selected by env SX_SRC (in | out | mix) and SX_TGT (anti | antifs | mav). Nothing here is a method choice: paths, the stage-1 checkpoints of the
source, the source's target-standardisation stats (decoder must match training), image sizes, crop-zoom caps and input normalisation.
Method constants (LAM .75, conf .5, gate 10 deg, mirror 12, alpha .5, ...) stay inside the copied scripts, unchanged."""
import os, json
SRC = os.environ.get('SX_SRC', 'out'); TGT = os.environ.get('SX_TGT', 'anti')
TEACH = os.environ.get('SX_TEACH', 'full')   # 11-12 no-teacher control: full | ir | lid | none (only the privileged knowledge changes)
assert TEACH in ('full', 'ir', 'lid', 'none')
NOIR = TEACH in ('lid', 'none')            # refiner without IR silhouette supervision / composites; one colour class
COMBO = '%s2%s' % (SRC, TGT) + ('' if TEACH == 'full' else '_' + TEACH)
HOME = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx'
O = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H5/mav_s10'          # [heads] H5 mav shift 10 (ORI)
W = 'D:/S2MUAV3D/srcx_work/heads_H5_in2mav_s10'
O_REF = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/sup3/run'; W_REF = 'D:/S2MUAV3D/srcx_work/in2mav_seed2'   # reference run, read only
ROOT = 'E:/Open3DUAVDet'; G = ROOT + '/output/camnorm/geo1004'; M = ROOT + '/output/models/uavdet_3d/camnorm'
CFGD = 'cfgs/models/uavdet_3d/camnorm/'
OLDSCR = 'C:/Users/jiang/AppData/Local/Temp/claude/0fa2f3b7e932577a/26739ba4-3497-41f0-8f86-fdad1ac856ee/scratchpad'

SOURCES = {
    # indoor UE5 sim = the source of the official method; stage-1 networks = the indoor-final ones (gproto/wf21/FINAL.md table 2, wf24 §7)
    'in': dict(
        sim='E:/mmcache/full_v2', sim_wh=(1280, 720), sim_zmax=4.0,                       # orientation-crop zoom cap on sim frames = original w_sim.py [1, 4]
        sim_norm=([0.3628, 0.4037, 0.4394], [0.2662, 0.2109, 0.2286]),
        det_cfg=CFGD + 'sim_full_r34_geo_tr_mix_sym8_mblur_ramp.yaml',
        kp_cfg=CFGD + 'sim_full_r34_kp_w12_zoom.yaml', kps_cfg=CFGD + 'sim_full_r34_kp_w12_zoom_irlid_kps.yaml',
        sim_crops=OLDSCR + '/kz/crops_6000',                                               # = original ou_train.py source crops
        depth=(5.9527, 3.7845), size=([0.6609, 0.6330, 0.2555], [0.5517, 0.5008, 0.1951]),
        det=[ROOT + '/output/release_20260920/ramp_s%d.pth' % k for k in (0, 1, 2)],
        irlid=[G + '/lidar/irlid/release/kzm_s%d.pth' % k for k in (0, 1)],
        kdir=[M + '/sim_full_r34_kp_w12_zoom_kdir/KZKDir_s%d/ckpt/checkpoint_epoch_30.pth' % k for k in (0, 1)],
        lid=[G + '/lidar/irlid/release/kzl_s%d.pth' % k for k in (0, 1)],
        kps=M + '/sim_full_r34_kp_w12_zoom_irlid_kps/KPSs0/ckpt/checkpoint_epoch_8.pth'),
    # CARLA outdoor sim (gproto/outdoor stage-1 networks)
    'out': dict(
        sim='D:/S2MUAV3D/sim_outdoor_carla/cache', sim_wh=(1280, 720), sim_zmax=12.5,
        sim_norm=([0.5174, 0.4442, 0.3115], [0.2616, 0.1835, 0.2408]),
        det_cfg=CFGD + 'sim_out_r34_geo_ramp_out.yaml',
        kp_cfg=CFGD + 'sim_out_r34_kp_w12_zoom_out.yaml', kps_cfg=CFGD + 'sim_out_r34_kp_w12_zoom_irlid_kps_out.yaml',
        sim_crops='D:/S2MUAV3D/pcmd_out_work/crops_5367',
        depth=(28.3561, 20.9749), size=([0.8036, 0.7784, 0.3668], [0.4473, 0.4304, 0.2376]),
        det=[M + '/sim_out_r34_geo_ramp_out/DET_s%d/ckpt/checkpoint_epoch_5.pth' % k for k in (0, 1, 2)],
        irlid=[M + '/sim_out_r34_kp_w12_zoom_irlid_out/OKZirlid_s%d/ckpt/checkpoint_epoch_30.pth' % k for k in (0, 1)],
        kdir=[M + '/sim_out_r34_kp_w12_zoom_kdir_out/OKZkdir_s%d/ckpt/checkpoint_epoch_30.pth' % k for k in (0, 1)],
        lid=[M + '/sim_out_r34_kp_w12_zoom_lid_out/OKZlid_s%d/ckpt/checkpoint_epoch_30.pth' % k for k in (0, 1)],
        kps=M + '/sim_out_r34_kp_w12_zoom_irlid_kps_out/OKZkps_s0/ckpt/checkpoint_epoch_8.pth'),
}
# mixed curated sim: filled from srcx/mix_source.json (written from MIX_STAGE1.md once the other agent's stage 1 is done)
_mj = HOME + '/mix_source.json'
if os.path.exists(_mj):
    SOURCES['mix'] = json.load(open(_mj, encoding='utf-8'))

TARGETS = {
    'anti': dict(real='D:/mmcache/outdoor_real_out', wh=(1920, 1080), zmax=18.75,
                 rn=([0.5068, 0.5156, 0.5173], [0.1559, 0.1533, 0.1683]),          # unlabeled real train frames (= detector real config NORM)
                 det_base=CFGD + 'outreal_r34_geo_full_tr_out.yaml', agree_rel=0.5 / 3.5, score_split='outdoor_test',
                 val_labels='D:/S2MUAV3D/real_outdoor_antiuav3d/labels/val.pkl', gt_meta=None),
    # 11-12 AntiUAV3D outdoor subset v2: intrinsics verified against images (kverify.py), sequence-disjoint focal/distance-balanced split.
    # Same camera/image geometry as 'anti' (1920x1080 raw, 512x288 input) -> zmax / agree_rel unchanged.
    'antifs': dict(real='D:/mmcache/outdoor_real_fs', wh=(1920, 1080), zmax=18.75,
                   rn=([0.4968, 0.5059, 0.5084], [0.1854, 0.1816, 0.1970]),
                   det_base=CFGD + 'outreal_fs_r34_geo_full_tr.yaml', agree_rel=0.5 / 3.5, score_split='antifs_test',
                   val_labels='D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/val.pkl',
                   gt_meta='D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz'),
    # 11-13 MAV6D (indoor real) = the target of the original indoor final chain (gproto/nolabel, wf24 README §6-7); every value mirrors it:
    #   cache E:/mmcache/mav6d_cn_obj (official object box; undistorted to K_pin, 1920x1080 raw -> 512x288), crop zoom cap 4 (indoor [1, 4]),
    #   input norm = cache norm.txt = camnorm_mav6d.yaml NORM (600 unlabeled train frames), det_base = mav6d_r34_geo_full_tr.yaml (nolabel REAL),
    #   pseudo-label agreement 0.5 m at 3.5 m, scoring = cmpeval/paper_table metrics (== pack score.py 'real_test': abs. thresholds
    #   0.1/0.2/0.4/0.8 m, a = 1 m) on cmpeval/test_meta.npz (byte-identical to pack data/gt/real_test_meta.npz); val sanity on cmpeval/val_meta.npz.
    #   No raw_path in MAV6D metas: raw frames are read from E:/MAV6D and undistorted to K_raw exactly as tools/kp_zoom.RawSource (imread_raw).
    'mav': dict(real='E:/mmcache/mav6d_cn_obj', wh=(1920, 1080), zmax=4.0,
                rn=([0.4360, 0.4418, 0.4374], [0.2239, 0.2244, 0.2200]),
                det_base=CFGD + 'mav6d_r34_geo_full_tr.yaml', agree_rel=0.5 / 3.5, score_split='real_test', rules='real_test',
                val_labels=G + '/cmpeval/val_meta.npz', gt_meta=G + '/cmpeval/test_meta.npz', raw_root='E:/MAV6D',
                wscale=1920.0 / 512.0),   # 11-13: indoor original applies the 6-170 px width gate in 512-px INPUT pixels (wf21/strict recs K fx=492); raw K here -> divide by 3.75
}

assert (SRC, TGT, TEACH) == ('in', 'mav', 'full'), (SRC, TGT, TEACH)   # [heads] this copy is for H5 mav s10 only
HEADS_ARM = 'H5'; ONECLASS = False   # [heads] one colour class (H1 / H1c)
S = SOURCES.get(SRC); T = TARGETS.get(TGT)
SIM = S['sim'] if S else None
SIM_CFG = S['det_cfg'] if S else None
REAL = T['real'] if T else None
TW, TH = T['wh'] if T else (1920, 1080)
REAL_CFG = CFGD + 'srcx/real_%s.yaml' % COMBO      # target data + source decoder stats (written by mkcfg.py)
ST_CFG = CFGD + 'srcx/st_%s.yaml' % COMBO          # Module 3 self-training config
PSEUDO_TAG = 'srcx_' + COMBO
_MAPS = {}


def imread_raw(m):
    """[11-13] raw frame of a cache meta in the K_raw / raw_wh geometry. AntiUAV3D metas carry raw_path (pinhole already) -> unchanged.
    MAV6D metas have no raw_path: read the distorted JPEG from T['raw_root'] and undistort to K_raw, identical to tools/kp_zoom.RawSource
    (the indoor final chain's zoom-crop source)."""
    import cv2, numpy as np
    if 'raw_path' in m:
        return cv2.imread(m['raw_path'], cv2.IMREAD_COLOR)
    import sys as _s
    _s.path.insert(0, ROOT + '/tools'); from build_mav6d_cache import K_CALIB, D_CALIB   # noqa
    cls, scene, sq = m['seq'].split('/')
    p = os.path.join(T['raw_root'], cls, 'JPEGImages', scene, sq, m['frame'])
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    assert img is not None, p
    K_raw = np.asarray(m['K_raw'], np.float64); key = tuple(np.round(K_raw.reshape(-1), 6))
    if key not in _MAPS:
        _MAPS[key] = cv2.initUndistortRectifyMap(K_CALIB, D_CALIB, None, K_raw, tuple(int(v) for v in m['raw_wh']), cv2.CV_32FC1)
    m1, m2 = _MAPS[key]
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


for d in ((O, W) if os.environ.get('SX_NOMKDIR') != '1' else ()):
    os.makedirs(d, exist_ok=True)
