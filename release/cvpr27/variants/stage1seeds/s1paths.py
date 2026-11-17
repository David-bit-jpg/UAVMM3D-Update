# -*- coding: utf-8 -*-
"""[stage1seeds] single source of truth for the RE-SEEDED stage-1 (simulation-only) networks of a source.
Seed sets: SD = 30 -> seeds 30,31,32 ; SD = 40 -> seeds 40,41,42 (nominal member k of the reported chain -> seed SD + k).
Recipes = the exact flags of the reported networks' train logs (args block of output/models/.../<run>/train_*.log; trainer script from the
original launchers: train.py | lidar/kzlid/train_mm.py | lidar/kzlid/train_kz_lid.py | lidar/kpsize/train_mmk.py). Only --seed and --extra_tag
(prefix S1X_, never used before) change; kdir's IR teacher and kps' init point at the SAME-SEED-SET re-trained networks.
CLI (used by train_stage1.sh):  python s1paths.py sh <src in|out> <sd 30|40> <item>   -> shell assignments
                                python s1paths.py items <src>                          -> item list in training order
                                python s1paths.py nets <src> <sd>                      -> json of the sx.SOURCES override (what downstream uses)"""
import os, sys, json
ROOT = 'E:/Open3DUAVDet'; M = ROOT + '/output/models/uavdet_3d/camnorm'; CF = 'cfgs/models/uavdet_3d/camnorm/'
G = ROOT + '/output/camnorm/geo1004'
# new run dirs live on D: (E: < 60 GB free): s1_train.py prefixes EXP_GROUP_PATH with EXPROOT -> <EXPROOT>/models/uavdet_3d/camnorm/<stem>/<tag>
EXPROOT = 'D:/S2MUAV3D/stage1seeds_models'; MS1 = EXPROOT + '/models/uavdet_3d/camnorm'
HERE = G + '/gproto/review1113/stage1seeds'
TR = {'train': ROOT + '/tools/train.py', 'mm': G + '/lidar/kzlid/train_mm.py', 'lid': G + '/lidar/kzlid/train_kz_lid.py', 'mmk': G + '/lidar/kpsize/train_mmk.py'}
# indoor common flags = RAMP / KZs / KZTIRs0 / KZKDir_s / KZMs / KZLs / KPSs logs (identical apart from epochs / max_ckpt_save_num / --set / init)
CIN = '--batch_size 8 --workers {W} --fix_random_seed --cudnn_benchmark --use_amp --logger_iter_interval 500 --val_split test --val_interval 10 --val_every 100 --val_match greedy --skip_test_eval'
# outdoor KP flags = gproto/outdoor/train_kp_out.sh COMMON (= OKZ* logs) ; outdoor detector = srcx/stage1_det.sh (no amp / cudnn_benchmark, val_match top1, logger 50 default)
COUT = CIN
CDOUT = '--batch_size 8 --workers {W} --fix_random_seed --val_split test --val_interval 10 --val_every 100 --skip_test_eval --max_ckpt_save_num 2'
SPEC = {
    'in': {
        'det':   dict(cfg=CF + 'sim_full_r34_geo_tr_mix_sym8_mblur_ramp.yaml', tr='train', ep=5, fl=CIN + ' --max_ckpt_save_num 10', tag='S1X_DET', n=3),
        'rgb':   dict(cfg=CF + 'sim_full_r34_kp_w12_zoom.yaml', tr='train', ep=30, fl=CIN + ' --max_ckpt_save_num 3', tag='S1X_KZ', n=2),
        'irT':   dict(cfg=CF + 'sim_full_r34_kp_w12_zoom_irT.yaml', tr='train', ep=30, fl=CIN + ' --max_ckpt_save_num 2', tag='S1X_KZTIR', n=1),
        'kdir':  dict(cfg=CF + 'sim_full_r34_kp_w12_zoom_kdir.yaml', tr='train', ep=30, fl=CIN + ' --max_ckpt_save_num 2', tag='S1X_KZKDir', n=2, teacher='irT'),
        'irlid': dict(cfg=CF + 'sim_full_r34_kp_w12_zoom_irlid.yaml', tr='mm', ep=30, fl=CIN + ' --max_ckpt_save_num 3', tag='S1X_KZM', n=2, strip=True),
        'lid':   dict(cfg=CF + 'sim_full_r34_kp_w12_zoom_lid.yaml', tr='lid', ep=30, fl=CIN + ' --max_ckpt_save_num 3', tag='S1X_KZL', n=2, strip=True),
        'kps':   dict(cfg=CF + 'sim_full_r34_kp_w12_zoom_irlid_kps.yaml', tr='mmk', ep=8, fl=CIN + ' --max_ckpt_save_num 2', tag='S1X_KPS', n=1, init='irlid'),
    },
    'out': {
        'det':   dict(cfg=CF + 'sim_out_r34_geo_ramp_out.yaml', tr='train', ep=5, fl=CDOUT, tag='S1X_DET', n=3),
        'rgb':   dict(cfg=CF + 'sim_out_r34_kp_w12_zoom_out.yaml', tr='train', ep=30, fl=COUT + ' --max_ckpt_save_num 2', tag='S1X_OKZrgb', n=2),
        'irT':   dict(cfg=CF + 'sim_out_r34_kp_w12_zoom_irT_out.yaml', tr='train', ep=30, fl=COUT + ' --max_ckpt_save_num 2', tag='S1X_OKZirT', n=1),
        'kdir':  dict(cfg=CF + 'sim_out_r34_kp_w12_zoom_kdir_out.yaml', tr='train', ep=30, fl=COUT + ' --max_ckpt_save_num 2', tag='S1X_OKZkdir', n=2, teacher='irT'),
        'irlid': dict(cfg=CF + 'sim_out_r34_kp_w12_zoom_irlid_out.yaml', tr='mm', ep=30, fl=COUT + ' --max_ckpt_save_num 2', tag='S1X_OKZirlid', n=2),
        'lid':   dict(cfg=CF + 'sim_out_r34_kp_w12_zoom_lid_out.yaml', tr='lid', ep=30, fl=COUT + ' --max_ckpt_save_num 2', tag='S1X_OKZlid', n=2),
        'kps':   dict(cfg=CF + 'sim_out_r34_kp_w12_zoom_irlid_kps_out.yaml', tr='mmk', ep=8, fl=COUT + ' --max_ckpt_save_num 2', tag='S1X_OKZkps', n=1, init='irlid'),
    }}
# the reported networks whose train logs define the recipes (for check_cfg_drift.py and the README)
REFERENCE_RUNS = {
    'in': {'det': M + '/sim_full_r34_geo_tr_mix_sym8_mblur_ramp/RAMP', 'rgb': M + '/sim_full_r34_kp_w12_zoom/KZs0',
           'irT': M + '/sim_full_r34_kp_w12_zoom_irT/KZTIRs0', 'kdir': M + '/sim_full_r34_kp_w12_zoom_kdir/KZKDir_s0',
           'irlid': M + '/sim_full_r34_kp_w12_zoom_irlid/KZMs0', 'lid': M + '/sim_full_r34_kp_w12_zoom_lid/KZLs0',
           'kps': M + '/sim_full_r34_kp_w12_zoom_irlid_kps/KPSs0'},
    'out': {'det': M + '/sim_out_r34_geo_ramp_out/DET_s0', 'rgb': M + '/sim_out_r34_kp_w12_zoom_out/OKZrgb_s0',
            'irT': M + '/sim_out_r34_kp_w12_zoom_irT_out/OKZirT_s0', 'kdir': M + '/sim_out_r34_kp_w12_zoom_kdir_out/OKZkdir_s0',
            'irlid': M + '/sim_out_r34_kp_w12_zoom_irlid_out/OKZirlid_s0', 'lid': M + '/sim_out_r34_kp_w12_zoom_lid_out/OKZlid_s0',
            'kps': M + '/sim_out_r34_kp_w12_zoom_irlid_kps_out/OKZkps_s0'}}
ORDER = ['det:0', 'det:1', 'det:2', 'rgb:0', 'rgb:1', 'irT:0', 'kdir:0', 'kdir:1', 'irlid:0', 'irlid:1', 'kps:0', 'lid:0', 'lid:1']


def stem(cfg):
    return os.path.basename(cfg)[:-5]


def ck(src, sd, typ, k=0):
    s = SPEC[src][typ]
    return '%s/%s/%s_s%d/ckpt/checkpoint_epoch_%d.pth' % (MS1, stem(s['cfg']), s['tag'], sd + k, s['ep'])


def rel(src, sd, typ, k=0):
    """checkpoint the CHAIN uses. indoor det: model_state-only copy (= release_20260920/ramp_s*.pth format, weights identical; verified for
    s0-s2); indoor irlid/lid: aux heads stripped (= lidar/irlid/release/kzm_s*/kzl_s*.pth via lidar/irlid/strip_aux.py, verified
    tensor-identical on the kept keys). Outdoor: the raw epoch checkpoint (as srcx/lib/sx.py SOURCES['out'])."""
    if src == 'in' and typ == 'det':
        return '%s/release/in_sd%d/ramp_s%d.pth' % (HERE, sd, sd + k)
    if src == 'in' and typ in ('irlid', 'lid'):
        return '%s/release/in_sd%d/%s_s%d.pth' % (HERE, sd, {'irlid': 'kzm', 'lid': 'kzl'}[typ], sd + k)
    return ck(src, sd, typ, k)


def nets(src, sd):
    """sx.SOURCES[src] override for a downstream run of seed set sd (+ the plain-RGB pair used by SX_TEACH=none)"""
    return dict(det=[rel(src, sd, 'det', k) for k in range(3)], irlid=[rel(src, sd, 'irlid', k) for k in range(2)],
                kdir=[rel(src, sd, 'kdir', k) for k in range(2)], lid=[rel(src, sd, 'lid', k) for k in range(2)],
                kps=rel(src, sd, 'kps'), rgb=[rel(src, sd, 'rgb', k) for k in range(2)])


def item_sh(src, sd, item, workers=2):
    typ, k = item.split(':'); k = int(k); s = SPEC[src][typ]
    assert k < s['n'], item
    seed = sd + k; tag = '%s_s%d' % (s['tag'], seed)
    fl = s['fl'].format(W=workers) + ' --epochs %d --seed %d --extra_tag %s' % (s['ep'], seed, tag)
    dep = ''; depitem = ''
    if 'teacher' in s:   # kdir: IR teacher = this seed set's irT (reported: KZTIRs0 / OKZirT_s0 for BOTH kdir seeds)
        dep = ck(src, sd, s['teacher'], 0); depitem = s['teacher'] + '0'; fl += ' --set MODEL.DISTILL.TEACHER_CKPT %s' % dep
    if 'init' in s:      # kps: init = same-seed IR+LiDAR net, raw epoch-30 checkpoint (reported: KZMs0 / OKZirlid_s0 epoch 30)
        dep = ck(src, sd, s['init'], k); depitem = s['init'] + str(k); fl += ' --pretrained_model %s' % dep
    out = dict(TYP=typ, K=k, SEED=seed, TAG=tag, CFG=s['cfg'], TRAINER=TR[s['tr']], EPOCHS=s['ep'], FLAGS=fl, CKPT=ck(src, sd, typ, k),
               CKDIR=os.path.dirname(ck(src, sd, typ, k)), RUNDIR=os.path.dirname(os.path.dirname(ck(src, sd, typ, k))),
               REL=rel(src, sd, typ, k), STRIP=int(bool(s.get('strip'))), RELDET=int(src == 'in' and typ == 'det'), DEP=dep, DEPITEM=depitem,
               REFRUN=REFERENCE_RUNS[src][typ])
    return out


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'sh':
        src, sd, item = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        w = int(os.environ.get('S1_WORKERS', '2'))
        for kk, v in item_sh(src, sd, item, w).items():
            print("%s='%s'" % (kk, v))
    elif cmd == 'items':
        print(' '.join(ORDER))
    elif cmd == 'nets':
        print(json.dumps(nets(sys.argv[2], int(sys.argv[3])), indent=1))
