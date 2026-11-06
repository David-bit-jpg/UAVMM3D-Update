# -*- coding: utf-8 -*-
"""KZ-IRLiD 训练后（PREREG_irlid B：试点 IR+LiDAR 胜出，补齐 5 种子后走全链）：
    export      第 30 轮权重去掉 lid_head.* → lidar/kzlid/release/kzl_s{k}.pth（与 release_20260923/kz_s{k}.pth 同格式）
    adabn       同一批真实 train 裁窗重估 BN（kzst1/adabn_kz.py 原样；成员换成 KZ-LiD，回放核对跳过——它核对的是原 KZ 的已记录输出）
                → night1003/scratch/kzst1/adabn_kzl/kz_s{k}_adabn_kzl.pth
    dump <split>  KZ-IRLiD + AdaBN 成员在 train/val/test 上推理（位置 ST2P，同部署命令）→ repl/st2p_<split>_kzma"""
import os, sys
import torch
SCR = '@@ROOT@@/output/camnorm/night1003/scratch'
HERE = os.path.dirname(os.path.abspath(__file__))
CK = '@@ROOT@@/output/models/uavdet_3d/camnorm/sim_full_r34_kp_w12_zoom_irlid/KZMs%d/ckpt/checkpoint_epoch_30.pth'
REL = '@@ROOT@@/output/camnorm/geo1004/lidar/irlid/release/kzm_s%d.pth'
ADA = SCR + '/kzst1/adabn_kzm/kz_s%d_adabn_kzm.pth'


def export():
    os.makedirs(HERE + '/release', exist_ok=True)
    for k in range(5):
        a = torch.load(CK % k, map_location='cpu', weights_only=False)
        sd = {kk: v for kk, v in a['model_state'].items() if not (kk.startswith('lid_head.') or kk.startswith('aux_head.'))}
        n_drop = len(a['model_state']) - len(sd)
        torch.save({'model_state': sd, 'epoch': a.get('epoch'), 'src': CK % k + ' (lid_head.* removed: %d tensors)' % n_drop}, REL % k)
        print('kzm_s%d：%d 个张量（去掉 LiDAR 头 %d 个）' % (k, len(sd), n_drop), flush=True)


def adabn():
    for p in ('@@ROOT@@', '@@ROOT@@/tools', SCR + '/repl', SCR + '/kzst1'):     # kzst1 最后插入 = 最先找到（它的 kzst_common 指向 kzst1 的预登记）
        sys.path.insert(0, p)
    import adabn_kz as AB
    AB.C.RELEASE_KZ = [REL % k for k in range(5)]
    AB.verify_replay = lambda *a, **kw: {'pass': True, 'skipped': 'KZ-LiD 权重与记录的 KZ 成员输出不同，回放核对不适用'}
    return AB.main(['--out', SCR + '/kzst1/adabn_kzm', '--tag', 'adabn_kzm', '--force'])


def dump(split):
    sys.path.insert(0, SCR + '/repl')
    import dump_members_repl as M
    M.POS_SETS['st2p'] = ['@@ROOT@@/output/models/uavdet_3d/camnorm/st2plus_mav6d_r34_geo_full_tr/ST2P_s%d/ckpt/checkpoint_epoch_3.pth' % i for i in range(5)]
    ck = [ADA % k for k in range(5)]
    for c in ck:
        assert os.path.exists(c), c
    return M.main(['--split', split, '--pos', 'st2p', '--ckpts'] + ck + ['--tag', 'st2p_%s_kzma' % split])


if __name__ == '__main__':
    {'export': lambda: export(), 'adabn': lambda: adabn(), 'dump': lambda: dump(sys.argv[2])}[sys.argv[1]]()
