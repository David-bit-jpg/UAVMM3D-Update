# -*- coding: utf-8 -*-
"""Replicate the PST pseudo-label tables (make_pseudo_pcmd_clean.py -> table; make_pseudo_pcmd_out.py logic on srcx v2)
and export the sensor-free variants. No real labels are read (labels.pkl only provides seq/frame keys)."""
import os, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot

MAV_LAB = 'D:/UAV3D_baselines_pack/data/cmpfull/real_train/labels.pkl'
OUT_LAB = 'D:/UAV3D_pack_update_1112/data/cmpfull/outdoor_train/labels.pkl'
RZ = [Rot.from_euler('z', 90 * q, degrees=True).as_matrix() for q in range(4)]


def load_pos(p):
    idx = pickle.load(open(p, 'rb'))
    return {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64).reshape(-1, 9)[0] for m in idx['metas'] if m}


def mav_table(tag, orient, gate_only):
    # == make_pseudo_pcmd_clean.py, then boxes[0] -> table fields (center, lwh, euler_xyz)
    pos = load_pos('E:/mmcache/mav6d_cn_obj/train/index_pseudo_%s.pkl' % tag)
    T = np.load(orient, allow_pickle=True)
    lab = {(l['seq'], l['frame']) for l in pickle.load(open(MAV_LAB, 'rb'))}
    out, dang = [], []
    for i in range(len(T['seq'])):
        k = (str(T['seq'][i]), str(T['frame'][i]))
        if k not in pos or k not in lab or (gate_only and not T['gate'][i]):
            continue
        b = pos[k]; Rm = np.asarray(T['R'][i], np.float64)
        dang.append(np.degrees(Rot.from_matrix(Rot.from_euler('xyz', b[6:9]).as_matrix().T @ Rm).magnitude()))
        out.append(dict(seq=k[0], frame=k[1], center=b[:3].copy(), lwh=b[3:6].copy(), euler_xyz=Rot.from_matrix(Rm).as_euler('xyz')))
    return out, np.array(dang)


def out_table(tag, orient):
    # == make_pseudo_pcmd_out.py logic with srcx v2 sources
    pos = load_pos('D:/mmcache/outdoor_real_fs/train/index_pseudo_%s.pkl' % tag)
    T = np.load(orient, allow_pickle=True)
    lab = {(l['seq'], l['frame']) for l in pickle.load(open(OUT_LAB, 'rb'))}
    out, dang = [], []
    for i in range(len(T['seq'])):
        k = (str(T['seq'][i]), str(T['frame'][i]))
        if k not in pos or k not in lab or not T['gate'][i]:
            continue
        b = pos[k]; Rd = Rot.from_euler('xyz', b[6:9]).as_matrix(); R0 = np.asarray(T['R'][i], np.float64)
        q = int(np.argmin([Rot.from_matrix(Rd.T @ R0 @ RZ[j]).magnitude() for j in range(4)]))
        Rm = R0 @ RZ[q]; lwh = b[3:6].copy()
        if q % 2:
            lwh[[0, 1]] = lwh[[1, 0]]
        dang.append(np.degrees(Rot.from_matrix(Rd.T @ Rm).magnitude()))
        out.append(dict(seq=k[0], frame=k[1], center=b[:3].copy(), lwh=lwh, euler_xyz=Rot.from_matrix(Rm).as_euler('xyz')))
    return out, np.array(dang)


def same(a, b):
    if len(a) != len(b): return 'len %d vs %d' % (len(a), len(b))
    da = {(r['seq'], r['frame']): r for r in a}; db = {(r['seq'], r['frame']): r for r in b}
    if set(da) != set(db): return 'keys differ (%d common)' % len(set(da) & set(db))
    e = max(max(np.abs(np.asarray(da[k][f]) - np.asarray(db[k][f])).max() for f in ('center', 'lwh', 'euler_xyz')) for k in da)
    return 'IDENTICAL keys, max abs diff %.2e' % e


if __name__ == '__main__':
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    pst_mav = pickle.load(open('D:/UAV3D_pack_fix_1114/data/pcmd_teacher_pseudo_real_train_clean.pkl', 'rb'))
    pst_out = pickle.load(open('D:/UAV3D_pack_update_1112/data/cmpfull/pcmd_teacher_pseudo_outdoor_train.pkl', 'rb'))
    print('PST MAV6D table %d, PST AntiUAV3D table %d' % (len(pst_mav), len(pst_out)))
    # also compare against the full-format intermediate
    full = pickle.load(open('D:/cmpfull_runs/pcmd_pseudo_real_train_clean_in2mav.pkl', 'rb'))
    full_t = [dict(seq=d['seq'], frame=d['frame'], center=d['boxes'][0]['center'], lwh=d['boxes'][0]['lwh'], euler_xyz=d['boxes'][0]['euler_xyz']) for d in full]
    print('intermediate full pkl -> table vs PST table:', same(full_t, pst_mav))
    for g in (False, True):
        t, _ = mav_table('srcx_in2mav', 'D:/S2MUAV3D/srcx_work/in2mav/pseudo_tr1.npz', g)
        print('replicate MAV6D PST gate_only=%s: %s' % (g, same(t, pst_mav)))
    t, _ = out_table('srcx_out2antifs', 'D:/S2MUAV3D/srcx_work/out2antifs/pseudo_tr1.npz')
    print('replicate AntiUAV3D PST: %s' % same(t, pst_out))
    GATE = '--gate-only' in sys.argv
    jobs = [('real_train', 'rgbonly', lambda: mav_table('srcx_in2mav_none', 'D:/S2MUAV3D/srcx_work/in2mav_none/pseudo_tr1.npz', GATE)),
            ('real_train', 'nosensor', lambda: mav_table('srcx_in2mav_none_sfE1s0', 'D:/S2MUAV3D/srcx_work/sf_E1_in2mav_none_s0/pseudo_tr1.npz', GATE)),
            ('outdoor_train', 'rgbonly', lambda: out_table('srcx_out2antifs_none', 'D:/S2MUAV3D/srcx_work/out2antifs_none/pseudo_tr1.npz')),
            ('outdoor_train', 'nosensor', lambda: out_table('srcx_out2antifs_none_sfE1s0', 'D:/S2MUAV3D/srcx_work/sf_E1_out2antifs_none_s0/pseudo_tr1.npz'))]
    for split, name, fn in jobs:
        t, dang = fn()
        lab = {(l['seq'], l['frame']) for l in pickle.load(open(MAV_LAB if split == 'real_train' else OUT_LAB, 'rb'))}
        assert all((r['seq'], r['frame']) in lab for r in t)
        assert all(set(r) == {'seq', 'frame', 'center', 'lwh', 'euler_xyz'} for r in t)
        ref = pst_mav if split == 'real_train' else pst_out
        ov = len({(r['seq'], r['frame']) for r in t} & {(r['seq'], r['frame']) for r in ref})
        p = os.path.join(outdir, 'pcmd_teacher_pseudo_%s_%s.pkl' % (split, name))
        pickle.dump(t, open(p, 'wb'))
        print('%s %-8s: %d frames (PST %d, overlap %d), det-vs-teacher angle median %.1f deg -> %s' % (split, name, len(t), len(ref), ov, np.median(dang), p))
