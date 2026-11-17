# -*- coding: utf-8 -*-
"""[review1113 breakdown B] per-aircraft metrics of the reported runs (CPU only, existing predictions only, nothing re-selected).
Scoring = srcx/lib/score_x.score_meta (pack score.metrics, target rules) on a GT meta restricted to one aircraft's frames
(subset metas written to breakdown/meta/). Aircraft identity: MAV6D cmpeval/test_meta.npz 'cls' (= seq prefix mavic2/phantom4);
AntiUAV3D sealed antifs_test_meta.npz 'cls' (M4E / M400).
Distance bias: median log(z_pred / z_gt) with z = camera-frame depth (box centre z), over TP frames (pack rule: top-1 box, projected-2D IoU >= 0.1),
plus the same over every frame with a prediction, and the range-based median log(|t_pred| / |t_gt|) on TP frames (= c1a_bias.py definition)."""
import os, sys, json, pickle, importlib
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
OUT = os.path.dirname(os.path.abspath(__file__)); os.makedirs(OUT + '/meta', exist_ok=True)
X = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx'
SETS = {
    'mav': dict(src='in', meta='E:/Open3DUAVDet/output/camnorm/geo1004/cmpeval/test_meta.npz', split='real_test',
                arms={'PST (in2mav)': X + '/in2mav/preds/pcmd_fused_test.pkl', 'RGB-only teachers (in2mav_none)': X + '/in2mav_none/preds/pcmd_fused_test.pkl',
                      'Sim-only (srconly)': X + '/in2mav/preds/srconly_test.pkl'},
                air={'Mavic 2': 'mavic2', 'Phantom 4': 'phantom4'}),
    'antifs': dict(src='out', meta='D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz', split='antifs_test',
                   arms={'PST (out2antifs)': X + '/out2antifs/preds/pcmd_fused_test.pkl', 'RGB-only teachers (out2antifs_none)': X + '/out2antifs_none/preds/pcmd_fused_test.pkl',
                         'Sim-only (srconly)': X + '/out2antifs/preds/srconly_test.pkl'},
                   air={'Matrice 4E': 'M4E', 'Matrice 400': 'M400'}),
}


def score_module(src, tgt):
    os.environ.update(SX_SRC=src, SX_TGT=tgt, SX_TEACH='full')
    sys.path.insert(0, X + '/lib')
    for m in ('sx', 'score_x'):
        sys.modules.pop(m, None)
    return importlib.import_module('score_x')


def bias(Sp, meta, pred):
    M = np.load(meta); N = len(M['seq']); g = M['gt']; K = M['K']
    key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
    pt = np.full((N, 3), np.nan); pl = np.full((N, 3), np.nan); pe = np.full((N, 3), np.nan); cf = np.full(N, np.nan)
    for r in pickle.load(open(pred, 'rb')):   # = score_meta loading
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); pt[i], pl[i], pe[i], cf[i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    has = np.isfinite(pt).all(1) & np.isfinite(cf); tp = np.zeros(N, bool)
    for i in np.where(has)[0]:
        eul = pe[i] if np.isfinite(pe[i]).all() else np.zeros(3)
        tp[i] = Sp.iou(Sp.box2d(pt[i], pl[i], eul, K[i]), Sp.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) >= 0.1
    lz = np.log(pt[:, 2] / g[:, 2]); lr = np.log(np.linalg.norm(pt, axis=1) / np.linalg.norm(g[:, :3], axis=1))
    f = lambda a, m: float(np.median(a[m])) if m.any() else float('nan')
    return dict(n=N, n_pred=int(has.sum()), n_tp=int(tp.sum()), bias_logz_tp=f(lz, tp), bias_logz_pred=f(lz, has), bias_logrange_tp=f(lr, tp),
                frac_under_tp=float(np.mean(lz[tp] < 0)) if tp.any() else float('nan'))


res = {}
for tgt, cfg in SETS.items():
    SX = score_module(cfg['src'], tgt); Sp = SX.S
    M = np.load(cfg['meta']); cls = np.array([str(x) for x in M['cls']])
    subs = {'all': cfg['meta']}
    for name, code in cfg['air'].items():
        sel = cls == code; p = '%s/meta/%s_test_%s.npz' % (OUT, tgt, code)
        np.savez(p, **{k: M[k][sel] for k in M.files}); subs[name] = p
    res[tgt] = {}
    for arm, pred in cfg['arms'].items():
        stored = json.load(open(pred + '.score_%s.json' % cfg['split']))['all']
        res[tgt][arm] = {}
        for sub, meta in subs.items():
            m = SX.score_meta(pred, meta)['all']; m.update(bias(Sp, meta, pred))
            if sub == 'all':
                assert abs(m['UDS'] - stored['UDS']) < 1e-9 and abs(m['AP'] - stored['AP']) < 1e-9, (tgt, arm, m, stored)
            res[tgt][arm][sub] = m
            print('%-7s %-38s %-12s n %5d  AP %6.2f ATE %.3f ASE %6.2f AOE %6.2f mDR %6.2f UDS %6.2f  bias(logz,TP) %+.3f (pred %+.3f, range %+.3f)' % (
                tgt, arm, sub, m['n'], m['AP'], m['ATE'], m['ASE'], m['AOE'], m['mDR'], m['UDS'], m['bias_logz_tp'], m['bias_logz_pred'], m['bias_logrange_tp']), flush=True)
json.dump(res, open(OUT + '/per_aircraft.json', 'w'), indent=1, default=float)
print('wrote', OUT + '/per_aircraft.json')
