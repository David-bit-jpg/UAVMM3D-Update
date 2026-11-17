# 11-14: MAV6D val per aircraft (Mavic 2 dark / Phantom 4 white), same scorer as per_aircraft.py; existing predictions only
import os, sys, json, numpy as np
os.environ['SX_SRC'] = 'in'; os.environ['SX_TGT'] = 'mav'
G = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'
sys.path.insert(0, G + '/srcx/lib'); import score_x as SX, sx
meta = sx.T['val_labels']; M = np.load(meta)
seq = np.array([str(s) for s in M['seq']]); cls = np.array([s.split('/')[0] for s in seq])
OUT = os.path.dirname(os.path.abspath(__file__))
subs = {'all': meta}
for c in ['mavic2', 'phantom4']:
    p = OUT + '/meta/mav_val_%s.npz' % c; sel = cls == c
    np.savez(p, **{k: (M[k][sel] if M[k].shape[:1] == (len(seq),) else M[k]) for k in M.files}); subs[c] = p
arms = {'PST': G + '/srcx/in2mav/preds/pcmd_fused_val.pkl', 'RGB-only': G + '/srcx/in2mav_none/preds/pcmd_fused_val.pkl',
        'Sim-only': G + '/srcx/in2mav/preds/srconly_val.pkl'}
res = {}
for a, pred in arms.items():
    for s, mp in subs.items():
        m = SX.score_meta(pred, mp)['all']; res['%s|%s' % (a, s)] = m
        print('%-9s %-9s n=%4d  AP %.2f ATE %.3f ASE %.2f AOE %.2f mDR %.2f UDS %.2f' % (a, s, len(np.load(mp)['seq']), m['AP'], m['ATE'], m['ASE'], m['AOE'], m['mDR'], m['UDS']))
json.dump(res, open(OUT + '/per_aircraft_val.json', 'w'), indent=1, default=float)
