# R3: deployment offsets b_k estimated from unlabeled TRAIN images instead of VAL images; re-score nl075 chain (MAV6D).
# Read-only copy of nolabel/fuse_eval_nl.py logic; writes only to this folder. CPU only.
import os, sys, json, pickle
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.update(NL_ARM='nl075', FE_UNC='z2', FE_ORI='ouS6I2M', FE_NMEM='4', FE_ZKP='1', FE_CONST='train', FE_COLOR='new', FE_KPS='strict', FE_HT='color')
NL = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/nolabel'
sys.path.insert(0, NL)
import numpy as np
import chain_eval_m_nl as CE
G = CE.G; NLA = CE.NLA; OUT = os.path.dirname(os.path.abspath(__file__))
W = float(json.load(open(NLA + '/w_sim.json', encoding='utf-8'))['w'])
R = {sp: CE.build(sp) for sp in ('val', 'test')}
cls = lambda K, lum: int(np.argmin(np.abs(np.asarray(K['col_c']) - lum))) if lum is not None else -1

def bias_from_recs(recs, K):
    lr = {}
    for r in recs:
        if r.get('z_kp') and len(r['pred']):
            j = int(np.argmax(r['conf'])); zf = float(np.asarray(r['pred']).reshape(-1, 9)[j][2])
            if zf > 0 and r['z_kp'] > 0:
                lr.setdefault(cls(K, r['lum']), []).append(np.log(r['z_kp'] / zf))
    return {k: float(np.median(v)) for k, v in lr.items()}, {k: len(v) for k, v in lr.items()}

B_val, n_val = bias_from_recs(*R['val'])
# --- train images: same quantity, z_fin = detector z (top box) x exp(d - c_k) with the deployed stage-3 net (unc_train_z2),
#     z_kp = corner-PnP distance on train frames (ztrain_nl075_train1_ouS6I2M.pkl, same 4 members / SA-TM / PnP as deployment)
K = CE.constants('train'); U = CE.load_unc('train'); lum = U['lum']
recs = pickle.load(open(NLA + '/recs/nl075_train.pkl', 'rb'))['records']
assert [(r['seq_id'], r['frame_id']) for r in recs] == [tuple(k) for k in U['keys']]
Z = pickle.load(open(NLA + '/ztrain_nl075_train1_ouS6I2M.pkl', 'rb'))
lr = {}
nmiss = 0
for r, u in zip(recs, U['unc']):
    if not len(r['pred']):
        continue
    k = (str(r['seq_id']), str(r['frame_id']))
    z = Z.get(k) or Z.get((r['seq_id'], r['frame_id']))
    if z is None or not z['z_kp'] or z['z_kp'] <= 0:
        nmiss += 1; continue
    j = int(np.argmax(np.asarray(r['conf']).reshape(-1)))
    p = np.asarray(r['pred'], np.float64).reshape(-1, 9)
    if z['det'] is not None and np.abs(z['det'][:3] - p[j, :3]).max() > 1e-3:
        nmiss += 1; continue          # z_kp computed on a different box
    lu = lum.get(k)
    cc = K['cal_col'][cls(K, lu)] if lu is not None else K['c_glob']
    d = np.asarray(u['d'], np.float64).reshape(-1)
    ok = np.isfinite(d[j]) and abs(d[j]) <= CE.GUARD
    zf = p[j, 2] * (np.exp(d[j] - cc) if ok else 1.0)
    if zf > 0:
        lr.setdefault(cls(K, lu), []).append(np.log(z['z_kp'] / zf))
B_tr = {k: float(np.median(v)) for k, v in lr.items() if k >= 0}
n_tr = {k: len(v) for k, v in lr.items()}
print('b_k val images', {k: round(v, 4) for k, v in B_val.items()}, n_val)
print('b_k train images', {k: round(v, 4) for k, v in B_tr.items()}, n_tr, 'skipped', nmiss)

def predict(sp, w, B):
    recs, K = R[sp]; out = {}
    for r in recs:
        if not len(r['pred']):
            continue
        j = int(np.argmax(r['conf'])); q = np.asarray(r['pred'], np.float64).reshape(-1, 9)[j].copy(); k = cls(K, r['lum'])
        if w > 0 and r.get('z_kp') and r['z_kp'] > 0 and q[2] > 0 and k in B:
            lz = (1 - w) * np.log(q[2]) + w * (np.log(r['z_kp']) - B[k]); q[:3] *= np.exp(lz) / q[2]
        out[(str(r['seq_id']), str(r['frame_id']))] = (q, float(np.asarray(r['conf']).reshape(-1)[j]))
    return out

res = dict(w=W, b_val_images=B_val, n_val=n_val, b_train_images=B_tr, n_train=n_tr, rows={})
for name, B in (('b_k from val images (paper)', B_val), ('b_k from train images', B_tr)):
    for sp in ('val', 'test'):
        pd = predict(sp, W, B)
        M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp))
        idx = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}
        valid = np.where(M['n_gt'] > 0)[0]; N = len(M['seq'])
        P = {'pt': np.full((N, 3), np.nan), 'plwh': np.full((N, 3), np.nan), 'peul': np.full((N, 3), np.nan), 'conf': np.full(N, np.nan)}
        for k, (q, c) in pd.items():
            i = idx.get(k)
            if i is not None:
                P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = q[:3], q[3:6], q[6:9], c
        m = CE.PT.metrics(P, M, valid)
        pe = float(np.nanmedian(np.linalg.norm(P['pt'] - M['gt'][:, :3], axis=1)[valid]))
        row = {k: float(m[k]) for k in ('AP', 'ATE', 'ASE', 'AOE', 'mDR', 'UDS')}; row['pos_med'] = pe
        res['rows'].setdefault(name, {})[sp] = row
        print('%-28s %-4s AP %.2f ATE %.3f ASE %.2f AOE %.2f mDR %.1f UDS %.2f pos_med %.3f' % (name, sp, row['AP'], row['ATE'], row['ASE'], row['AOE'], row['mDR'], row['UDS'], pe))
json.dump(res, open(OUT + '/r3_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
