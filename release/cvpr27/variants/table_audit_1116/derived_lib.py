import sys, pickle, numpy as np
sys.path.insert(0, 'D:/UAV3D_baselines_pack/scripts'); import score as S
METAS = {'real_test': 'D:/UAV3D_baselines_pack/data/gt/real_test_meta.npz',
         'outdoor_test': 'D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz'}
MM = {k: np.load(v) for k, v in METAS.items()}
def load(pred, split):
    M = MM[split]; key = {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}; N = len(key)
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(pred, 'rb')):
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], float).reshape(-1, 9); s = np.asarray(r['scores'], float).reshape(-1)
        if i is None or not len(b): continue
        j = int(np.argmax(s)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], s[j]
    return P, M
def met(P, M, split, sel=None):
    N = len(M['gt']); sel = np.arange(N) if sel is None else sel
    r = S.metrics(P, M['gt'], M['K'], sel, S.CFG[split])
    if split == 'outdoor_test': r['ATE'] *= 100
    return r
