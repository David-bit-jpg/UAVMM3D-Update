# -*- coding: utf-8 -*-
"""[review1113/cpu5] Five CPU-only checks, existing predictions only (no training, no GPU, no inference). Writes only to review1113/cpu5/.
 1. Pooled paired sequence bootstrap of PST - RGB-only over 3 seed-run pairs (MAV6D test/val), + per-pair intervals.
 2. IR<->LiDAR gate off (sup2/nogate) - PST (in2mav), paired sequence bootstrap.
 3. Provenance checks of the fused vs IR-only (omega = 0) comparison (hashes of shared inputs, constants).
 4. Single model, no TTA vs ensemble on val (MAV6D: sequence bootstrap; AntiUAV3D: block bootstrap L=53 val / 176 test).
 5. AntiUAV3D out2antifs: (c) det1 box + student rotation (detector distance), (c') + height r_k, (d) pcmd_nofuse (depth students, no fusion);
    paired block bootstrap (176 test / 53 val).
MAV6D metrics = review1113/clean2/buildup/buildup.py (asserted == pack score.metrics, real_test). AntiUAV3D = pack score.py outdoor_test,
same vectorised AP as review1113/A3/a_block_boot.py (asserted == S.metrics)."""
import os, sys, json, pickle, hashlib, importlib.util
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from scipy.spatial.transform import Rotation as Rot
OUT = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
GP = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto'; G = GP.rsplit('/', 1)[0]; R = GP + '/review1113'; X = GP + '/srcx'
B = 2000


def mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


RES = {}
# ---------------------------------------------------------------- MAV6D part (items 1, 2, 4a) ----------------------------------------------
os.environ.update(SX_SRC='in', SX_TGT='mav', SX_TEACH='full')
BU = mod('buildup', R + '/clean2/buildup/buildup.py'); S = BU.S; CFGM = BU.CFG
MAV = {'PST1': X + '/in2mav/preds/pcmd_fused_%s.pkl', 'RGB1': X + '/in2mav_none/preds/pcmd_fused_%s.pkl',
       'PST2': R + '/sup3/run/preds/pcmd_fused_%s.pkl', 'RGB2': R + '/seeds/mav_none_s10/preds/pcmd_fused_%s.pkl',
       'PST3': R + '/seeds/mav_full_s20/preds/pcmd_fused_%s.pkl', 'RGB3': R + '/seeds/mav_none_s20/preds/pcmd_fused_%s.pkl',
       'NOGATE': R + '/sup2/nogate/preds/nogate_fused_%s.pkl', 'SINGLE': R + '/clean/single/preds/single_fused_%s.pkl'}
SEEDPAIRS = [('PST1', 'RGB1'), ('PST2', 'RGB2'), ('PST3', 'RGB3')]
KEYS = ('UDS', 'AP', 'ATE', 'ASE', 'AOE', 'mDR')


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


RES['mav'] = {}
for sp in ('test', 'val'):
    M = np.load('%s/cmpeval/%s_meta.npz' % (G, sp)); seq = np.array([str(s) for s in M['seq']]); N = len(seq)
    FF, rows = {}, {}
    for nm, p in MAV.items():
        P = BU.load_list(pickle.load(open(p % sp, 'rb')), M); F = BU.frame_q(P, M); s = BU.stats(F, np.arange(N))
        ref = S.metrics(P, M['gt'], M['K'], np.arange(N), CFGM)
        assert all(abs(ref[k] - s[k]) < 1e-9 for k in KEYS), (nm, ref, s)
        sj = (p % sp) + ('.score_real_test.json' if sp == 'test' else '.score_val.json')
        stored = json.load(open(sj))['all']['UDS'] if os.path.exists(sj) else None
        if stored is not None:
            assert abs(stored - s['UDS']) < 1e-6, (nm, stored, s['UDS'])
        FF[nm] = F; rows[nm] = {k: float(s[k]) for k in KEYS}; rows[nm]['stored_UDS'] = stored
    us = np.unique(seq); idx = {u: np.where(seq == u)[0] for u in us}
    rng = np.random.default_rng(0)
    D = {k: [] for k in ('pair1', 'pair2', 'pair3', 'pooled', 'pooled_AP', 'pooled_ATE', 'pooled_AOE', 'nogate', 'nogate_AP', 'nogate_ATE', 'nogate_AOE',
                         'single', 'single_AP', 'single_ATE', 'single_AOE')}
    for _ in range(B):
        ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        st = {nm: BU.stats(FF[nm], ii) for nm in MAV}
        dd = [st[a]['UDS'] - st[b]['UDS'] for a, b in SEEDPAIRS]
        for j in range(3):
            D['pair%d' % (j + 1)].append(dd[j])
        D['pooled'].append(np.mean(dd))
        for k in ('AP', 'ATE', 'AOE'):
            D['pooled_' + k].append(np.mean([st[a][k] - st[b][k] for a, b in SEEDPAIRS]))
        D['nogate'].append(st['NOGATE']['UDS'] - st['PST1']['UDS']); D['single'].append(st['SINGLE']['UDS'] - st['PST1']['UDS'])
        for k in ('AP', 'ATE', 'AOE'):
            D['nogate_' + k].append(st['NOGATE'][k] - st['PST1'][k]); D['single_' + k].append(st['SINGLE'][k] - st['PST1'][k])
    pt = {'pair%d' % (j + 1): rows[a]['UDS'] - rows[b]['UDS'] for j, (a, b) in enumerate(SEEDPAIRS)}
    pt['pooled'] = float(np.mean([pt['pair1'], pt['pair2'], pt['pair3']]))
    for k in ('AP', 'ATE', 'AOE'):
        pt['pooled_' + k] = float(np.mean([rows[a][k] - rows[b][k] for a, b in SEEDPAIRS]))
        pt['nogate_' + k] = rows['NOGATE'][k] - rows['PST1'][k]; pt['single_' + k] = rows['SINGLE'][k] - rows['PST1'][k]
    pt['nogate'] = rows['NOGATE']['UDS'] - rows['PST1']['UDS']; pt['single'] = rows['SINGLE']['UDS'] - rows['PST1']['UDS']
    boot = {k: dict(point=float(pt[k]), ci=ci(v), frac_le0=float(np.mean(np.array(v) <= 0))) for k, v in D.items()}
    perseq = {}
    for nm, (a, b) in {'pair1': SEEDPAIRS[0], 'pair2': SEEDPAIRS[1], 'pair3': SEEDPAIRS[2], 'nogate': ('NOGATE', 'PST1'), 'single': ('SINGLE', 'PST1')}.items():
        perseq[nm] = int(sum(BU.stats(FF[a], idx[u])['UDS'] > BU.stats(FF[b], idx[u])['UDS'] for u in us))
    pooled_seq = int(sum(np.mean([BU.stats(FF[a], idx[u])['UDS'] - BU.stats(FF[b], idx[u])['UDS'] for a, b in SEEDPAIRS]) > 0 for u in us))
    perseq['pooled'] = pooled_seq
    RES['mav'][sp] = dict(rows=rows, boot=boot, n_seq=int(len(us)), n_frames=int(N), seqs_a_beats_b=perseq,
                          seed_mean={'PST': {k: float(np.mean([rows['PST%d' % j][k] for j in (1, 2, 3)])) for k in KEYS},
                                     'RGB': {k: float(np.mean([rows['RGB%d' % j][k] for j in (1, 2, 3)])) for k in KEYS}})
    print('mav', sp, json.dumps({k: [round(v['point'], 2)] + [round(x, 2) for x in v['ci']] for k, v in boot.items()}), flush=True)

# ---------------------------------------------------------------- item 3 provenance -------------------------------------------------------


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 22), b''):
            h.update(ch)
    return h.hexdigest()


A, W0 = X + '/in2mav', R + '/clean2/w0'
same = {}
for rel in ('recs/det0.pkl', 'recs/det1.pkl', 'ou/read_tr1_stu.pkl', 'ou/read_val1_stu.pkl', 'ou/read_te1_stu.pkl', 'kps_tr1.pkl', 'kps_val1.pkl', 'kps_te1.pkl',
            'ref/ref_st2_s5.pt', 'ref/ref_st2_s6.pt', 'ref/inf_st2_det1_train.pkl', 'ref/real_d0_meta.pkl', 'ref/sim_meta.pkl'):
    pa, pb = A + '/' + rel, W0 + '/' + rel
    same[rel] = (md5(pa) == md5(pb)) if os.path.exists(pa) and os.path.exists(pb) else 'missing: %s %s' % (os.path.exists(pa), os.path.exists(pb))
for rel in ('ref/dp_fused.npy', 'ref/ref_st3_s5.pt', 'ref/ref_st3_s6.pt'):
    pa, pb = A + '/' + rel, W0 + '/' + rel
    same[rel + ' (expected to differ)'] = (md5(pa) == md5(pb)) if os.path.exists(pa) and os.path.exists(pb) else 'missing'
dpa, dpb = np.load(A + '/ref/dp_fused.npy'), np.load(W0 + '/ref/dp_fused.npy')
DA, DB = json.load(open(A + '/deploy_pcmd.json')), json.load(open(W0 + '/deploy_w0.json'))
RES['item3'] = dict(hash_equal=same, dp_finite_same_mask=bool(np.array_equal(np.isfinite(dpa), np.isfinite(dpb))), dp_n=int(np.isfinite(dpa).sum()),
                    dp_mean_abs_diff=float(np.nanmean(np.abs(dpa - dpb))),
                    deploy={k: dict(PST=DA.get(k), w0=DB.get(k)) for k in ('w', 's_bar', 'col_centres', 'c_k', 'r_k', 'b_k_val_images', 'alpha')})
print('item3', json.dumps(RES['item3'], default=str)[:3000], flush=True)

# ---------------------------------------------------------------- AntiUAV3D part (items 4b, 5) ---------------------------------------------
os.environ.update(SX_SRC='out', SX_TGT='antifs', SX_TEACH='full')
for k in [k for k in sys.modules if k in ('sx', 'chain_x', 'ori_x', 'score_x')]:
    del sys.modules[k]
sys.path.insert(0, X + '/lib')
import sx  # noqa
assert sx.O.endswith('out2antifs'), sx.O
import chain_x as CX  # noqa
SP = mod('score_pack', 'D:/UAV3D_baselines_pack/scripts/score.py'); CO = SP.CFG['outdoor_test']


def anti_meta(sp):
    if sp == 'test':
        M = np.load('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz')
        return dict(gt=M['gt'], K=M['K'], seq=np.array([str(s) for s in M['seq']]), frame=np.array([str(f) for f in M['frame']]))
    V = pickle.load(open('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/val.pkl', 'rb'))
    return dict(gt=np.array([r['box9d_ours'] for r in V], np.float64), K=np.array([r['K'] for r in V], np.float64),
                seq=np.array([str(r['seq']) for r in V]), frame=np.array([str(r['frame']) for r in V]))


def a_load(L_, M):
    key = {(s, f): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))}; N = len(M['seq'])
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in L_:
        i = key.get((str(r['seq']), str(r['frame']))); b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); sc = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b):
            continue
        j = int(np.argmax(sc)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], sc[j]
    return P


def a_frame(P, M):
    g, K = M['gt'], M['K']; N = len(g)
    has = np.isfinite(P['pt']).all(1) & np.isfinite(P['conf'])
    d = np.linalg.norm(P['pt'] - g[:, :3], axis=1) / np.linalg.norm(g[:, :3], axis=1)
    tp = np.zeros(N, bool); se = np.full(N, np.nan); ae = np.full(N, np.nan)
    for i in np.where(has)[0]:
        fe = np.isfinite(P['peul'][i]).all(); eul = P['peul'][i] if fe else np.zeros(3)
        if SP.iou(SP.box2d(P['pt'][i], P['plwh'][i], eul, K[i]), SP.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], K[i])) < 0.1:
            continue
        tp[i] = True; se[i] = 100 * np.mean(np.abs(P['plwh'][i] - g[i, 3:6]))
        ae[i] = SP.sym(Rot.from_euler('xyz', eul).as_matrix(), Rot.from_euler('xyz', g[i, 6:9]).as_matrix()) if fe else 90.0
    return dict(has=has, conf=P['conf'], d=d, tp=tp, se=se, ae=ae)


def a_stats(F, idx):
    h = F['has'][idx]; n = len(idx)
    ap = 100 * np.mean([BU.fast_ap(F['conf'][idx][h], F['d'][idx][h] <= th, n) for th in CO['thr']])
    t = idx[F['tp'][idx]]
    if not len(t):
        return dict(UDS=50 * ap / 100, AP=ap, ATE=np.nan, ASE=np.nan, AOE=np.nan, mDR=0.0)
    ATE, ASE, AOE = F['d'][t].mean(), F['se'][t].mean(), F['ae'][t].mean(); mdr = len(t) / n
    uds = 100 * (0.5 * ap / 100 + 0.5 * np.mean([1 - min(1, ATE / CO['a']), 1 - min(1, ASE / 10.0), 1 - min(1, AOE / 57.2958), mdr]))
    return dict(UDS=uds, AP=ap, ATE=ATE, ASE=ASE, AOE=AOE, mDR=100 * mdr)


def groups_of(M):
    seq = M['seq']; frm = np.array([int(f.split('.')[0]) for f in M['frame']]); gr = []
    for s in sorted(np.unique(seq)):
        ix = np.where(seq == s)[0]; gr.append(ix[np.argsort(frm[ix], kind='stable')])
    return gr


def block_indices(rng, groups, L):   # = review1113/A3/a_block_boot.py
    out = []
    for ix in groups:
        n = len(ix); Lg = min(L, n); nb = int(np.ceil(n / Lg))
        st = rng.integers(0, n, nb); pos = (st[:, None] + np.arange(Lg)[None, :]) % n
        out.append(ix[pos.ravel()[:n]])
    return np.concatenate(out)


AO = X + '/out2antifs'


def build_c(sp, with_rk):
    """= clean2 buildup.build_c on out2antifs: det1 boxes (own distance, own s_bar size), rotation = student ensemble by the deploy rule;
    (c') adds the deploy height h = r_k (l+w)/2 with the refiner correction forced to 0."""
    D = pickle.load(open(AO + '/recs/det1.pkl', 'rb')); sbar = np.asarray(D['size'], np.float64)
    DEP = json.load(open(AO + '/deploy_pcmd.json')); cc = np.asarray(DEP['col_centres']); rk = {int(k): v for k, v in DEP['r_k'].items()}
    U = pickle.load(open('%s/ref/inf_fin_det1_%s.pkl' % (AO, sp), 'rb'))
    nm = {'val': 'val1', 'test': 'te1'}[sp]
    KP = pickle.load(open('%s/kps_%s.pkl' % (AO, nm), 'rb')); KPtr = pickle.load(open(AO + '/kps_tr1.pkl', 'rb')); rk_all = float(np.median(list(KPtr.values())))
    ZL = CX.zL_table(nm, '%s/ou/read_%s_stu.pkl' % (AO, nm))
    out = []; nr = 0
    for r in D['recs'][sp]:
        if r['box'] is None:
            continue
        k = (r['seq'], r['frame']); q = np.asarray(r['box'], np.float64).copy()
        if with_rk:
            u = U.get(k); c = CX.cls_of(cc, u['lum']) if u is not None else -1
            l, wd = sbar[0], sbar[1]
            q[3:6] = [l, wd, rk[c] * (l + wd) / 2 if c >= 0 else KP.get(k, rk_all) * (l + wd) / 2]
        z = ZL.get(k)
        if z is not None:
            q[6:9] = Rot.from_matrix(z['R']).as_euler('xyz'); nr += 1
        out.append(dict(seq=r['seq'], frame=r['frame'], boxes9d=q.reshape(1, 9), scores=np.array([r['conf']])))
    return out, nr


RES['anti'] = {}
for sp, L in (('test', 176), ('val', 53)):
    M = anti_meta(sp); N = len(M['seq'])
    D1 = pickle.load(open(AO + '/recs/det1.pkl', 'rb'))['recs'][sp]
    det1 = [dict(seq=r['seq'], frame=r['frame'], boxes9d=np.asarray(r['box'], np.float64).reshape(1, 9), scores=np.array([r['conf']])) for r in D1 if r['box'] is not None]
    cL, nrc = build_c(sp, False); cpL, _ = build_c(sp, True)
    for nm_, L_ in (('anti_c_students', cL), ('anti_cp_students_rk', cpL), ('anti_b_det1', det1)):
        pickle.dump(L_, open('%s/%s_%s.pkl' % (OUT, nm_, sp), 'wb'))
    arms = {'ENS (PST fused)': pickle.load(open(AO + '/preds/pcmd_fused_%s.pkl' % sp, 'rb')),
            'SINGLE (no TTA, fused)': pickle.load(open(R + '/M12/anti/single/preds/single_fused_%s.pkl' % sp, 'rb')),
            '(b) det1 (R_det)': det1, '(c) det1 + students (detector distance)': cL, "(c') (c) + height r_k": cpL,
            '(d) pcmd_nofuse (depth students, no fusion)': pickle.load(open(AO + '/preds/pcmd_nofuse_%s.pkl' % sp, 'rb'))}
    stored = {'ENS (PST fused)': AO + '/preds/pcmd_fused_%s.pkl', 'SINGLE (no TTA, fused)': R + '/M12/anti/single/preds/single_fused_%s.pkl',
              '(d) pcmd_nofuse (depth students, no fusion)': AO + '/preds/pcmd_nofuse_%s.pkl'}
    FF, rows = {}, {}
    for nm_, L_ in arms.items():
        P = a_load(L_, M); F = a_frame(P, M); s = a_stats(F, np.arange(N))
        ref = SP.metrics(P, M['gt'], M['K'], np.arange(N), CO)
        assert all(abs(ref[k] - s[k]) < 1e-9 for k in KEYS), (nm_, ref, s)
        st_ = None
        if nm_ in stored:
            jf = (stored[nm_] % sp) + ('.score_antifs_test.json' if sp == 'test' else '.score_val.json')
            if os.path.exists(jf):
                st_ = json.load(open(jf))['all']['UDS']; assert abs(st_ - s['UDS']) < 1e-6, (nm_, st_, s)
        FF[nm_] = F; rows[nm_] = {k: float(s[k]) for k in KEYS}; rows[nm_]['stored_UDS'] = st_
    names = list(arms)
    pairs = [(names[1], names[0]), (names[3], names[2]), (names[5], names[3]), (names[4], names[3]), (names[5], names[4]), (names[0], names[5])]
    gr = groups_of(M); rng = np.random.default_rng(0)
    acc = {p: {k: [] for k in ('UDS', 'ATE', 'AP', 'AOE', 'ASE')} for p in pairs}
    for _ in range(B):
        ii = block_indices(rng, gr, L); st = {nm_: a_stats(FF[nm_], ii) for nm_ in names}
        for a, b in pairs:
            for k in acc[(a, b)]:
                acc[(a, b)][k].append(st[a][k] - st[b][k])
    boot = {'%s  minus  %s' % (a, b): {k: dict(point=float(rows[a][k] - rows[b][k]), ci=ci(v), frac_le0=float(np.mean(np.array(v) <= 0))) for k, v in A_.items()}
            for (a, b), A_ in acc.items()}
    RES['anti'][sp] = dict(rows=rows, boot=boot, L=L, n_frames=int(N), seq_sizes={s: int((M['seq'] == s).sum()) for s in np.unique(M['seq'])}, n_rot_student_c=nrc)
    print('anti', sp, json.dumps(rows, default=lambda x: round(float(x), 3)), flush=True)
    print(json.dumps({k: {m: [round(v['point'], 3)] + [round(x, 3) for x in v['ci']] for m, v in d.items()} for k, d in boot.items()}), flush=True)

json.dump(RES, open(OUT + '/cpu5.json', 'w', encoding='utf-8'), indent=1, default=float)
print('done', flush=True)
