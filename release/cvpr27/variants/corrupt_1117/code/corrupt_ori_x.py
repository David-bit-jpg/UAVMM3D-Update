# -*- coding: utf-8 -*-
"""[corrupt_1117] test-time path of one trained srcx chain on a (possibly corrupted) REAL TEST set. Inference only.
  python corrupt_ori_x.py <arm> <cond>        arm = out2antifs | out2antifs_ir | out2antifs_lid | out2antifs_none | in2mav | in2mav_none | ...
                                              cond = clean | <corr>_s<sev>   (corr in corrupt.CORR)
Steps (= run_x.sh test path; chain files are only READ; every output goes to review1113/corrupt_1117/<arm>/<cond>/ and
D:/S2MUAV3D/corrupt_1117/work/<arm>/<cond>/):
  1 det_infer_x: the chain's 3 self-trained detectors (det1.pkl['ckpt']), ensemble + flip, KNOWN size = the chain's s_bar (det1.pkl['size']),
    split test only, cache = the corrupted copy of the 512x288 test cache. train/val records = the chain's (clean) ones.
  2 ori_x crops te1 from the (corrupted) raw frames; 3 student read (ORI_MIRROR=1, ou/S_ab0,1.pth);
  4 height: full -> height net with BN re-estimated on the chain's CLEAN tr1 crops (cap 80, = run_x.sh) then read on te1; else constant aspect.
  5 refiner_x infer ref_st3_s5/s6 on test (corrupted raw frames) + avg_inf;
  6 chain_x deploy with constants from the chain's clean train/val inputs (cc, c_k, r_k from train; b_k from val); test inputs = ours.
  7 score_x (pack metric code) on the target's sealed test meta."""
import os, sys, json, pickle, shutil, time
import numpy as np

ARM, COND = sys.argv[1], sys.argv[2]
_p = ARM.split('_'); _st = _p[0].split('2')
os.environ['SX_SRC'], os.environ['SX_TGT'] = _st[0], _st[1]
os.environ['SX_TEACH'] = _p[1] if len(_p) > 1 else 'full'
for k, v in dict(PYTHONUTF8='1', TORCH_HOME='E:/torch_home', UAV_PNP_EPNP_FALLBACK='1').items():
    os.environ.setdefault(k, v)
HERE = os.path.dirname(os.path.abspath(__file__))
R = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/corrupt_1117'
OUT = '%s/%s/%s' % (R, ARM, COND); WK = 'D:/S2MUAV3D/corrupt_1117/work/%s/%s' % (ARM, COND)
for d in (OUT + '/recs', OUT + '/ref', OUT + '/ou', OUT + '/preds', WK):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, HERE); import corrupt as CR  # noqa
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib')
import sx  # noqa
assert sx.COMBO == ARM, (sx.COMBO, ARM)
CH, CW = sx.O, sx.W      # chain dirs (read only)
LOG = open(OUT + '/run.log', 'a', encoding='utf-8')


def say(*a):
    s = time.strftime('%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a); print(s, flush=True); print(s, file=LOG, flush=True)


if COND == 'clean':
    corr = sev = None; DATA_PATH = sx.REAL
else:
    corr, sev = COND.rsplit('_s', 1); sev = int(sev); assert corr in CR.CORR
    DATA_PATH = '%s/%s/%s_s%d' % (CR.DST, sx.TGT, corr, sev)
    assert os.path.exists(DATA_PATH + '/test/DONE'), 'build the corrupted cache first: ' + DATA_PATH
    _raw = sx.imread_raw

    def imread_raw_corrupt(m):
        return CR.apply(_raw(m), corr, sev, m['seq'], m['frame'])
    sx.imread_raw = imread_raw_corrupt     # ori_x / refiner_x call sx.imread_raw (module attribute) -> corrupted raw test frames
say('start', ARM, COND, 'cache', DATA_PATH)
D1 = pickle.load(open(CH + '/recs/det1.pkl', 'rb'))

# ---------------- 1 detector on test
DET = OUT + '/recs/det1.pkl'
if not os.path.exists(DET):
    src = None
    if sx.TEACH == 'ir':   # ir chain reuses the full chain's self-trained detectors -> identical records; reuse if already computed
        f = '%s/%s/%s/recs/det1.pkl' % (R, ARM.rsplit('_', 1)[0], COND)
        if os.path.exists(f) and list(pickle.load(open(f, 'rb'))['ckpt']) == list(D1['ckpt']):
            src = f
    if src:
        T = pickle.load(open(src, 'rb'))['recs']['test']; say('det: reuse test records of', src)
    else:
        import det_infer_x as DI
        cfg, model, logger = DI.build(D1['ckpt'], sx.REAL_CFG)
        model.model_cfg.POST_PROCESSING.SIZE_SOURCE = 'known'
        model.model_cfg.POST_PROCESSING.KNOWN_SIZE = [float(v) for v in D1['size']]
        cfg.DATA_CONFIG.DATA_PATH = DATA_PATH
        T = DI.run(cfg, model, logger, 'test')
        del model
        import torch; torch.cuda.empty_cache()
    c = np.array([x['conf'] for x in T])
    say('det test: %d frames, conf median %.3f, >=0.5 %.1f%%' % (len(c), np.median(c), 100 * (c >= .5).mean()))
    pickle.dump(dict(size=D1['size'], ckpt=D1['ckpt'], recs=dict(train=D1['recs']['train'], val=D1['recs']['val'], test=T),
                     note='[corrupt_1117 %s] train/val = chain records (clean); test = this run' % COND), open(DET, 'wb'))
if COND == 'clean':   # sanity: detector records equal the chain's
    A, B = D1['recs']['test'], pickle.load(open(DET, 'rb'))['recs']['test']
    same = len(A) == len(B) and all(a['seq'] == b['seq'] and a['frame'] == b['frame'] and ((a['box'] is None and b['box'] is None) or
                                    (a['box'] is not None and b['box'] is not None and np.array_equal(a['box'], b['box']))) and a['conf'] == b['conf'] for a, b in zip(A, B))
    mx = max([float(np.abs(a['box'] - b['box']).max()) for a, b in zip(A, B) if a['box'] is not None and b['box'] is not None] + [0])
    say('SANITY det test records identical to chain: %s (max |box diff| %.3g)' % (same, mx))

# ---------------- 2-4 orientation branch on te1
import ori_x as OO  # noqa
OO.W = WK
if not os.path.exists(WK + '/oc_te1.pkl'):
    OO.cmd_crops(DET, 'test', 1, 'te1')
RD = OUT + '/ou/read_te1_stu.pkl'
if not os.path.exists(RD):
    os.environ['ORI_MIRROR'] = '1'
    OO.cmd_read('te1', RD, [CH + '/ou/S_ab0.pth', CH + '/ou/S_ab1.pth'])
KPF = OUT + '/kps_te1.pkl'
if not os.path.exists(KPF):
    if sx.TEACH == 'full':
        import torch
        for p in ('E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kzlid', 'E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kzbody', 'E:/Open3DUAVDet/output/camnorm/geo1004/lidar/kpsize'):
            sys.path.insert(0, p)
        import uavdet3d.model.detectors as dets
        from center_det_mm import CenterDetMM
        from center_det_mmk import CenterDetMMK
        dets.__all__['CenterDetMM'] = CenterDetMM; dets.__all__['CenterDetMMK'] = CenterDetMMK
        OO.CFG = sx.S['kps_cfg']
        m = OO.build(sx.S['kps'])
        # = ori_x.cmd_kps adabn part, on the chain's CLEAN tr1 crops (cap 80 per sequence, seed 0)
        X = np.load('%s/oc_tr1.npy' % CW, mmap_mode='r'); info = pickle.load(open('%s/oc_tr1.pkl' % CW, 'rb'))['info']
        el = [i for i, r in enumerate(info) if r['conf'] >= 0.5]
        rng = np.random.default_rng(0); by = {}
        for i in el:
            by.setdefault(info[i]['seq'], []).append(i)
        el = sorted(sum([list(rng.choice(v, min(80, len(v)), replace=False)) for v in by.values()], []))
        bns = [mm for mm in m.modules() if isinstance(mm, torch.nn.modules.batchnorm._BatchNorm)]
        for b in bns:
            b.reset_running_stats(); b.momentum = None; b.train()
        with torch.no_grad():
            for _, x in OO.batches(X, np.array(el), 16):
                tb = {'image': x, 'batch_size': x.shape[0]}
                for mod in m.module_list:
                    tb = mod(tb)
                    if mod is m.dense_head_2d:
                        break
                f = tb['features_2d']; m.kps_head(f.reshape(f.shape[0] * f.shape[1], *f.shape[2:]))
        m.eval(); say('kps adabn on %d clean tr1 crops' % len(el))
        X = np.load('%s/oc_te1.npy' % WK, mmap_mode='r'); info = pickle.load(open('%s/oc_te1.pkl' % WK, 'rb'))['info']
        res = {}
        with torch.no_grad():
            for ii, x in OO.batches(X, np.arange(len(info)), 16):
                tb = {'image': x, 'batch_size': x.shape[0]}
                for mod in m.module_list:
                    tb = mod(tb)
                    if mod is m.dense_head_2d:
                        break
                pc = tb.get('pred_center_dict', m.dense_head_2d.forward_loss_dict['pred_center_dict'])
                hm = torch.sigmoid(pc['hm'].float()).cpu().numpy().reshape(len(ii), -1, 36, 64)[:, 0]
                f = tb['features_2d']; kk = m.kps_head(f.reshape(f.shape[0] * f.shape[1], *f.shape[2:])).float().cpu().numpy()
                for b, i in enumerate(ii):
                    r0, c0 = int(np.clip(info[i]['vc'] // 8, 0, 35)), int(np.clip(info[i]['uc'] // 8, 0, 63))
                    rs, cs = np.clip(np.arange(r0 - 2, r0 + 3), 0, 35), np.clip(np.arange(c0 - 2, c0 + 3), 0, 63)
                    h = hm[b][np.ix_(rs, cs)]; r, c = np.unravel_index(int(np.argmax(h)), h.shape)
                    L = kk[b, 16:, rs[r], cs[c]].reshape(8, 3); e = np.ptp(L, 0)
                    res[(info[i]['seq'], info[i]['frame'])] = float(e[2] / max(e[:2].mean(), 1e-6))
        pickle.dump(res, open(KPF, 'wb')); say('kps te1: %d crops, ratio median %.3f' % (len(res), np.median(list(res.values()))))
        del m; torch.cuda.empty_cache()
    else:
        s = np.asarray(D1['size'], float); r = float(s[2] / ((s[0] + s[1]) / 2))
        I = pickle.load(open('%s/oc_te1.pkl' % WK, 'rb'))['info']
        pickle.dump({(f['seq'], f['frame']): r for f in I}, open(KPF, 'wb')); say('constant height aspect', r)
if COND == 'clean':
    a, b = pickle.load(open(CH + '/kps_te1.pkl', 'rb')), pickle.load(open(KPF, 'rb'))
    say('SANITY kps_te1 identical: %s (n %d/%d, max diff %.3g)' % (a == b, len(a), len(b), max([abs(a[k] - b[k]) for k in a if k in b] + [0])))

# ---------------- 5 refiner on test
import refiner_x as RX  # noqa
for j in (5, 6):
    f = '%s/ref/fin_s%d_test.pkl' % (OUT, j)
    if not os.path.exists(f):
        RX.cmd_infer('%s/ref/ref_st3_s%d.pt' % (CH, j), DET, 'test', f)
INF = OUT + '/ref/inf_fin_det1_test.pkl'
if not os.path.exists(INF):   # = avg_inf.py
    U = [pickle.load(open('%s/ref/fin_s%d_test.pkl' % (OUT, j), 'rb')) for j in (5, 6)]; RR = {}
    for k in U[0]:
        if not all(k in u for u in U):
            continue
        lum = [u[k]['lum'] for u in U if u[k]['lum'] is not None]
        RR[k] = dict(d=float(np.mean([u[k]['d'] for u in U])), lb=float(np.mean([u[k]['lb'] for u in U])), lum=float(np.mean(lum)) if lum else None)
    pickle.dump(RR, open(INF, 'wb')); say('avg_inf %d keys' % len(RR))
if COND == 'clean':
    a, b = pickle.load(open(CH + '/ref/inf_fin_det1_test.pkl', 'rb')), pickle.load(open(INF, 'rb'))
    say('SANITY inf_fin_det1_test identical: %s' % (a == b))

# ---------------- 6 deploy (= chain_x.cmd_deploy; constants from the chain's clean train / val inputs)
import chain_x as CX  # noqa
from scipy.spatial.transform import Rotation as Rot  # noqa


def zL(oc_pkl, reads):
    I = pickle.load(open(oc_pkl, 'rb')); lwh = np.asarray(I['size'], np.float64)
    Rd = pickle.load(open(reads, 'rb')); out = {}
    for i, f in enumerate(I['info']):
        s = OO.ensemble(Rd['reads'], Rd['order'], i, Rd['cells'][i], f['K_crop'], lwh)
        if s is not None and s[1][2] > 0:
            out[(f['seq'], f['frame'])] = dict(z=float(s[1][2]), R=s[0])
    return out


w = float(json.load(open(CH + '/w_sim.json'))['w'])
D = pickle.load(open(DET, 'rb')); sbar = np.asarray(D['size'], np.float64)
U = {'train': pickle.load(open(CH + '/ref/inf_fin_det1_train.pkl', 'rb')), 'val': pickle.load(open(CH + '/ref/inf_fin_det1_val.pkl', 'rb')),
     'test': pickle.load(open(INF, 'rb'))}
KP = {'tr1': pickle.load(open(CH + '/kps_tr1.pkl', 'rb')), 'val1': pickle.load(open(CH + '/kps_val1.pkl', 'rb')), 'te1': pickle.load(open(KPF, 'rb'))}
ZL = {'val': zL(CW + '/oc_val1.pkl', CH + '/ou/read_val1_stu.pkl'), 'test': zL(WK + '/oc_te1.pkl', RD)}
tr = list(U['train'].values()); cc = CX.two_means(np.array([u['lum'] for u in tr if u['lum'] is not None]))
C = {}
for c in (0, 1):
    C[c] = float(np.median(np.array([u['d'] for u in tr if CX.cls_of(cc, u['lum']) == c and abs(u['d']) <= CX.GUARD])))
C[-1] = float(np.median(np.array([u['d'] for u in tr if abs(u['d']) <= CX.GUARD])))
rk = {c: float(np.median([KP['tr1'][k] for k, u in U['train'].items() if CX.cls_of(cc, u['lum']) == c and k in KP['tr1']])) for c in (0, 1)}
rk_all = float(np.median(list(KP['tr1'].values())))


def base(sp, r):
    k = (r['seq'], r['frame']); q = np.asarray(r['box'], np.float64).copy(); u = U[sp].get(k)
    c = CX.cls_of(cc, u['lum']) if u is not None else -1
    f = 0.0
    if u is not None and abs(u['d']) <= CX.GUARD:
        f = u['d'] - C[c]
    q[:3] *= np.exp(f)
    l, wd = sbar[0] * np.exp(CX.ALPHA * f), sbar[1] * np.exp(CX.ALPHA * f)
    kpk = {'val': 'val1', 'test': 'te1'}[sp]
    h = rk[c] * (l + wd) / 2 if c >= 0 else KP[kpk].get(k, rk_all) * (l + wd) / 2
    q[3:6] = [l, wd, h]
    return q, c


Bv = {}
for r in D['recs']['val']:
    if r['box'] is None:
        continue
    q, c = base('val', r); z = ZL['val'].get((r['seq'], r['frame']))
    if z is not None and c >= 0:
        Bv.setdefault(c, []).append(np.log(z['z'] / q[2]))
Bv = {c: float(np.median(v)) for c, v in Bv.items()}
res = dict(arm=ARM, cond=COND, w=w, s_bar=sbar.tolist(), col_centres=cc.tolist(), c_k=C, r_k=rk, b_k_val_images=Bv, alpha=CX.ALPHA)
chain_dep = json.load(open(CH + '/deploy_pcmd.json'))
res['constants_equal_chain'] = all(json.dumps(res[k], default=float, sort_keys=True) == json.dumps(chain_dep[k], default=float, sort_keys=True)
                                   for k in ('w', 's_bar', 'col_centres', 'r_k', 'c_k', 'b_k_val_images'))
P = []; nf = nr = 0
for r in D['recs']['test']:
    if r['box'] is None:
        continue
    q, c = base('test', r); z = ZL['test'].get((r['seq'], r['frame']))
    if z is not None:
        q[6:9] = Rot.from_matrix(z['R']).as_euler('xyz'); nr += 1
    if w > 0 and z is not None and c in Bv:
        lz = (1 - w) * np.log(q[2]) + w * (np.log(z['z']) - Bv[c])
        q[:3] *= np.exp(lz) / q[2]; nf += 1
    P.append(dict(seq=r['seq'], frame=r['frame'], boxes9d=q.reshape(1, 9), scores=np.array([r['conf']])))
PF = OUT + '/preds/pcmd_fused_test.pkl'
pickle.dump(P, open(PF, 'wb')); res['fused_test'] = dict(n=len(P), n_fused=nf, n_rot_student=nr, file=PF)
json.dump(res, open(OUT + '/deploy_pcmd.json', 'w'), indent=1, default=float)
say('deploy: n %d fused %d rot %d; constants equal chain: %s; c_k %s b_k %s' % (len(P), nf, nr, res['constants_equal_chain'], C, Bv))

# ---------------- 7 score (pack metric code via score_x.score_meta on the sealed test meta)
import score_x as SC  # noqa
sres = SC.score_meta(PF, sx.T['gt_meta'])
SPL = sx.T['score_split']
json.dump(sres, open(PF + '.score_%s.json' % SPL, 'w'), indent=1, default=float)
say('score all: ' + '  '.join('%s %.3f' % (k, v) for k, v in sres['all'].items()))
if COND == 'clean':
    ref = json.load(open(CH + '/preds/pcmd_fused_test.pkl.score_%s.json' % SPL))
    same = json.dumps(ref, sort_keys=True) == json.dumps(json.load(open(PF + '.score_%s.json' % SPL)), sort_keys=True)
    Pc = pickle.load(open(CH + '/preds/pcmd_fused_test.pkl', 'rb'))
    pe = len(Pc) == len(P) and all(a['seq'] == b['seq'] and a['frame'] == b['frame'] and np.array_equal(a['boxes9d'], b['boxes9d']) for a, b in zip(Pc, P))
    say('SANITY score json identical to chain: %s ; prediction boxes identical: %s' % (same, pe))
    open(OUT + '/SANITY_%s' % ('OK' if same else 'FAIL'), 'w').write('score json identical: %s; preds identical: %s\n' % (same, pe))
say('done', ARM, COND)
