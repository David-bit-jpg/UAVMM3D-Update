# -*- coding: utf-8 -*-
"""[review1113 sensorfree E1geo, 11-15] DEPLOY-ONLY variant of the sensor-free control E1 (MAV6D). CPU only, no GPU step.
usage: python e1geo.py <E1 run dir, e.g. .../sensorfree/E1/mav_s0> <out dir, e.g. .../sensorfree/E1geo/mav_s0> [--ref-prefix e1] [--no-score]

What changes vs E1 (everything upstream of deployment is E1's own, read only; training targets stay ONE-class):
 (a) two brightness classes: chain_x two_means / cls_of run their normal (PST) rule on E1's own refiner 'lum' (silhouette body brightness
     from E1's hull-supervised region head, stored in ref/inf_fin_det1_<split>.pkl) instead of the NOIR one-class shortcut. c_k (train),
     beta'_k (unlabeled val images) and r_k are then per class exactly as chain_x cmd_deploy computes them.
 (b) per-class height ratio r_k from a sensor-free source: box-corner PnP of the RGB-only orientation nets -> least-squares refit of the
     box height with l, w fixed at s_bar (= heads/kpsbox_x.fit_ratio, the H6 rule), per-crop r = h / mean(l, w), r_k = median over the
     unlabeled train crops of class k (chain_x cmd_deploy, unchanged).
     PRIMARY source (scored) = the RGB-only KZ teachers KZs0/1 after E1's AdaBN (ou/T_rgb{0,1}.pth; their train-crop reading
     ou/read_tr1_teach.pkl already exists -> no GPU). Chosen from code, before any result: E1's students are trained on real crops toward
     rigid projections of s_bar-shaped pseudo boxes (ou_train_x: boxes = (t_pnp, lwh = s_bar, R)), so their corner aspect is pulled toward
     the constant aspect E1 already uses; the sim-only teachers are not. The student-derived r_k (read_*_stu.pkl, E1's own students) is
     computed and logged as a diagnostic only (not scored). val/test crops without a class (no lum) fall back to the train median (rk_all),
     because the teacher reading exists for train crops only.
Faithfulness check: the patched chain copy with both switches OFF must reproduce E1's own preds/e1_fused_test.pkl (asserted, no labels).
Test is scored ONCE (skipped if the score file exists). No real label is read except by score_x.py / heads_boot.py on TEST."""
import os, sys, re, shutil, pickle, json, subprocess
import numpy as np

RUN = sys.argv[1].replace('\\', '/').rstrip('/'); OUT = sys.argv[2].replace('\\', '/').rstrip('/')
REFP = sys.argv[sys.argv.index('--ref-prefix') + 1] if '--ref-prefix' in sys.argv else 'e1'
SCORE = '--no-score' not in sys.argv
PY = sys.executable
SRC_LIB = RUN + '/lib' if os.path.isdir(RUN + '/lib') else 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib'
LIB = OUT + '/lib'
os.makedirs(LIB, exist_ok=True); os.makedirs(OUT + '/preds', exist_ok=True)


def say(m):
    print('[E1geo] ' + m, flush=True)


# ---------------------------------------------------------------- 1. isolated lib copies (sx, ori_x, score_x, chain_x patched)
for f in ('sx.py', 'ori_x.py', 'score_x.py', 'chain_x.py'):
    t = open(SRC_LIB + '/' + f, encoding='utf-8').read()
    t = t.replace(SRC_LIB, LIB)
    if f == 'chain_x.py':
        t = '# [E1geo copy of the run lib chain_x.py] two switches, default OFF = behaviour of the source\n' + t
        def rp(a, b):
            global t
            assert t.count(a) == 1, (f, a)
            t = t.replace(a, b)
        rp("GUARD = float(np.log(1.4));", "CLS2 = os.environ.get('E1GEO_CLS2') == '1'; KPS = os.environ.get('E1GEO_KPS', ''); OUTD = os.environ.get('E1GEO_OUT', '')   # [E1geo]\nGUARD = float(np.log(1.4));")
        assert t.count("    if sx.NOIR:") == 2
        t = t.replace("    if sx.NOIR:", "    if sx.NOIR and not CLS2:   # [E1geo] CLS2: two brightness classes (PST rule) despite NOIR")
        rp("    KP = {sp: pickle.load(open(O + '/kps_%s.pkl' % sp, 'rb')) for sp in ('tr1', 'val1', 'te1')}",
           "    KP = {sp: (pickle.load(open(O + '/kps_%s.pkl' % sp, 'rb')) if not KPS else (pickle.load(open(KPS % sp, 'rb')) if os.path.exists(KPS % sp) else {})) for sp in ('tr1', 'val1', 'te1')}   # [E1geo]")
        rp("            fn = '%s/preds/%s_%s_%s.pkl' % (V, prefix, tag, sp)", "            fn = '%s/preds/%s_%s_%s.pkl' % (OUTD or V, prefix, tag, sp)   # [E1geo] outputs under E1geo")
        rp("    json.dump(res, open('%s/deploy_%s.json' % (V, prefix), 'w'), indent=1, default=float)", "    json.dump(res, open('%s/deploy_%s.json' % (OUTD or V, prefix), 'w'), indent=1, default=float)")
        rp("    if sys.argv[1] == 'targets':", "    assert sys.argv[1] == 'deploy'   # [E1geo] deploy only\n    if sys.argv[1] == 'targets':")
    open(LIB + '/' + f, 'w', encoding='utf-8').write(t)
for f in os.listdir(LIB):
    if f.endswith('.py'):
        assert SRC_LIB not in open(LIB + '/' + f, encoding='utf-8').read(), f
os.environ['SX_NOMKDIR'] = '1'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, LIB)
import sx  # noqa
import ori_x as OO  # noqa
assert os.path.normcase(os.path.abspath(sx.O)) == os.path.normcase(os.path.abspath(RUN)), (sx.O, RUN)
assert sx.TEACH == 'none' and sx.NOIR and sx.TGT == 'mav', (sx.TEACH, sx.TGT)
V = sx.O
say('run %s | lib copy %s | W %s' % (V, LIB, sx.W))

# ---------------------------------------------------------------- 2. box-corner height ratios (H6 rule, CPU)
from scipy.optimize import least_squares  # noqa
import cv2  # noqa


def fit_ratio(c2, Kc, lwh):   # = review1113/heads/kpsbox_x.fit_ratio, verbatim logic
    s = OO.pnp(c2, Kc, lwh)
    if s is None:
        return None
    R0, t0 = s
    rv0 = cv2.Rodrigues(np.asarray(R0, np.float64))[0].reshape(3)
    c2 = np.asarray(c2, np.float64)

    def res(p):
        R = cv2.Rodrigues(p[:3].reshape(3, 1))[0]
        X = (OO.PROTO8 * np.array([lwh[0], lwh[1], lwh[2] * np.exp(p[6])])) @ R.T + p[3:6]
        if (X[:, 2] <= 1e-3).any():
            return np.full(16, 1e3)
        q = X @ Kc.T
        return (q[:, :2] / q[:, 2:3] - c2).ravel()
    p0 = np.r_[rv0, t0, 0.0]
    lo = np.r_[np.full(6, -np.inf), -1.5]; hi = np.r_[np.full(6, np.inf), 1.5]
    try:
        r = least_squares(res, p0, bounds=(lo, hi), method='trf', max_nfev=200)
    except Exception:
        return None
    if not r.success or not np.isfinite(r.x).all() or r.x[5] <= 0:
        return None
    return float(lwh[2] * np.exp(r.x[6]) / np.mean(lwh[:2]))


def ratios(name, reads, out):
    if os.path.exists(out):
        return pickle.load(open(out, 'rb'))
    I = pickle.load(open('%s/oc_%s.pkl' % (sx.W, name), 'rb')); lwh = np.asarray(I['size'], np.float64)
    R = pickle.load(open(reads, 'rb')); res = {}; nf = 0
    lim = int(os.environ.get('E1GEO_LIMIT', '0'))
    for i, f in enumerate(I['info'][:lim] if lim else I['info']):
        s = OO.ensemble(R['reads'], R['order'], i, R['cells'][i], f['K_crop'], lwh)   # symmetric ensemble (MODULES 0.2) -> averaged corners
        v = fit_ratio(s[2], np.asarray(f['K_crop'], np.float64), lwh) if s is not None else None
        if v is None:
            nf += 1
        else:
            res[(f['seq'], f['frame'])] = v
    if not lim:
        pickle.dump(res, open(out, 'wb'))
    v = np.array(list(res.values()))
    say('box-corner ratios %s from %s (%d nets): %d / %d crops (failed %d), median %.3f, s_bar aspect %.3f' % (
        name, os.path.basename(reads), len(R['order']), len(res), len(I['info'][:lim] if lim else I['info']), nf, np.median(v), lwh[2] / lwh[:2].mean()))
    return res


KT = OUT + '/kpsbox_teach_%s.pkl'; KS = OUT + '/kpsbox_stu_%s.pkl'
ratios('tr1', V + '/ou/read_tr1_teach.pkl', KT % 'tr1')                     # PRIMARY: RGB-only KZ teachers (AdaBN'd by E1)
for n in ('tr1', 'val1', 'te1'):
    ratios(n, V + '/ou/read_%s_stu.pkl' % n, KS % n)                         # diagnostic: E1's own students (orig + mirror)
if os.environ.get('E1GEO_LIMIT'):
    say('LIMIT plumbing test: stop'); sys.exit(0)


# ---------------------------------------------------------------- 3. deployments with the patched chain copy
def deploy(prefix, cls2, kps, w):
    env = dict(os.environ, E1GEO_CLS2='1' if cls2 else '0', E1GEO_KPS=kps, E1GEO_OUT=OUT, PYTHONUTF8='1')
    lg = open('%s/deploy_%s.log' % (OUT, prefix), 'w')
    subprocess.run([PY, LIB + '/chain_x.py', 'deploy', repr(w), prefix], env=env, stdout=lg, stderr=subprocess.STDOUT, check=True)
    return json.load(open('%s/deploy_%s.json' % (OUT, prefix)))


w = float(json.load(open(V + '/w_sim.json'))['w'])
rep = deploy('e1rep', False, '', w)                                          # switches OFF -> must equal E1's own deployment
A = pickle.load(open('%s/preds/e1rep_fused_test.pkl' % OUT, 'rb')); B = pickle.load(open('%s/preds/%s_fused_test.pkl' % (V, REFP), 'rb'))
assert len(A) == len(B) and all((a['seq'], a['frame']) == (b['seq'], b['frame']) for a, b in zip(A, B))
dmax = max(float(np.abs(np.asarray(a['boxes9d']) - np.asarray(b['boxes9d'])).max()) for a, b in zip(A, B))
assert dmax < 1e-6, dmax
say('faithfulness: patched chain with switches OFF reproduces %s/preds/%s_fused_test.pkl (max |diff| %.1e, %d frames)' % (V, REFP, dmax, len(A)))
geo = deploy('e1geo', True, KT, w)                                           # PRIMARY (scored)
stu = deploy('e1geo_stuR', True, KS, w)                                      # diagnostic (never scored)
cls_only = deploy('e1geo_clsonly', True, '', w)                              # diagnostic: (a) alone, constant height (never scored)
U = pickle.load(open(V + '/ref/inf_fin_det1_test.pkl', 'rb'))
D1 = pickle.load(open(V + '/recs/det1.pkl', 'rb'))['recs']['test']
cc = np.asarray(geo['col_centres'])
nocls = sum(1 for r in D1 if r['box'] is not None and (U.get((r['seq'], r['frame'])) is None or U[(r['seq'], r['frame'])]['lum'] is None))
Utr = pickle.load(open(V + '/ref/inf_fin_det1_train.pkl', 'rb')); lum = np.array([u['lum'] for u in Utr.values() if u['lum'] is not None])
frac1 = float(np.mean(np.abs(lum[:, None] - cc[None]).argmin(1) == 1))
CONST = dict(E1=dict(c_k=rep['c_k'], r_k=rep['r_k'], b_k=rep['b_k_val_images'], col=rep['col_centres']),
             E1geo=dict(c_k=geo['c_k'], r_k=geo['r_k'], b_k=geo['b_k_val_images'], col=geo['col_centres'], train_frac_bright=frac1, test_frames_without_class=nocls),
             diag_student_r_k=stu['r_k'], diag_clsonly=dict(c_k=cls_only['c_k'], r_k=cls_only['r_k'], b_k=cls_only['b_k_val_images']))
json.dump(CONST, open(OUT + '/constants.json', 'w'), indent=1, default=float)
fmt = lambda d: '{' + ', '.join('%s: %.3f' % (k, float(v)) for k, v in sorted(d.items(), key=lambda x: str(x[0]))) + '}'
say('constants E1 (one class): c_k %s r_k %s beta\'_k %s' % (fmt(rep['c_k']), fmt(rep['r_k']), fmt(rep['b_k_val_images'])))
say('constants E1geo: centres %s (train bright frac %.2f) c_k %s r_k(teachers) %s beta\'_k %s | r_k(students, diag) %s | test frames w/o class %d' % (
    np.round(cc, 1).tolist(), frac1, fmt(geo['c_k']), fmt(geo['r_k']), fmt(geo['b_k_val_images']), fmt(stu['r_k']), nocls))

# ---------------------------------------------------------------- 4. ONE test scoring + paired bootstrap
if SCORE:
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    pf = OUT + '/preds/e1geo_fused_test.pkl'
    if not os.path.exists(pf + '.score_real_test.json'):
        with open(OUT + '/score.log', 'w') as lg:
            subprocess.run([PY, LIB + '/score_x.py', pf, 'real_test'], stdout=lg, stderr=subprocess.STDOUT, check=True)
    say('scored test once: %s.score_real_test.json' % pf)
print('OK')
