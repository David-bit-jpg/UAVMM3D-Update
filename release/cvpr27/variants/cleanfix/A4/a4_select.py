# -*- coding: utf-8 -*-
"""[cleanfix A4] Clean replacement of X_suppl.tex:276 ('RGB-pair matched count -0.46 [-1.01,+0.09]'; 'dropping the IR+LiDAR group 16.52 -> 17.06').
Dev originals: gproto/wf24/README.md section 3 (offline, same teacher readings, only the members / check rule change; GT diagnostic only).
Clean inputs (label-free selection; MAV6D train GT read ONLY for the post-hoc label error):
  srcx/in2mav/ou/read_tr1_teach.pkl        6 teachers (irlid0/1, kdir0/1, lid0/1) on the deployed det1 train crops (oc_tr1)
  srcx/in2mav_none/ou/read_tr1_teach.pkl   2 RGB-only sim nets (KZs0 / KZs1 = same recipe, different seed) on the RGB-only control's det1 crops
  D:/S2MUAV3D/srcx_work/in2mav/pseudo_tr1.npz   deployed pseudo labels (6-net label R, d_il, gate) -> reproduction check
Ensembles = ori_x.ensemble (MODULES 0.2 symmetric ensemble), exactly as cmd_pseudo. Label error = 4-fold symmetric geodesic (AOE) vs the nearest
train GT box. Scorable frames = det0 centre inside the projected GT box (figures/data/pl_quality_mav6d.csv det_center_in_gt2d; n 6,047), the
filter behind the paper's 15.4 / 18.8 deg.
Selections: (a) IR<->LiDAR check d_il <= 10 deg (deployed, 4,301 of 12,866); (b) RGB<->RGB agreement (KZs0 vs KZs1), the 4,301 frames with the
smallest disagreement (matched count over all frames); (b') same with the control's own fixed 10-deg rule; (c) random 4,301 (expected value).
Labels: L6 (deployed 6-net label) for every selection [isolates the selection]; also the RGB pair's own label on (b).
Drop group: L4 = ensemble of IR-distilled x2 + LiDAR x2 (irlid pair dropped) vs L6, on the (a)-passing frames.
CIs: sequence-cluster bootstrap (1,000), differences recomputed per resample."""
import csv, json, os, pickle, sys
import numpy as np
os.environ.update(SX_SRC='in', SX_TGT='mav', SX_TEACH='full')
sys.path.insert(0, 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx/lib')
import ori_x as OO  # noqa
from scipy.spatial.transform import Rotation as Rot  # noqa
HERE = os.path.dirname(os.path.abspath(__file__))
H = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/srcx'
WF = 'D:/S2MUAV3D/srcx_work/in2mav'; WN = 'D:/S2MUAV3D/srcx_work/in2mav_none'


def load(W, reads):
    I = pickle.load(open('%s/oc_tr1.pkl' % W, 'rb')); R = pickle.load(open(reads, 'rb'))
    return np.asarray(I['size'], np.float64), I['info'], R


def ens(R, info, lwh, mems):
    out = {}
    for i, f in enumerate(info):
        s = OO.ensemble(R['reads'], mems, i, R['cells'][i], f['K_crop'], lwh)
        if s is not None:
            out[(f['seq'], f['frame'])] = s[0]
    return out


cache = HERE + '/a4_ens.pkl'
if os.path.exists(cache):
    E = pickle.load(open(cache, 'rb'))
else:
    lwh, info, R = load(WF, H + '/in2mav/ou/read_tr1_teach.pkl')
    nm = lambda t: H + '/in2mav/ou/T_%s.pth' % t
    G = {'T6': [nm(x) for x in ('irlid0', 'irlid1', 'kdir0', 'kdir1', 'lid0', 'lid1')], 'IR': [nm('kdir0'), nm('kdir1')], 'LiD': [nm('lid0'), nm('lid1')],
         'L4': [nm(x) for x in ('kdir0', 'kdir1', 'lid0', 'lid1')]}
    E = {g: ens(R, info, lwh, m) for g, m in G.items()}
    lwhn, infon, Rn = load(WN, H + '/in2mav_none/ou/read_tr1_teach.pkl')
    nn = lambda t: H + '/in2mav_none/ou/T_%s.pth' % t
    E.update(rgb0=ens(Rn, infon, lwhn, [nn('rgb0')]), rgb1=ens(Rn, infon, lwhn, [nn('rgb1')]), rgb2=ens(Rn, infon, lwhn, [nn('rgb0'), nn('rgb1')]))
    E['t_pos'] = {(f['seq'], f['frame']): np.asarray(f['box'][:3], np.float64) for f in info}
    pickle.dump(E, open(cache, 'wb'))

# reproduction check against the deployed pseudo labels
P = np.load(WF + '/pseudo_tr1.npz')
keys = [(str(s), str(f)) for s, f in zip(P['seq'], P['frame'])]
d_il = {k: float(v) for k, v in zip(keys, P['d_il'])}
dmax = max(OO.sym(E['T6'][k], P['R'][i]) for i, k in enumerate(keys))
d_ilre = max(abs(OO.sym(E['IR'][k], E['LiD'][k]) - d_il[k]) for k in keys)
print('reproduction: max |R_T6 - pseudo R| %.2e deg, max |d_il - deployed| %.2e deg, n %d, gate %d' % (dmax, d_ilre, len(keys), int(P['gate'].sum())))

# GT (post-hoc only) and scorable filter
I = pickle.load(open('E:/mmcache/mav6d_cn_obj/train/index.pkl', 'rb')); assert I.get('euler_seq', 'xyz') == 'xyz'
GT = {(m['seq'], m['frame']): np.asarray(m['boxes9d'], np.float64).reshape(-1, 9) for m in I['metas'] if m is not None}
scor = {}
with open('D:/CVPR27_UAV/figures/data/pl_quality_mav6d.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        scor[(r['seq'], r['frame'])] = r['det_center_in_gt2d'] in ('True', '1', 'true')


def err(k, R):
    g = GT[k]; j = int(np.argmin(np.linalg.norm(g[:, :3] - E['t_pos'][k], axis=1)))
    return OO.sym(R, Rot.from_euler('xyz', g[j, 6:9]).as_matrix())


rgbk = [k for k in keys if k in E['rgb0'] and k in E['rgb1']]
d_rr = {k: OO.sym(E['rgb0'][k], E['rgb1'][k]) for k in rgbk}
N = int(P['gate'].sum())
selA = {k for k in keys if d_il[k] <= 10.0}
selB = set(sorted(rgbk, key=lambda k: d_rr[k])[:N])
selB10 = {k for k in rgbk if d_rr[k] <= 10.0}
S = [k for k in keys if scor.get(k, False)]
e6 = {k: err(k, E['T6'][k]) for k in S}
e4 = {k: err(k, E['L4'][k]) for k in S if k in E['L4']}
eR = {k: err(k, E['rgb2'][k]) for k in S if k in E['rgb2']}
check = np.mean([e6[k] for k in S if k in selA])
print('check: L6 error on gate-pass scorable %.2f (paper 15.4), all scorable %.2f (paper 18.8)' % (check, np.mean(list(e6.values()))))

seqs = sorted({k[0] for k in S}); bys = {s: [k for k in S if k[0] == s] for s in seqs}


def stats(ks):
    def f(kk):
        kA = [k for k in kk if k in selA]; kB = [k for k in kk if k in selB]; kB10 = [k for k in kk if k in selB10]
        kA4 = [k for k in kA if k in e4]; kBR = [k for k in kB if k in eR]; kB10R = [k for k in kB10 if k in eR]
        m = lambda d, kl: float(np.mean([d[k] for k in kl])) if kl else np.nan
        return dict(A_L6=m(e6, kA), B_L6=m(e6, kB), B10_L6=m(e6, kB10), all_L6=m(e6, kk), B_ownRGB=m(eR, kBR), B10_ownRGB=m(eR, kB10R),
                    A_L4=m(e4, kA4), A_L6_onL4frames=m(e6, kA4), nA=len(kA), nB=len(kB), nB10=len(kB10), n=len(kk))
    return f(ks)


pt = stats(S)
pt['B_minus_A'] = pt['B_L6'] - pt['A_L6']; pt['B10_minus_A'] = pt['B10_L6'] - pt['A_L6']; pt['L4_minus_L6'] = pt['A_L4'] - pt['A_L6_onL4frames']
pt['Bown_minus_A'] = pt['B_ownRGB'] - pt['A_L6']
rng = np.random.default_rng(0); bs = {k: [] for k in ('B_minus_A', 'B10_minus_A', 'L4_minus_L6', 'Bown_minus_A')}
for _ in range(1000):
    kk = [k for s in rng.choice(seqs, len(seqs), replace=True) for k in bys[s]]
    q = stats(kk)
    bs['B_minus_A'].append(q['B_L6'] - q['A_L6']); bs['B10_minus_A'].append(q['B10_L6'] - q['A_L6'])
    bs['L4_minus_L6'].append(q['A_L4'] - q['A_L6_onL4frames']); bs['Bown_minus_A'].append(q['B_ownRGB'] - q['A_L6'])
ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items()}
# random matched-count expectation: = mean over all scorable frames (selection independent of error)
res = dict(n_frames=len(keys), n_gate_all=N, n_rgb_matched_all=len(selB), n_rgb_10deg_all=len(selB10), overlap_A_B_all=len(selA & selB),
           n_scorable=len(S), n_seq=len(seqs), point=pt, ci95_seq_bootstrap=ci, repro_max_R_diff_deg=dmax, repro_max_dil_diff_deg=d_ilre)
json.dump(res, open(HERE + '/a4_result.json', 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
