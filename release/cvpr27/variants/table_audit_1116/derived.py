# Derived numbers (means, SDs, differences, threshold-raised rows, subsets) from the unified rescoring (pack score.py 11-16, stable sort).
import sys, json, pickle, numpy as np
sys.path.insert(0, 'D:/UAV3D_baselines_pack/scripts'); import score as S
G = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/'; R = G + 'review1113/'
W = 'C:/Users/jiang/OneDrive/文档/xwechat_files/wxid_pu1uzzbuzwsx22_b4ed/msg/file/2026-11/results_to_send_20261116_1459/runs/'
SC = {}
for f in ('scan_baselines.json', 'scan_pst.json'):
    SC.update(json.load(open(f, encoding='utf-8')))
def m(p): return SC[p]
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

def tp_mask(P, M, split):
    g = M['gt']; N = len(g); tp = np.zeros(N, bool); d = np.full(N, np.nan)
    for i in range(N):
        if not (np.isfinite(P['pt'][i]).all() and np.isfinite(P['conf'][i])): continue
        eul = P['peul'][i] if np.isfinite(P['peul'][i]).all() else np.zeros(3)
        if S.iou(S.box2d(P['pt'][i], P['plwh'][i], eul, M['K'][i]), S.box2d(g[i, :3], g[i, 3:6], g[i, 6:9], M['K'][i])) >= 0.1:
            tp[i] = True; dd = np.linalg.norm(P['pt'][i] - g[i, :3]); d[i] = dd / np.linalg.norm(g[i, :3]) if split == 'outdoor_test' else dd
    return tp, d

out = {}
def put(k, v):
    out[k] = v
    print('%-55s %s' % (k, ('%.4f' % v) if isinstance(v, float) else json.dumps(v)), flush=True)

K6 = ['AP', 'ATE', 'ASE', 'AOE', 'mDR', 'UDS']
def mean_rows(paths):
    A = np.array([[m(p)[k] for k in K6] for p in paths])
    return dict(zip(K6, A.mean(0).round(4).tolist())), dict(zip(K6, A.std(0, ddof=1).round(4).tolist()))

GROUPS = {
 'MAV GUPNet+PST': [W + 'gupnet_in_kdc_mav_s%d/pred_real_test.pkl' % s for s in range(3)],
 'MAV MonoCD+PST': [W + 'monocd_in_kdc_mav_s%d/pred_real_test.pkl' % s for s in range(3)],
 'MAV MonoDGP+PST': [W + 'monodgp_in_kdc_mav_s%d/pred_real_test.pkl' % s for s in range(3)],
 'MAV MonoCD sim (s1,s2)': [W + 'monocd_in_sim_s%d/pred_real_test.pkl' % s for s in (1, 2)],
 'RD MonoDETR+PST': [W + 'monodetr_out_kd_anti_s%d/pred_outdoor_test.pkl' % s for s in range(3)],
 'RD MonoCD+PST': [W + 'monocd_out_kd_anti_s%d/pred_outdoor_test.pkl' % s for s in range(3)],
 'RD MonoDGP+PST': [W + 'monodgp_out_kd_anti_s%d/pred_outdoor_test.pkl' % s for s in range(3)],
 'RD MonoCoP+PST': [W + 'monocop_out_kd_anti_s%d/pred_outdoor_test.pkl' % s for s in range(3)],
 'RD GUPNet+PST': [W + 'gupnet_out_kd_anti_s%d/pred_outdoor_test.pkl' % s for s in range(3)],
 'MAV PST 3 runs': [G + 'srcx/in2mav/preds/pcmd_fused_test.pkl', R + 'sup3/run/preds/pcmd_fused_test.pkl', R + 'seeds/mav_full_s20/preds/pcmd_fused_test.pkl'],
 'RD PST 3 runs': [G + 'srcx/out2antifs/preds/pcmd_fused_test.pkl', R + 'seeds/antifs_full_s10/preds/pcmd_fused_test.pkl', R + 'seeds/antifs_full_s20/preds/pcmd_fused_test.pkl'],
 'MAV RGBonly 3 runs': [G + 'srcx/in2mav_none/preds/pcmd_fused_test.pkl', R + 'seeds/mav_none_s10/preds/pcmd_fused_test.pkl', R + 'seeds/mav_none_s20/preds/pcmd_fused_test.pkl'],
 'RD RGBonly 3 runs': [G + 'srcx/out2antifs_none/preds/pcmd_fused_test.pkl', R + 'seeds/antifs_none_s10/preds/pcmd_fused_test.pkl', R + 'seeds/antifs_none_s20/preds/pcmd_fused_test.pkl'],
 'MAV SF 3 runs': [R + 'sensorfree/E1/mav_s%d/preds/e1_fused_test.pkl' % s for s in (0, 10, 20)],
 'RD SF 3 runs': [R + 'sensorfree/E1/antifs_s%d/preds/e1_fused_test.pkl' % s for s in (0, 10, 20)],
 'MAV SF+mask (E1b)': [R + 'sensorfree/E1b/mav_s%d/preds/e1b_fused_test.pkl' % s for s in (0, 10, 20)],
 'RD SF+mask (E1b)': [R + 'sensorfree/E1b/antifs_s%d/preds/e1b_fused_test.pkl' % s for s in (0, 10, 20)],
 'MAV SF+classes (E1geo)': [R + 'sensorfree/E1geo/mav_s%d/preds/e1geo_fused_test.pkl' % s for s in (0, 10, 20)],
 'MAV noLiDARstruct H6': [R + 'heads/H6/mav_s%d/preds/h6_fused_test.pkl' % s for s in (0, 10, 20)],
 'MAV noIRint H5': [R + 'heads/H5/mav_s%d/preds/h5_fused_test.pkl' % s for s in (0, 10, 20)],
 'MAV noLiDARbody H4': [R + 'heads/H4/mav_s%d/preds/h4_fused_test.pkl' % s for s in (0, 10, 20)],
 'MAV oneclass H1c': [R + 'heads/H1c/mav_s%d/preds/h1c_fused_test.pkl' % s for s in (0, 10, 20)],
 'MAV boxhull 3 runs': [R + 'boxhull/in2mav/preds/boxhull_fused_test.pkl', R + 'heads/H7/mav_s10/preds/h7_fused_test.pkl', R + 'heads/H7/mav_s20/preds/h7_fused_test.pkl'],
 'RD boxhull 3 runs': [R + 'boxhull/out2antifs/preds/boxhull_fused_test.pkl', R + 'sensorfree/E2/antifs_s10/preds/e2_fused_test.pkl', R + 'sensorfree/E2/antifs_s20/preds/e2_fused_test.pkl'],
 'MAV initabl': [R + 'initabl/s%d/preds/initb_fused_test.pkl' % s for s in (0, 10, 20)],
 'MAV PST nofuse 3 runs': [G + 'srcx/in2mav/preds/pcmd_nofuse_test.pkl', R + 'sup3/run/preds/pcmd_nofuse_test.pkl', R + 'seeds/mav_full_s20/preds/pcmd_nofuse_test.pkl'],
 'MAV IRonly target w0 3 runs': [R + 'clean2/w0/preds/w0_fused_test.pkl', R + 'B8/s10/preds/w0_fused_test.pkl', R + 'B8/s20/preds/w0_fused_test.pkl'],
 'MAV H2 2 runs': [R + 'heads/H2/mav_s0/preds/h2_fused_test.pkl', R + 'heads/H2/mav_s10/preds/h2_fused_test.pkl'],
}
for name, paths in GROUPS.items():
    mu, sd = mean_rows(paths); put(name + ' mean', mu); put(name + ' sd', sd)
    out[name + ' per-run UDS'] = [round(m(p)['UDS'], 4) for p in paths]; print('   per-run UDS', out[name + ' per-run UDS'])

def u(p): return m(p)['UDS']
P0 = G + 'srcx/in2mav/preds/pcmd_fused_test.pkl'; A0 = G + 'srcx/out2antifs/preds/pcmd_fused_test.pkl'
put('MAV PST-RGBselftrain', u(P0) - u(R + 'sensorfree/E4/mav_s0/preds/e4_fused_test.pkl'))
put('RD PST-RGBselftrain', u(A0) - u(R + 'sensorfree/E4/antifs_s0/preds/e4_fused_test.pkl'))
put('MAV RGBonlyT-RGBselftrain', u(G + 'srcx/in2mav_none/preds/pcmd_fused_test.pkl') - u(R + 'sensorfree/E4/mav_s0/preds/e4_fused_test.pkl'))
put('RD RGBonlyT-RGBselftrain', u(G + 'srcx/out2antifs_none/preds/pcmd_fused_test.pkl') - u(R + 'sensorfree/E4/antifs_s0/preds/e4_fused_test.pkl'))
put('MAV SF-PST run2', u(R + 'sensorfree/E1/mav_s10/preds/e1_fused_test.pkl') - u(R + 'sup3/run/preds/pcmd_fused_test.pkl'))
put('RD students depth gain 41.69-38.66', u(G + 'srcx/out2antifs/preds/pcmd_nofuse_test.pkl') - u(R + 'cpu5/anti_c_students_test.pkl'))
put('MAV boxhull s10 diff', u(R + 'heads/H7/mav_s10/preds/h7_fused_test.pkl') - u(R + 'sup3/run/preds/pcmd_fused_test.pkl'))
put('MAV E2a diff', u(R + 'sensorfree/E2a/mav_s0/preds/e2a_fused_test.pkl') - u(P0))
put('MAV E2b diff', u(R + 'sensorfree/E2b/mav_s0/preds/e2b_fused_test.pkl') - u(P0))
put('MAV nogate diff', u(R + 'sup2/nogate/preds/nogate_fused_test.pkl') - u(P0))
put('MAV H1 diff', u(R + 'heads/H1/mav_s0/preds/h1_fused_test.pkl') - u(P0))
put('MAV RGBonlyT run1 gain', u(P0) - u(G + 'srcx/in2mav_none/preds/pcmd_fused_test.pkl'))
put('RD RGBonlyT run1 gain', u(A0) - u(G + 'srcx/out2antifs_none/preds/pcmd_fused_test.pkl'))

# threshold-raised rows: keep the top fraction of frames by PST's own confidence
for name, pred, split, tgt in [('MAV raised to mDR 69.8', P0, 'real_test', 69.8), ('MAV raised to mDR 85.0', P0, 'real_test', 85.0),
                               ('RD raised to mDR 78.5', A0, 'outdoor_test', 78.5), ('RD raised to mDR 85.0', A0, 'outdoor_test', 85.0)]:
    P, M = load(pred, split); c = P['conf']; qs = np.sort(c[np.isfinite(c)])[::-1]; N = len(M['gt']); res = None
    for frac in np.arange(0.70 if split == 'real_test' else 0.75, 0.93, 0.001):
        n = int(round(frac * N))
        if n > len(qs): break
        th = qs[n - 1]; Q = {k: v.copy() for k, v in P.items()}; drop = ~(c >= th)
        for k in ('pt', 'plwh', 'peul'): Q[k][drop] = np.nan
        Q['conf'][drop] = np.nan; r = met(Q, M, split)
        if res is None or abs(r['mDR'] - tgt) < abs(res[1]['mDR'] - tgt): res = (frac, r)
    put(name, dict(keep_pct=round(100 * res[0], 1), **{k: round(v, 3) for k, v in res[1].items()}))

# common-frame ATE (frames detected = 2D IoU>=0.1 by both)
for name, a, b, split in [('MAV common frames', G + 'srcx/in2mav/preds/srconly_test.pkl', P0, 'real_test'),
                          ('RD common frames', G + 'srcx/out2antifs/preds/srconly_test.pkl', A0, 'outdoor_test')]:
    Pa, M = load(a, split); Pb, _ = load(b, split); ta, da = tp_mask(Pa, M, split); tb, db = tp_mask(Pb, M, split)
    both = ta & tb; only = tb & ~ta; sc = 100 if split == 'outdoor_test' else 1
    put(name, dict(n_src_tp=int(ta.sum()), n_both=int(both.sum()), ATE_pst=round(float(sc * np.mean(db[both])), 3), ATE_src=round(float(sc * np.mean(da[both])), 3),
                   n_only=int(only.sum()), ATE_only=round(float(sc * np.mean(db[only])), 3)))

for nm, p in [('PST', A0), ('src', G + 'srcx/out2antifs/preds/srconly_test.pkl'), ('RGBonly', G + 'srcx/out2antifs_none/preds/pcmd_fused_test.pkl')]:
    P, M = load(p, 'outdoor_test'); dist = np.linalg.norm(M['gt'][:, :3], axis=1)
    put('RD dist bins UDS ' + nm, [round(met(P, M, 'outdoor_test', np.where((dist >= lo) & (dist < hi))[0])['UDS'], 2) for lo, hi in ((0, 25), (25, 50), (50, 999))])
    put('RD per-cls UDS ' + nm, {str(c): [int((M['cls'] == c).sum()), round(met(P, M, 'outdoor_test', np.where(M['cls'] == c)[0])['UDS'], 2)] for c in np.unique(M['cls'])})
    fx = M['K'][:, 0, 0]; keep = np.where(np.abs(fx - 1909) > 5)[0]
    put('RD drop fx~1909 ' + nm, dict(n_drop=int(len(fx) - len(keep)), UDS=round(met(P, M, 'outdoor_test', keep)['UDS'], 2)))
for nm, p in [('PST', P0), ('src', G + 'srcx/in2mav/preds/srconly_test.pkl'), ('RGBonly', G + 'srcx/in2mav_none/preds/pcmd_fused_test.pkl')]:
    P, M = load(p, 'real_test'); mdl = np.array([s.split('/')[0] for s in M['seq']])
    put('MAV per-model ' + nm, {k: dict(n=int((mdl == k).sum()), **{kk: round(vv, 2) for kk, vv in met(P, M, 'real_test', np.where(mdl == k)[0]).items()}) for k in np.unique(mdl)})

rng = np.random.default_rng(0)
for nm in ('monocd_out_sim_s0', 'monocd_out_st_anti_s0', 'monocd_out_kd_anti_s0'):
    P, M = load(W + nm + '/pred_outdoor_test.pkl', 'outdoor_test'); us = []
    for t in range(100):
        perm = rng.permutation(len(M['gt'])); Q = {k: v[perm] for k, v in P.items()}
        us.append(met(Q, {'gt': M['gt'][perm], 'K': M['K'][perm]}, 'outdoor_test')['UDS'])
    put('MonoCD 100 random tie orders ' + nm, dict(mean=round(float(np.mean(us)), 2), max=round(float(np.max(us)), 2), min=round(float(np.min(us)), 2)))
json.dump(out, open('derived.json', 'w'), indent=1)
