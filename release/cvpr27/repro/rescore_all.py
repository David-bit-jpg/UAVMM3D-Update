# -*- coding: utf-8 -*-
"""Recompute every number of Tab. 1-3 of the CVPR'27 PST paper (and the main-text numbers that are not table cells)
from the saved prediction files with the pack scorer, and compare with the number printed in the .tex at its printed precision.

CPU only, read-only on every input. Usage:
    D:/Miniconda3/envs/city/python.exe rescore_all.py            -> rescore_report.md, rescore_results.json, manifest.json (next to this file)
Scorer: D:/UAV3D_baselines_pack/scripts/score.py (11-16 stable-sort version): S.metrics, S.CFG['real_test'] / ['outdoor_test'];
the most confident box of each frame; MAV6D meta = pack data/gt/real_test_meta.npz (byte-identical to cmpeval/test_meta.npz),
RealDrone3D meta = sealed antifs_test_meta.npz. RealDrone3D ATE is printed in % of the distance (x100).
Aggregation of a multi-run cell: arithmetic mean of the per-run metric (UDS averaged per run, i.e. mean of per-run UDS);
Tab. 3 'sd' = sample standard deviation (ddof = 1) of the per-run UDS.
"""
import os, sys, json, pickle, hashlib, time, csv
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, HERE); sys.path.insert(0, 'D:/UAV3D_baselines_pack/scripts')
import score as S
from texparse import main_rows, ablation_rows, sensor_rows, PAPER

META = {'mav': ('D:/UAV3D_baselines_pack/data/gt/real_test_meta.npz', 'real_test'),
        'anti': ('D:/S2MUAV3D/real_outdoor_antiuav3d_fs/labels/sealed/antifs_test_meta.npz', 'outdoor_test')}
MM = {k: np.load(v[0], allow_pickle=True) for k, v in META.items()}
KEYS = {k: {(str(s), str(f)): i for i, (s, f) in enumerate(zip(M['seq'], M['frame']))} for k, M in MM.items()}
K6 = ['AP', 'ATE', 'ASE', 'AOE', 'mDR', 'UDS']

# ---------------------------------------------------------------- prediction files
A = 'D:/advisor_results_1117b/runs/'            # advisor server, delivered 11-17 (b)
W = 'C:/Users/jiang/OneDrive/文档/xwechat_files/wxid_pu1uzzbuzwsx22_b4ed/msg/file/2026-11/results_to_send_20261116_1459/runs/'  # advisor delivery 11-16
L = 'D:/cmpfull_runs/'                          # local baseline runs
G = 'E:/Open3DUAVDet/output/camnorm/geo1004/gproto/'
X = G + 'srcx/'; R = G + 'review1113/'
from baseline_runs import BASELINE             # {(row label, tgt): [pred files]}  (discovered by work/match_baselines.py)

RUNS = dict(BASELINE)
RUNS.update({
    ('CenterNet', 'mav'): [R + 'ourdet_baseproto/preds/in2mav_sim_s0_test.pkl'],
    ('CenterNet', 'anti'): [R + 'ourdet_baseproto/preds/out2antifs_sim_s0_test.pkl'],
    ('CenterNet +DANN', 'mav'): [R + 'ourdet_baseproto/preds/in2mav_dann_test.pkl'],
    ('CenterNet +DANN', 'anti'): [R + 'ourdet_baseproto/preds/out2antifs_dann_test.pkl'],
    ('CenterNet +ST', 'mav'): [R + 'ourdet_baseproto/preds/in2mav_st_test.pkl'],
    ('CenterNet +ST', 'anti'): [R + 'ourdet_baseproto/preds/out2antifs_st_test.pkl'],
    ('CenterNet +PST', 'mav'): [R + 'ourdet_baseproto/preds/in2mav_pst_test.pkl'],
    ('CenterNet +PST', 'anti'): [R + 'ourdet_baseproto/preds/out2antifs_pst_test.pkl'],
    ('CenterNet$^*$', 'mav'): [X + 'in2mav/preds/srconly_test.pkl'],
    ('CenterNet$^*$', 'anti'): [X + 'out2antifs_k249/preds/srconly_test.pkl'],      # byte-for-byte same predictions as out2antifs/preds/srconly_test.pkl (copied det0)
    ('CenterNet$^*$, full pipeline of PST', 'mav'): [X + 'in2mav/preds/pcmd_fused_test.pkl'],
    ('CenterNet$^*$, full pipeline of PST', 'anti'): [X + 'out2antifs_k249/preds/pcmd_fused_test.pkl'],
    ('CenterNet, real labels', 'mav'): [R + 'ourdet_baseproto/preds/in2mav_ub_test.pkl'],
    ('CenterNet, real labels', 'anti'): [R + 'ourdet_baseproto/preds/out2antifs_ubzoom_test.pkl'],
})
ABL = {
    'CenterNet$^*$': X + 'in2mav/preds/srconly_test.pkl',
    '+ Step 1, pseudo box': R + 'clean2/buildup/b_det1_test.pkl',
    '+ Step 2, pseudo orientation': R + 'clean2/buildup/c_students_test.pkl',
    '+ Step 3, pseudo depth, height ratio and fusion': X + 'in2mav/preds/pcmd_fused_test.pkl',
    'PST w/o composites in the depth refiner': R + 'heads/H3/mav_s0/preds/h3_fused_test.pkl',
    'PST w/ detection rate of row one': X + 'in2mav/preds/pcmd_fused_test.pkl',   # derived: top-K frames by confidence, K = frames detected by row one
}
SENS = {   # three runs each: shift 0 (reported run), 10, 20
    ('PST', 'mav'): [X + 'in2mav/preds/pcmd_fused_test.pkl', R + 'sup3/run/preds/pcmd_fused_test.pkl', R + 'seeds/mav_full_s20/preds/pcmd_fused_test.pkl'],
    ('PST', 'anti'): [X + 'out2antifs/preds/pcmd_fused_test.pkl', R + 'seeds/antifs_full_s10/preds/pcmd_fused_test.pkl', R + 'seeds/antifs_full_s20/preds/pcmd_fused_test.pkl'],
    ('PST w/o sensors', 'mav'): [R + 'sensorfree/E1/mav_s%d/preds/e1_fused_test.pkl' % s for s in (0, 10, 20)],
    ('PST w/o sensors', 'anti'): [R + 'sensorfree/E1/antifs_s%d/preds/e1_fused_test.pkl' % s for s in (0, 10, 20)],
    ('PST w/o sensors and composites', 'mav'): [R + 'sensorfree/E4/mav_s%d/preds/e4_fused_test.pkl' % s for s in (0, 10, 20)],
    ('PST w/o sensors and composites', 'anti'): [R + 'sensorfree/E4/antifs_s%d/preds/e4_fused_test.pkl' % s for s in (0, 10, 20)],
    ('PST w/ RGB-only teachers', 'mav'): [X + 'in2mav_none/preds/pcmd_fused_test.pkl', R + 'seeds/mav_none_s10/preds/pcmd_fused_test.pkl', R + 'seeds/mav_none_s20/preds/pcmd_fused_test.pkl'],
    ('PST w/ RGB-only teachers', 'anti'): [X + 'out2antifs_none/preds/pcmd_fused_test.pkl', R + 'seeds/antifs_none_s10/preds/pcmd_fused_test.pkl', R + 'seeds/antifs_none_s20/preds/pcmd_fused_test.pkl'],
}
FIGCSV = PAPER + '/figures/data/pl_quality_mav6d.csv'     # written by review1113/figdata/fig_pl_quality.py mav (training frames, post hoc)

_cache, _sha = {}, {}
# Optional: REPRO_PRED_ROOT=<dir> reads every input file from <dir>/<drive letter>/<rest of the path> (the layout of release/cvpr27/repro/preds).
_ROOT = os.environ.get('REPRO_PRED_ROOT')


def rp(p):
    return p if not _ROOT else _ROOT.rstrip('/') + '/' + p.replace(':', '', 1)



def sha(p):
    if p not in _sha:
        h = hashlib.sha256()
        with open(rp(p), 'rb') as f:
            for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
        _sha[p] = h.hexdigest()
    return _sha[p]


def load(p, tgt):
    if (p, tgt) in _cache: return _cache[(p, tgt)]
    key = KEYS[tgt]; N = len(key)
    P = {k: np.full((N, 3), np.nan) for k in ('pt', 'plwh', 'peul')}; P['conf'] = np.full(N, np.nan)
    for r in pickle.load(open(rp(p), 'rb')):
        i = key.get((str(r['seq']), str(r['frame'])))
        b = np.asarray(r['boxes9d'], np.float64).reshape(-1, 9); s = np.asarray(r['scores'], np.float64).reshape(-1)
        if i is None or not len(b): continue
        j = int(np.argmax(s)); P['pt'][i], P['plwh'][i], P['peul'][i], P['conf'][i] = b[j, :3], b[j, 3:6], b[j, 6:9], s[j]
    _cache[(p, tgt)] = P; return P


def met(P, tgt):
    M = MM[tgt]; r = S.metrics(P, M['gt'], M['K'], np.arange(len(M['gt'])), S.CFG[META[tgt][1]])
    r = {k: float(v) for k, v in r.items()}
    if tgt == 'anti': r['ATE'] *= 100          # printed in % of the distance
    return r


def score_file(p, tgt): return met(load(p, tgt), tgt)


def mean_runs(files, tgt):
    per = [score_file(p, tgt) for p in files]
    return {k: float(np.mean([r[k] for r in per])) for k in K6}, per


def fmt(v, s):
    d = len(s.split('.')[1]) if '.' in s else 0
    return ('%.' + str(d) + 'f') % v


CHECKS = []      # (id, location, printed, recomputed_str, recomputed_raw, ok, files, how)


def check(cid, loc, printed, raw, files, how):
    rec = fmt(raw, printed) if isinstance(raw, (int, float, np.floating)) else str(raw)
    if rec == '-0' or rec.startswith('-0.') and float(rec) == 0: rec = rec[1:]
    CHECKS.append(dict(id=cid, where=loc, printed=printed, recomputed=rec, raw=float(raw) if isinstance(raw, (int, float, np.floating)) else raw,
                       ok=(rec == printed), files=files, how=how))
    return raw


def claim(cid, loc, text, ok, detail, files=()):
    CHECKS.append(dict(id=cid, where=loc, printed=text, recomputed=detail, raw=None, ok=bool(ok), files=list(files), how='qualitative claim, checked numerically'))


if __name__ == '__main__':
    t0 = time.time()
    V = {}                         # (row, tgt) -> mean metrics
    # ------------------------------------------------ Tab. 1
    MR = main_rows()
    for (row, tgt), cells in MR.items():
        files = RUNS[(row, tgt)]
        vals, per = mean_runs(files, tgt); V[(row, tgt)] = vals
        how = 'single run' if len(files) == 1 else 'mean over %d runs (per-run metrics, UDS averaged per run)' % len(files)
        for k in K6:
            check('tab1|%s|%s|%s' % (row, tgt, k), 'tab_main.tex', cells[k], vals[k], files, how)
    # ------------------------------------------------ Tab. 2
    AR = ablation_rows(); VA = {}
    for row, cells in AR.items():
        p = ABL[row]
        if row == 'PST w/ detection rate of row one':
            P0 = load(ABL['CenterNet$^*$'], 'mav'); r0 = met(P0, 'mav'); N = len(MM['mav']['gt'])
            Kd = int(round(r0['mDR'] * N / 100))              # frames detected by row one (3349)
            P = load(p, 'mav'); c = P['conf']; order = np.argsort(-np.nan_to_num(c, nan=-np.inf), kind='stable')
            keep = np.zeros(N, bool); keep[order[:Kd]] = True
            Q = {k: v.copy() for k, v in P.items()}
            for k in ('pt', 'plwh', 'peul'): Q[k][~keep] = np.nan
            Q['conf'][~keep] = np.nan
            vals = met(Q, 'mav'); how = 'derived: full pipeline, keep the %d most confident test frames (= frames detected by row one), others dropped' % Kd
            # alternative used by table_audit_1116/raised.py: K = floor(0.698 N) = 3350
            Q2 = {k: v.copy() for k, v in P.items()}; keep2 = np.zeros(N, bool); keep2[order[:3350]] = True
            for k in ('pt', 'plwh', 'peul'): Q2[k][~keep2] = np.nan
            Q2['conf'][~keep2] = np.nan; VA['_K3350'] = met(Q2, 'mav')
        else:
            vals = score_file(p, 'mav'); how = 'single run'
        VA[row] = vals
        for k in K6:
            check('tab2|%s|%s' % (row, k), 'tab_ablation.tex', cells[k], vals[k], [p], how)
    # ------------------------------------------------ Tab. 3
    SR = sensor_rows(); VS = {}
    for (row, tgt), cells in SR.items():
        files = SENS[(row, tgt)]; per = [score_file(p, tgt)['UDS'] for p in files]
        VS[(row, tgt)] = per
        check('tab3|%s|%s|mean' % (row, tgt), 'tab_sensor.tex', cells['mean'], float(np.mean(per)), files, 'mean of per-run UDS over 3 runs (shift 0/10/20)')
        check('tab3|%s|%s|sd' % (row, tgt), 'tab_sensor.tex', cells['sd'], float(np.std(per, ddof=1)), files, 'sample sd (ddof=1) of per-run UDS')
    # ------------------------------------------------ main-text numbers
    DET = ['MonoDETR', 'MonoCD', 'MonoDGP', 'MonoCoP', 'GUPNet', 'CenterNet']
    BL5 = DET[:5]
    g = lambda r, s, t, k: V[(r + s, t)][k]
    fl = lambda r, s, t: sorted(set(RUNS[(r + s, t)]))
    allf = lambda t, steps: sorted({f for d in DET for s in steps for f in RUNS[(d + s, t)]})
    E = 'sec/4_experiments.tex'
    gm = [g(d, ' +PST', 'mav', 'UDS') - g(d, '', 'mav', 'UDS') for d in DET]; ga = [g(d, ' +PST', 'anti', 'UDS') - g(d, '', 'anti', 'UDS') for d in DET]
    check('text|gain PST-sim MAV min', E, '14.5', min(gm), allf('mav', ('', ' +PST')), 'min over 6 detectors of UDS(+PST) - UDS(sim), unrounded means')
    check('text|gain PST-sim MAV max', E, '39.5', max(gm), allf('mav', ('', ' +PST')), 'max, same')
    check('text|gain PST-sim RD min', E, '3.1', min(ga), allf('anti', ('', ' +PST')), 'min over 6 detectors')
    check('text|gain PST-sim RD max', E, '21.0', max(ga), allf('anti', ('', ' +PST')), 'max over 6 detectors')
    ate_sim = [g(d, '', 'mav', 'ATE') for d in BL5]; ate_pst = [g(d, ' +PST', 'mav', 'ATE') for d in BL5]
    check('text|ATE sim 5 baselines min', E, '0.51', min(ate_sim), allf('mav', ('',)), 'min ATE of the 5 baselines, sim only')
    check('text|ATE sim 5 baselines max', E, '1.42', max(ate_sim), allf('mav', ('',)), 'max')
    check('text|ATE PST 5 baselines min', E, '0.28', min(ate_pst), allf('mav', (' +PST',)), 'min ATE of the 5 baselines, +PST')
    check('text|ATE PST 5 baselines max', E, '0.32', max(ate_pst), allf('mav', (' +PST',)), 'max')
    claim('text|ASE of every detector below 3 cm (MAV, +PST)', E, 'ASE < 3 cm for all six +PST rows', all(g(d, ' +PST', 'mav', 'ASE') < 3 for d in DET),
          'max %.2f' % max(g(d, ' +PST', 'mav', 'ASE') for d in DET))
    check('text|min mDR +PST MAV', E, '94.96', min(g(d, ' +PST', 'mav', 'mDR') for d in DET), allf('mav', (' +PST',)), 'min mDR over 6 detectors, +PST')
    claim('text|RD: +PST lowers ATE and AOE for all six', E, 'ATE and AOE lower than sim-only for all six on RD',
          all(g(d, ' +PST', 'anti', k) < g(d, '', 'anti', k) for d in DET for k in ('ATE', 'AOE')), 'checked 12 pairs')
    st_ate = [g(d, ' +ST', 'mav', 'ATE') for d in DET]
    check('text|ST ATE MAV min', E, '0.44', min(st_ate), allf('mav', (' +ST',)), 'min ATE of the six +ST rows')
    check('text|ST ATE MAV max', E, '0.99', max(st_ate), allf('mav', (' +ST',)), 'max')
    dst = [g(d, ' +ST', t, 'UDS') - g(d, '', t, 'UDS') for d in DET for t in ('mav', 'anti')]
    claim('text|ST changes UDS by at most 5', E, '|UDS(+ST) - UDS(sim)| <= 5 for all 12', max(abs(x) for x in dst) <= 5, 'max |change| %.2f' % max(abs(x) for x in dst))
    check('text|ST falls in N of 12', E, '3', float(sum(x < 0 for x in dst)), [], 'count of UDS(+ST) < UDS(sim) over 6 detectors x 2 targets')
    claim('text|DANN lowers UDS for CenterNet', E, 'UDS(+DANN) < UDS(sim) on both targets', all(g('CenterNet', ' +DANN', t, 'UDS') < g('CenterNet', '', t, 'UDS') for t in ('mav', 'anti')), '')
    check('text|GUPNet DANN MAV', E, '22.05', g('GUPNet', ' +DANN', 'mav', 'UDS'), fl('GUPNet', ' +DANN', 'mav'), 'table cell')
    check('text|GUPNet DANN RD', E, '14.34', g('GUPNet', ' +DANN', 'anti', 'UDS'), fl('GUPNet', ' +DANN', 'anti'), 'table cell')
    pm = [g(d, ' +PST', 'mav', 'UDS') - g(d, ' +ST', 'mav', 'UDS') for d in DET]; pa = [g(d, ' +PST', 'anti', 'UDS') - g(d, ' +ST', 'anti', 'UDS') for d in DET]
    check('text|PST-ST MAV min', E, '9.6', min(pm), allf('mav', (' +ST', ' +PST')), 'min over 6 detectors of UDS(+PST) - UDS(+ST)')
    check('text|PST-ST MAV max', E, '39.1', max(pm), allf('mav', (' +ST', ' +PST')), 'max')
    check('text|PST-ST RD min', E, '3.3', min(pa), allf('anti', (' +ST', ' +PST')), 'min')
    check('text|PST-ST RD max', E, '23.1', max(pa), allf('anti', (' +ST', ' +PST')), 'max')
    CS, FP = 'CenterNet$^*$', 'CenterNet$^*$, full pipeline of PST'
    check('text|CenterNet* ASE', E, '1.74', V[(CS, 'mav')]['ASE'], RUNS[(CS, 'mav')], 'table cell')
    check('text|CenterNet ASE', E, '5.92', V[('CenterNet', 'mav')]['ASE'], RUNS[('CenterNet', 'mav')], 'table cell')
    check('text|CenterNet* ATE', E, '0.283', V[(CS, 'mav')]['ATE'], RUNS[(CS, 'mav')], 'table cell')
    check('text|CenterNet ATE', E, '0.666', V[('CenterNet', 'mav')]['ATE'], RUNS[('CenterNet', 'mav')], 'table cell')
    for t, pr in (('mav', '11.3'), ('anti', '8.1')):
        check('text|full pipeline gain %s' % t, E, pr, V[(FP, t)]['UDS'] - V[(CS, t)]['UDS'], RUNS[(FP, t)] + RUNS[(CS, t)], 'UDS(full) - UDS(CenterNet*)')
        check('text|conclusion gain %s' % t, 'sec/5_conclusion.tex', pr, V[(FP, t)]['UDS'] - V[(CS, t)]['UDS'], RUNS[(FP, t)] + RUNS[(CS, t)], 'same')
    check('text|full pipeline UDS MAV', E, '60.81', V[(FP, 'mav')]['UDS'], RUNS[(FP, 'mav')], 'table cell')
    check('text|full pipeline UDS RD', E, '38.69', V[(FP, 'anti')]['UDS'], RUNS[(FP, 'anti')], 'table cell')
    check('text|AP CenterNet* MAV', E, '26.87', V[(CS, 'mav')]['AP'], RUNS[(CS, 'mav')], 'table cell')
    check('text|AP full MAV', E, '40.93', V[(FP, 'mav')]['AP'], RUNS[(FP, 'mav')], 'table cell')
    check('text|AP CenterNet* RD', E, '16.04', V[(CS, 'anti')]['AP'], RUNS[(CS, 'anti')], 'table cell')
    check('text|AP full RD', E, '23.28', V[(FP, 'anti')]['AP'], RUNS[(FP, 'anti')], 'table cell')
    claim('text|full pipeline detects more and lower AOE (both targets)', E, 'mDR up and AOE down vs CenterNet*',
          all(V[(FP, t)]['mDR'] > V[(CS, t)]['mDR'] and V[(FP, t)]['AOE'] < V[(CS, t)]['AOE'] for t in ('mav', 'anti')), '')
    claim('text|full pipeline best UDS of Tab. 1 on RD', E, 'max UDS on RD', V[(FP, 'anti')]['UDS'] >= max(v['UDS'] for (r, t), v in V.items() if t == 'anti'), '')
    check('text|within 4.6 of real labels', E, '4.6', V[('CenterNet, real labels', 'mav')]['UDS'] - V[(FP, 'mav')]['UDS'],
          RUNS[('CenterNet, real labels', 'mav')] + RUNS[(FP, 'mav')], 'UDS(real labels) - UDS(full), MAV6D')
    check('text|MonoCoP +PST MAV UDS', E, '61.90', g('MonoCoP', ' +PST', 'mav', 'UDS'), fl('MonoCoP', ' +PST', 'mav'), 'table cell')
    check('text|MonoDGP +PST MAV UDS', E, '60.55', g('MonoDGP', ' +PST', 'mav', 'UDS'), fl('MonoDGP', ' +PST', 'mav'), 'table cell')
    mu = lambda r, t: float(np.mean(VS[(r, t)]))
    for t, pr, pr1 in (('mav', '4.22', '4.2'), ('anti', '2.47', '2.5')):
        files = SENS[('PST', t)] + SENS[('PST w/o sensors', t)]
        check('text|PST - w/o sensors %s' % t, E, pr, mu('PST', t) - mu('PST w/o sensors', t), files, 'mean UDS(PST) - mean UDS(w/o sensors), 3 runs each')
        check('text|conclusion PST - w/o sensors %s' % t, 'sec/5_conclusion.tex', pr1, mu('PST', t) - mu('PST w/o sensors', t), files, 'same, 1 decimal')
    for t in ('mav', 'anti'):
        j = json.load(open(rp(R + 'sensorfree/E1/pooled_%s_vsPST.json' % ('mav' if t == 'mav' else 'antifs')), encoding='utf-8'))
        ci = j['boot']['UDS']['ci']
        claim('text|95%% interval excludes zero %s (read from saved bootstrap json)' % t, E, 'interval of PST - w/o sensors excludes zero',
              (ci is not None and (min(ci) > 0 or max(ci) < 0)), 'saved: %s' % json.dumps(ci), [R + 'sensorfree/E1/pooled_%s_vsPST.json' % ('mav' if t == 'mav' else 'antifs')])
    claim('text|PST above all controls on both targets', E, 'mean PST > every control mean',
          all(mu('PST', t) > mu(r, t) for t in ('mav', 'anti') for r in ('PST w/o sensors', 'PST w/o sensors and composites', 'PST w/ RGB-only teachers')), '')
    nc = [mu('PST', 'anti') - mu(r, 'anti') for r in ('PST w/o sensors and composites', 'PST w/ RGB-only teachers')]
    check('text|furthest behind on RD (up to)', E, '6.87', max(nc), SENS[('PST', 'anti')] + SENS[('PST w/o sensors and composites', 'anti')] + SENS[('PST w/ RGB-only teachers', 'anti')],
          'max over the two no-composite controls of mean UDS(PST) - mean UDS(control), RD')
    check('text|PST RD all frames', E, '41.71', mu('PST', 'anti'), SENS[('PST', 'anti')], 'Tab. 3 cell')
    check('text|PST RD restricted', E, '38.69', V[(FP, 'anti')]['UDS'], RUNS[(FP, 'anti')], 'Tab. 1 cell')
    T2 = lambda r: VA[r]
    check('text|tab2 mDR row1', E, '69.77', T2('CenterNet$^*$')['mDR'], [ABL['CenterNet$^*$']], 'table cell')
    check('text|tab2 mDR step1', E, '96.23', T2('+ Step 1, pseudo box')['mDR'], [ABL['+ Step 1, pseudo box']], 'table cell')
    check('text|tab2 AOE step2', E, '14.48', T2('+ Step 2, pseudo orientation')['AOE'], [ABL['+ Step 2, pseudo orientation']], 'table cell')
    check('text|tab2 AOE row1', E, '20.32', T2('CenterNet$^*$')['AOE'], [ABL['CenterNet$^*$']], 'table cell')
    check('text|tab2 AP step2', E, '34.04', T2('+ Step 2, pseudo orientation')['AP'], [ABL['+ Step 2, pseudo orientation']], 'table cell')
    check('text|tab2 AP step3', E, '40.93', T2('+ Step 3, pseudo depth, height ratio and fusion')['AP'], [ABL['+ Step 3, pseudo depth, height ratio and fusion']], 'table cell')
    claim('text|full ATE lower on as many frames as CenterNet* detects', E, 'ATE(row 6) < ATE(row 1)',
          T2('PST w/ detection rate of row one')['ATE'] < T2('CenterNet$^*$')['ATE'], '%.3f vs %.3f' % (T2('PST w/ detection rate of row one')['ATE'], T2('CenterNet$^*$')['ATE']))
    check('text|composites: UDS from', E, '60.81', T2('+ Step 3, pseudo depth, height ratio and fusion')['UDS'], [ABL['+ Step 3, pseudo depth, height ratio and fusion']], 'table cell')
    check('text|composites: UDS to', E, '55.32', T2('PST w/o composites in the depth refiner')['UDS'], [ABL['PST w/o composites in the depth refiner']], 'table cell')
    for t, pr in (('mav', '14.5'), ('anti', '12.8')):
        check('text|abstract CenterNet PST-sim %s' % t, 'sec/0_abstract.tex', pr, V[('CenterNet +PST', t)]['UDS'] - V[('CenterNet', t)]['UDS'],
              RUNS[('CenterNet +PST', t)] + RUNS[('CenterNet', t)], 'UDS(CenterNet +PST) - UDS(CenterNet)')
    # Fig. 4 numbers quoted in the text (training-frame pseudo-label analysis; source = the CSV the figure script wrote)
    rows = list(csv.DictReader(open(rp(FIGCSV), encoding='utf-8')))
    f = lambda r, k: float(r[k]) if r[k] not in ('', 'nan') else np.nan
    hp = [r for r in rows if r['has_pseudo'] == 'True']; inb = [r for r in rows if r['det_center_in_gt2d'] == 'True']
    med = lambda rs, k: float(np.nanmedian([abs(f(r, k)) for r in rs]))
    mean = lambda rs, k: float(np.nanmean([f(r, k) for r in rs]))
    check('text|fig4 det depth err', E, '7.6', med(hp, 'det_depth_relerr_pct'), [FIGCSV], 'median |rel. depth error| %, 5,227 pseudo-labelled training frames')
    check('text|fig4 PST depth err', E, '5.9', med(hp, 'teacher_depth_relerr_pct'), [FIGCSV], 'same, PST pseudo box')
    check('text|fig4 RGB-only depth err', E, '8.8', med(hp, 'rgbonly_teacher_depth_relerr_pct'), [FIGCSV], 'same, RGB-only teachers')
    check('text|fig4 det orient err', E, '30.1', mean(inb, 'det_yaw_err_deg'), [FIGCSV], 'mean AOE, frames whose detected centre lies on the UAV (6,047)')
    check('text|fig4 PST orient err', E, '18.8', mean(inb, 'teacher_yaw_err_deg'), [FIGCSV], 'same, PST orientation pseudo label')
    # ------------------------------------------------ outputs
    used = sorted({p for c in CHECKS for p in c['files']})
    man = dict(generated=time.strftime('%Y-%m-%d %H:%M:%S'), scorer='D:/UAV3D_baselines_pack/scripts/score.py', scorer_sha256=sha('D:/UAV3D_baselines_pack/scripts/score.py'),
               metas={k: dict(path=v[0], sha256=sha(v[0])) for k, v in META.items()},
               files={p: dict(sha256=sha(p), size=os.path.getsize(rp(p)), mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(rp(p))))) for p in used},
               cells=CHECKS, extra=dict(tab2_row6_with_K3350=VA.get('_K3350')))
    json.dump(man, open(HERE + '/manifest.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    tab = [c for c in CHECKS if c['id'].startswith('tab')]; txt = [c for c in CHECKS if not c['id'].startswith('tab')]
    L_ = ['# Rescore report (%s, %.0f s)' % (man['generated'], time.time() - t0), '',
          'Table cells: %d checked, %d reproduced exactly at the printed precision.' % (len(tab), sum(c['ok'] for c in tab)),
          'Main-text numbers and claims: %d checked, %d reproduced.' % (len(txt), sum(c['ok'] for c in txt)), '', '## Mismatches', '',
          '| id | printed | recomputed | unrounded | how |', '|---|---|---|---|---|']
    for c in CHECKS:
        if not c['ok']: L_.append('| %s | %s | %s | %s | %s |' % (c['id'], c['printed'], c['recomputed'], c['raw'], c['how']))
    L_ += ['', '## All checks', '', '| id | printed | recomputed | ok |', '|---|---|---|---|']
    for c in CHECKS: L_.append('| %s | %s | %s | %s |' % (c['id'], c['printed'], c['recomputed'], 'yes' if c['ok'] else '**NO**'))
    open(HERE + '/rescore_report.md', 'w', encoding='utf-8').write('\n'.join(L_) + '\n')
    print('\n'.join(L_[:12 + sum(not c['ok'] for c in CHECKS)]))
